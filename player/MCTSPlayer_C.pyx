import asyncio
from asyncio.queues import Queue
import uvloop
import time
import numpy as np
from collections import namedtuple
import logging
import daiquiri
from util.features import extract_features, bulk_extract_features
from player.Node import Node
from util.strategies import select_weighted_random, select_most_likely

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant
# c_PUCT = 5
# virtual_loss = 3
# cut_off_depth = 30
QueueItem = namedtuple("QueueItem", "feature future")
CounterKey = namedtuple("CounterKey", "board to_play depth")


class MCTSPlayerMixin(object):
    """MCTS Network Player Mix in
    """

    def __init__(self, net, num_playouts=1600):
        self.net = net
        self.now_expanding = set()
        self.expanded = set()
        # queue size should be >= the number of semmphores
        # in order to maxout the coroutines
        # There is not rule of thumbs to choose optimal semmphores
        # And keep in mind: the more coroutines, the less (?) quality (?)
        # of the Monte Carlo Tree obtains. As my searching is less deep
        # w.r.t a sequential MCTS. However, since MCTS is a randomnized
        # algorithm that tries to approximate a value by averaging over run_many
        # random processes, the quality of the search tree is hard to define.
        # It's a trade off among time, accuracy, and the frequency of NN updates.
        self.sem = asyncio.Semaphore(16)
        self.queue = Queue(16)
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        self.playouts = num_playouts  # the more playouts the better
        self.node = None

    """MCTS main functions

       The Asynchronous Policy Value Monte Carlo Tree Search:
       @ Q
       @ suggest_move
       @ suggest_move_mcts
       @ tree_search
       @ start_tree_search
       @ prediction_worker
       @ push_queue
    """

    def suggest_move(self, node: Node, inference=False)->tuple:
        self.node = node
        """Compute move prob"""
        if inference:
            """Use direct NN predition (pretty weak)"""
            move_probs, value = self.run_many(bulk_extract_features([node]))
            move_prob = move_probs[0]
            """Select move"""
            on_board_move_prob = node.predict_to_prob(move_prob)
            # logger.debug(on_board_move_prob)
            if node.n < 30:
                move = select_weighted_random(node, on_board_move_prob)
            else:
                move = select_most_likely(node, on_board_move_prob)
        else:
            """Use MCTS guided by NN"""
            move = self.suggest_move_mcts(node)

        """Get win ratio"""
        player = 'W' if node.to_play else 'B'

        if inference:
            """Use direct NN value prediction (almost always 50/50)"""
            win_rate = value[0, 0] / 2 + 0.5
        else:
            """Use MCTS guided by NN average win ratio"""
            win_rate = node.children[move].Q / 2 + 0.5
        logger.info('Win rate for player {0} is {1:.4f}'.format(player, win_rate))

        return move, win_rate

    def suggest_move_mcts(self, node: Node)->tuple:
        """Async tree search controller"""
        if node.is_game_over():
            return 0

        start = time.time()

        if node.is_leaf():
            logger.debug('Expanding Root Node...')
            move_probs, _ = self.run_many(bulk_extract_features([node]))
            node.expand_node(move_probs[0])

        coroutine_list = []
        for _ in range(self.playouts):
            coroutine_list.append(self.tree_search(node))
        coroutine_list.append(self.prediction_worker())
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))

        logger.debug("Searched for {0:.5f} seconds".format(time.time() - start))
        return node.prune_tree()

    async def tree_search(self, node: Node)->float:
        """Independent MCTS, stands for one simulation"""
        self.running_simulation_num += 1

        # reduce parallel search number
        with await self.sem:
            value = await self.start_tree_search(node)
            node.back_up_value(value)
            # logger.debug("value: {0}".format(value))
            # logger.debug('Current running threads : {0}'.format(RUNNING_SIMULATION_NUM))
            self.running_simulation_num -= 1

            return value

    async def start_tree_search(self, node: Node)->float:
        """Monte Carlo Tree search Select,Expand,Evauate,Backup"""
        if node.is_game_over():
            return 0

        now_expanding = self.now_expanding

        key = self.counter_key(node)

        while key in now_expanding:
            await asyncio.sleep(1e-4)

        if node.is_leaf():
            """is leaf node try evaluate and expand"""
            # add leaf node to expanding list
            self.now_expanding.add(key)

            """Show thinking history for fun"""
            logger.debug("Investigating following position:\n{0}".format(node))

            # perform dihedral manipuation
            features = extract_features(node)

            # push extracted features of leaf node to the evaluation queue
            future = await self.push_queue(features)
            await future
            move_probs, value = future.result()

            # expand by move probabilities
            node.expand_node(move_probs)

            # remove leaf node from expanding list
            self.now_expanding.remove(key)

            # must invert, because alternative layer has opposite objective
            return value[0] * -1
        else:
            """node has already expanded. Enter select phase."""
            # select child node with maximum action scroe
            action_t = node.select_action_by_score()

            child_node = node.children[action_t]

            # add virtual loss
            child_node.virtual_loss_do()
            value = await self.start_tree_search(child_node)  # next move
            child_node.virtual_loss_undo()

            logger.debug("value: {0:.2f} for position: {1}".format(value, node))

            # on returning search path
            # update: N, W, Q, U
            child_node.back_up_value(value)

            # must invert
            return value * -1

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        """
        q = self.queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(1e-3)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            logger.debug("predicting {0} items".format(len(item_list)))
            bulk_features = np.asarray([item.feature for item in item_list])
            policy_ary, value_ary = self.run_many(bulk_features)
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def push_queue(self, features):
        future = self.loop.create_future()
        item = QueueItem(features, future)
        await self.queue.put(item)
        return future

    """MCTS helper functioins
       @ counter_key
       @ run_many
    """
    @staticmethod
    def counter_key(position: Node)->namedtuple:
        if position is None:
            raise ValueError("Can't compress None position into a key!!!")
        return CounterKey(tuple(position.board_array()), position.to_play, position.n)

    #@profile
    def run_many(self, bulk_features):
        return self.net.forward(bulk_features)