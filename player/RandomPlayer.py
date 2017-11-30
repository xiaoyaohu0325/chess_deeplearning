from player.Node import Node
import numpy as np
import random


class RandomPlayerMixin(object):

    def suggest_move(self, node: Node)->tuple:
        size = len(node.legal_moves)
        selected = random.randint(0, size-1)
        node.expand_node(np.zeros((128,)))

        return node.legal_moves[selected], 0.5
