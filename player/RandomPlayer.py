from player.Node import Node
import numpy as np
import random


class RandomPlayerMixin(object):
    def __init__(self, name):
        self.name = name

    def suggest_move(self, node: Node)->tuple:
        node.expand_node(np.ones((128,)))
        size = len(node.legal_moves)
        selected = random.randint(0, size - 1)
        selected_move = node.legal_moves[selected]

        return selected_move, 0.5
