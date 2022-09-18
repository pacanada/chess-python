import time
from copy import deepcopy
from typing import Optional

from chess_python.chess import Chess

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
depth = 3


class Node:
    def __init__(self, game: Chess, depth: int, move: Optional[str] = None):
        self.move = move
        # no need to copy the entire game for the last branch since we are not doing any move
        game = deepcopy(game) if depth != 0 else game
        self.children = get_children(game, depth, move)


def get_children(game, depth, move):
    if depth == 0:
        return []
    if move is not None:
        game.move(move)
    allowed_moves_i_f = game.legal_moves()
    children = []
    for move in allowed_moves_i_f:
        children.append(Node(game, depth - 1, move))
    return children


def get_size_node(node):
    if len(node.children) == 0:
        return 1
    else:
        return sum([get_size_node(child) for child in node.children])


def get_move_combination(node):
    if len(node.children) == 0:
        return {}
    else:
        return {child.move: get_move_combination(child) for child in node.children}


if __name__ == "__main__":
    t1 = time.time()
    node = Node(Chess(fen), depth)
    t2 = time.time()
    print(f"Size of own tree {get_size_node(node)}, Elapsed time {(t2-t1):.2f} s")
    move_combination = get_move_combination(node)
