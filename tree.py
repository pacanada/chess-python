import time
from copy import deepcopy
from typing import Optional

from chess_python.chess import Chess

DICT_POSITIONS = {
    0: {
        "name": "init_position",
        "depth": 3,
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    },
    1: {
        "name": "pos_3",
        "depth": 3,
        "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    },
    2: {
        "name": "pos_4",
        "depth": 3,
        "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    },
    3: {
        "name": "pos_5",
        "depth": 3,
        "fen": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    },
    4: {
        "name": "pos_3",
        "depth": 4,
        "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    },
    5: {
        "name": "init_position",
        "depth": 4,
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    },
}


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
    for k, v in DICT_POSITIONS.items():
        print("Evaluating position {}".format(v["name"]))
        t1 = time.time()
        node = Node(Chess(v["fen"]), v["depth"])
        t2 = time.time()
        print(f"Size of own tree {get_size_node(node)}, Elapsed time {(t2-t1):.2f} s")
        move_combination = get_move_combination(node)
