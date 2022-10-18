import time

import pytest

from chess_python.chess import Chess
from tests.utils import Node, get_size_node


@pytest.fixture
def perft_of_positions():
    # https://www.chessprogramming.org/Perft_Results
    return {
        0: {
            "name": "init_position",
            "n_nodes": 8902,
            "approx_time": 0.05,
            "depth": 3,
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        },
        1: {
            "name": "pos_3",
            "n_nodes": 2812,
            "approx_time": 0.01,
            "depth": 3,
            "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        },
        2: {
            "name": "pos_4",
            "n_nodes": 9467,
            "approx_time": 0.03,
            "depth": 3,
            "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        },
        3: {
            "name": "pos_5",
            "n_nodes": 62379,
            "approx_time": 0.28,
            "depth": 3,
            "fen": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        },
        4: {
            "name": "pos_3",
            "n_nodes": 43238,
            "approx_time": 0.29,
            "depth": 4,
            "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        },
        5: {
            "name": "init_position",
            "n_nodes": 197281,
            "approx_time": 1.23,
            "depth": 4,
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        },
        6: {
            "name": "puzzle",
            "n_nodes": 53640,
            "approx_time": 0.26,
            "depth": 3,
            "fen": "3qr2k/pbpp2pp/1p5N/3Q2b1/2P1P3/P7/1PP2PPP/R4RK1 w - - 0 1",
        },
        7: {
            "name": "bug",
            "n_nodes": 3,
            "depth": 1,
            "approx_time": 0.001,
            "fen": "2N3k1/1P6/1B2P1Bp/7p/8/2b1r3/4R3/4K3 w - - 0 1",
        },
    }


def test_full_suite(perft_of_positions):
    for v in perft_of_positions.values():
        node = Node(game=Chess(fen=v["fen"]), depth=v["depth"], move=None)
        assert get_size_node(node) == v["n_nodes"], "Number of nodes does not match"


def test_quick(perft_of_positions):
    v = perft_of_positions[1]
    node = Node(game=Chess(fen=v["fen"]), depth=v["depth"], move=None)
    assert get_size_node(node) == v["n_nodes"], "Number of nodes does not match"


def test_performance(perft_of_positions):
    v = perft_of_positions[0]
    t1 = time.time()
    Node(game=Chess(fen=v["fen"]), depth=v["depth"], move=None)
    t2 = time.time()
    ref_time = v["approx_time"]
    # to see in the output run  pytest tests/test_tree.py::test_performance  -s
    print(f"Run in {(t2-t1):.2f}s. Reference {ref_time}")
    assert (t2 - t1) <= 1, f"Elapsed time {(t2-t1):.2f} s. Reference {ref_time}"
