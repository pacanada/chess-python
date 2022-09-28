import pytest

from chess_python.chess import _get_index_trajectory


@pytest.fixture
def diagonal_cases():
    return {
        0: {"pos_i": 0, "pos_f": 63, "trajectory": [9, 18, 27, 36, 45, 54]},
        1: {"pos_i": 63, "pos_f": 0, "trajectory": [9, 18, 27, 36, 45, 54]},
        2: {"pos_i": 7, "pos_f": 56, "trajectory": [14, 21, 28, 35, 42, 49]},
        3: {"pos_i": 56, "pos_f": 7, "trajectory": [14, 21, 28, 35, 42, 49]},
    }


def test_indexes(diagonal_cases):
    for _, v in diagonal_cases.items():
        pos_i = v["pos_i"]
        pos_f = v["pos_f"]
        expected = v["trajectory"]
        assert set(_get_index_trajectory(pos_i, pos_f)) == set(
            expected
        ), f"Failed for pos_i {pos_i}, pos_f {pos_f}"
