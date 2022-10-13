from chess_python.chess import Chess


def test_is_stale_mate():
    game = Chess("2qk1b2/4n2p/5p2/3p4/b1nK4/5r2/7r/8 w - - 0 1")
    game.update_outcome()
    assert game.is_stalemate, "This position should be stalemate"
    assert game.result == 0, "This positions should be a draw"


def test_is_not_stale_mate():
    game = Chess("2qk1b2/4n2p/1n3p2/3p4/b1BK4/5r2/7r/8 b - - 0 1")
    game.update_outcome()
    assert game.is_stalemate is False, "This position should not be stalemate"


def test_checkmate_white():
    game = Chess("8/7k/8/8/8/8/r7/r6K w - - 0 1")
    game.update_outcome()
    assert game.is_checkmate, "This position should  be checkmate"
    assert game.result == -1, "This position should be win for black"


def test_checkmate_black():
    game = Chess("8/7K/8/8/8/8/R7/R6k b - - 0 1")
    game.update_outcome()
    assert game.is_checkmate, "This position should  be checkmate"
    assert game.result == 1, "This position should be win for white"


def test_draw_by_repetition():
    game = Chess()
    game.move("e2e4")
    game.move("e7e5")
    # Start repetition
    game.move("f1e2")
    game.move("f8e7")
    game.move("e2f1")
    game.move("e7f8")
    game.move("f1e2")
    game.move("f8e7")
    game.update_outcome()
    assert (
        game.is_threefold_repetition is False
    ), "This combination of moves should not be draw by repetition"
    assert game.result is None, "This combination of moves should be draw by repetition"
    game.move("e2f1")
    game.move("e7f8")

    game.update_outcome()
    assert game.is_threefold_repetition, "This combination of moves should be draw by repetition"
    assert game.result == 0, "This combination of moves should be draw by repetition"


def test_draw_by_insufficient_material():
    game = Chess("8/7K/8/8/8/8/N7/7k b - - 0 1")
    game.update_outcome()
    assert (
        game.is_draw_by_insufficient_material
    ), "This combination of moves should be draw by insufficient material"
    assert game.result == 0, "This combination of moves should be draw by insufficient material"

    game = Chess("8/7K/8/8/8/8/N7/b6k w - - 0 1")
    game.update_outcome()
    assert (
        game.is_draw_by_insufficient_material
    ), "This combination of moves should be draw by insufficient material"
    assert game.result == 0, "This combination of moves should be draw by insufficient material"

    game = Chess("8/7K/7R/8/8/8/N7/7k b - - 0 1")
    game.update_outcome()
    assert (
        game.is_draw_by_insufficient_material is False
    ), "This combination of moves should not be draw by insufficient material"
    assert game.result is None, "This combination of moves should be draw by insufficient material"
