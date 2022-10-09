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
    game = Chess("8/8/8/8/8/8/r7/r6K w - - 0 1")
    game.update_outcome()
    assert game.is_checkmate, "This position should  be checkmate"
    assert game.result == -1, "This position should be win for black"


def test_checkmate_black():
    game = Chess("8/8/8/8/8/8/R7/R6k b - - 0 1")
    game.update_outcome()
    assert game.is_checkmate, "This position should  be checkmate"
    assert game.result == 1, "This position should be win for white"
