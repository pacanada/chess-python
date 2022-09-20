from chess_python.chess import Chess

chess = Chess(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0")
print(chess)
print(chess.legal_moves())
chess.move("e2e4")
print(chess)
