from chess_python.chess import Chess, get_allowed_moves, ChessUtils
def main():
    while True:
        inp = input("1. Start game. 2. Show allowed moves. 3 Move. 4. Show board. 5. Exit.")
        if inp == "1":
            print("Starting game")
            fen = input("Enter fen: ")
            if len(fen)<2:
                game = Chess()
            else:
                game = Chess(fen)
        elif inp == "2":
            pos = input("Enter the piece position: ")
            allowed_moves = get_allowed_moves(
                game.state.board,
                 ChessUtils.POSITION_DICT[pos],
                  game.state.en_passant_allowed,
                   game.state.castling_rights,
                   game.state, game.optimizer)
            print("Allowed moves: ", )
            game.print_allowed_moves(allowed_moves=allowed_moves, pos=ChessUtils.POSITION_DICT[pos])
        elif inp == "3":
            move = input("Enter the move: ")
            pos = move[:2]
            #allowed_moves = get_allowed_moves(game.state.board, ChessUtils.POSITION_DICT[pos], game.state.en_passant_allowed, game.state.castling_rights, game.state, game.optimizer)
            game.move(move, True)
            print(game)
        elif inp == "4":
            print(game)
        elif inp == "5":
            print("Exiting")
            break


if __name__ == "__main__":
    main()
