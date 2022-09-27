from typing import List, Optional, Tuple


def parse_fen(
    fen_string: Optional[str], piece_dict: dict
) -> Tuple[List[int], int, str, str, int, int]:
    """Parse a FEN string to a board, turn, castling rights, en passant square, half move clock, full move number.
    fen_string=rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1.
    """
    if fen_string is None:
        # default
        fen_string = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0"
    board = [0] * 64
    # Split the string into parts
    parts = fen_string.split(" ")
    # Parse the board
    board_string = parts[0]
    board_string_split = board_string.split("/")
    for j, rank in enumerate(board_string_split):
        offset = 0
        for square in rank:
            if square.isdigit():
                i0 = 64 - (j + 1) * 8 + (offset)
                i1 = offset + int(square)
                board[i0:i1] = [0] * (i1 - i0)
                offset += int(square)
            else:
                board[64 - (j + 1) * 8 + offset] = piece_dict[square]
                offset += 1
    # Parse the turn
    if parts[1] == "w":
        turn = 1
    elif parts[1] == "b":
        turn = -1
    # Parse the castling rights
    castling_rights = parts[2]
    # Parse the en passant square
    en_passant_square = parts[3]
    # Parse the half move clock
    half_move_clock = int(parts[4]) if len(parts[4]) != 0 else 0
    # Parse the full move number
    full_move_number = int(parts[5]) if len(parts) == 6 else 0
    # print(board, turn, castling_rights, en_passant_square, half_move_clock, full_move_number)
    return (
        board,
        turn,
        castling_rights,
        en_passant_square,
        half_move_clock,
        full_move_number,
    )
