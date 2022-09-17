from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np

from chess_python.utils import parse_fen

DEBUG = False


class ChessUtils:
    BISHOP_MOVES = np.delete(
        np.array([[i, i] for i in range(-7, 8)] + [[i, -i] for i in range(-7, 8)]),
        (7, 22),
        axis=0,
    )
    ROOK_MOVES = np.delete(
        np.array([[i, 0] for i in range(-7, 8)] + [[0, i] for i in range(-7, 8)]),
        (7, 22),
        axis=0,
    )
    QUEEN_MOVES = np.concatenate((BISHOP_MOVES, ROOK_MOVES))
    MOVE_DIRECTIONS_OFFSET = {
        1: np.array([[1, 0], [1, 1], [1, -1], [2, 0]]),
        -1: np.array([[-1, 0], [-1, -1], [-1, 1], [-2, 0]]),  # inverted pawn moves
        2: np.array(
            [[2, 1], [2, -1], [-2, 1], [-2, -1], [1, -2], [1, 2], [-1, 2], [-1, -2]]
        ),
        3: BISHOP_MOVES,
        4: ROOK_MOVES,
        5: QUEEN_MOVES,
        6: np.array(
            [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        ),
    }
    ATTACKED_OFFSET = MOVE_DIRECTIONS_OFFSET.copy()
    ATTACKED_OFFSET[1] = np.array([[1, 1], [1, -1]])
    ATTACKED_OFFSET[-1] = np.array([[-1, -1], [-1, 1]])
    PIECE_DICT = {
        " ": 0,
        "p": -1,
        "n": -2,
        "b": -3,
        "r": -4,
        "q": -5,
        "k": -6,
        "P": 1,
        "N": 2,
        "B": 3,
        "R": 4,
        "Q": 5,
        "K": 6,
    }
    PIECE_DICT_INV = {v: k for k, v in PIECE_DICT.items()}
    # use nice icons
    PIECE_DICT_INV_UNI = {
        0: " ",
        1: "♙",
        2: "♘",
        3: "♗",
        4: "♖",
        5: "♕",
        6: "♔",
        -1: "♟",
        -2: "♞",
        -3: "♝",
        -4: "♜",
        -5: "♛",
        -6: "♚",
    }
    # {"a1": 0, "a2": 8,..., "b1": 1, "b2": 9, ..., "h8": 63}
    POSITION_DICT = {
        file + str(rank + 1): pos + rank * 8
        for file, pos in zip("abcdefgh", range(64))
        for rank in range(8)
    }
    POSITION_DICT_INV = {v: k for k, v in POSITION_DICT.items()}
    CASTLING_ENCODING = {"Q": 0, "K": 1, "q": 2, "k": 3}
    # castling type, piece_type, allowed_moves (init, final)
    CASTLING_POS = {
        0: {6: [4, 2], 4: [0, 3]},
        1: {6: [4, 6], 4: [7, 5]},
        2: {6: [60, 58], 4: [56, 59]},
        3: {6: [60, 62], 4: [59, 61]},
    }


class State:
    def __init__(self, fen_string: str):
        """Class to keep track of the state of the game."""
        # TODO: castling and on passant can be encoded in a smarter way
        self.fen_string: str = fen_string
        (
            board,
            turn,
            castling_rights,
            en_passant_squares,
            half_move_clock,
            full_move_number,
        ) = parse_fen(fen_string, ChessUtils.PIECE_DICT)
        self.board: np.array = board
        self.turn: int = turn
        self.castling_rights: List[int] = (
            [ChessUtils.CASTLING_ENCODING[cast] for cast in castling_rights]
            if castling_rights != "-"
            else []
        )
        self.en_passant_allowed: list = (
            [
                ChessUtils.POSITION_DICT[en_passant_squares[i : i + 2]]
                for i in range(0, len(en_passant_squares), 2)
            ]
            if en_passant_squares != "-"
            else []
        )
        self.half_move_clock: int = half_move_clock
        self.full_move_number: int = full_move_number


class Optimizer:
    """In this class we keep track of positions attacked by enemy, and pin pieces"""

    def __init__(self, state: State):
        self.attacked_map_dict: Dict[int, np.array] = {}  # convenient function
        self.pin_map_dict: Dict[int, np.array] = {}
        self.pin_map = self.update_pin_map(state)
        self.attacked_map = self.update_attacked_map(state)

    def update(self, state: State):
        self.pin_map = self.update_pin_map(state)
        self.attacked_map = self.update_attacked_map(state)

    def update_pin_map(self, state: State):
        pin_positions = []
        # only the ones that can attack in diagonal or direct more than one square
        king_pos = np.where(state.board == state.turn * 6)[0][0]
        enemy_positions = np.where(
            (state.board * state.turn <= -3) & (state.board * state.turn >= -5)
        )[0]
        # own_positions = np.where(state.board*state.turn>0)[0]
        for pos in enemy_positions:
            if king_pos in get_allowed_moves_by_piece(
                pos, state.board[pos], ChessUtils.MOVE_DIRECTIONS_OFFSET
            ):
                # only if piece can attack king
                index_trajectory = get_index_trajectory(pos_i=pos, pos_f=king_pos)
            else:
                continue

            # if there is only one piece in the way, and is own, append to pin pieces
            is_only_one_piece = sum(state.board[index_trajectory] != 0) == 1
            if is_only_one_piece:
                is_own_piece = sum(state.board[index_trajectory]) * state.turn > 0
                if is_own_piece:
                    # TODO: simplify, ouch
                    pin_pos = np.array(index_trajectory)[
                        state.board[index_trajectory] != 0
                    ][0]
                    pin_positions.append(pin_pos)
                    self.pin_map_dict[pin_pos] = (
                        pos,
                        index_trajectory,
                    )  # enemy which is doing the pin

        return np.array(pin_positions)

    def update_attacked_map(self, state: State) -> np.array:
        # initialize to zero
        self.attacked_map_dict = {}
        attacked_map = np.array([])
        enemy_positions = np.where(state.board * state.turn < 0)[0]
        for pos in enemy_positions:
            piece = state.board[pos]
            direct_attacks = get_direct_attacks(pos, piece, state)
            self.attacked_map_dict[pos] = direct_attacks
            attacked_map = np.concatenate((attacked_map, direct_attacks))
        # probably duplicated
        return attacked_map

    def is_move_legal(self, pos_f: int, pos: int, state: State) -> bool:
        king_pos = np.where(state.board == state.turn * 6)[0][0]
        positions_of_attacking_pieces = self.get_positions_of_attacking_pieces(king_pos)
        is_king_in_check = king_pos in self.attacked_map
        is_piece_a_king = abs(state.board[pos]) == 6
        if is_king_in_check:
            if is_piece_a_king:
                if pos_f in self.attacked_map:
                    # if (pos_f in positions_of_attacking_pieces and len(positions_of_attacking_pieces)<2):
                    #    return True
                    # else:
                    return False
                else:
                    return True
            else:
                if (
                    pos_f in positions_of_attacking_pieces
                    and len(positions_of_attacking_pieces) < 2
                ) and pos not in self.pin_map:
                    return True
                # also where the piece block check block
                if abs(state.board[positions_of_attacking_pieces[0]]) in [3, 4, 5]:
                    # only in the case of bishop, rook or queen attck can be block
                    possible_blocking_trajectories = get_index_trajectory(
                        pos_i=king_pos, pos_f=positions_of_attacking_pieces[0]
                    )
                else:
                    possible_blocking_trajectories = []

                if (
                    len(positions_of_attacking_pieces) < 2
                    and pos_f in possible_blocking_trajectories
                ):
                    return True
                else:
                    return False
        else:
            if is_piece_a_king:
                if pos_f in self.attacked_map:
                    return False
                else:
                    return True
            else:
                if pos in self.pin_map:
                    # if it can move along the direction of the pin yes
                    if pos_f in self.pin_map_dict[pos][1]:
                        return True
                    if pos_f == self.pin_map_dict[pos][0]:
                        # taking enemy piece that is doing the pin
                        return True
                    else:
                        return False
                elif is_en_passant_discovering_check_move(pos_f, pos, state):
                    return False
                else:
                    return True

    def get_positions_of_attacking_pieces(self, king_pos):
        positions_of_attacking_pieces = []
        for pos_0, attacked_pos in self.attacked_map_dict.items():
            if king_pos in attacked_pos:
                positions_of_attacking_pieces.append(pos_0)
        return positions_of_attacking_pieces


def is_en_passant_discovering_check_move(pos_f: int, pos: int, state: State):
    if abs(state.board[pos]) == 1 and pos_f in state.en_passant_allowed:
        # check if king in check

        # 1 remove both pieces
        board_copy = deepcopy(state.board)
        capture_pawn_pos = pos_f + 8 if state.turn == -1 else pos_f - 8
        board_copy[pos] = 0
        board_copy[capture_pawn_pos] = 0
        # 2 and check if king in check
        targets = list(range(24, 32)) if state.turn == -1 else list(range(32, 40))
        if state.turn * 6 in board_copy[targets]:
            king_pos = np.where(board_copy == state.turn * 6)[0][0]
            # get the closest enemy piece
            # which direction to check, direction form the king to capture pwan pos i the only one where there can be a check coming
            direction = 1 if (capture_pawn_pos - king_pos) > 0 else -1
            # check_position = king_pos + direction
            # missing king is attacked
            targets_array = np.array(targets)
            king_index_in_target = np.where(targets_array == king_pos)[0][0]
            sorted_targets = (
                targets_array[king_index_in_target + 1 :]
                if direction == 1
                else np.flip(targets_array[:king_index_in_target])
            )

            for check_position in sorted_targets:
                if board_copy[check_position] * state.turn > 0 or abs(
                    board_copy[check_position]
                ) in [1, 2, 6]:
                    break  # own piece, do not continue, or enemypiece but cannot attack own king
                elif board_copy[check_position] * state.turn == 0:
                    continue
                elif board_copy[check_position] * state.turn < 0 and abs(
                    board_copy[check_position]
                ) in [3, 4, 5]:
                    return True
    return False


def get_direct_attacks(pos, piece, state: State) -> np.array:

    allowed_moves_by_piece = get_allowed_moves_by_piece(
        pos, piece, ChessUtils.ATTACKED_OFFSET
    )
    # When considering the update of attacked pieces, the turn is already updated (see sign)
    if abs(piece) in [3, 4, 5]:
        dig_hor_moves = []
        directions = [
            [0, 1],
            [0, -1],
            [1, 1],
            [1, -1],
            [1, 0],
            [-1, 0],
            [-1, 1],
            [-1, -1],
        ]
        # go that direction until piece is on the way
        for direction in directions:
            pos_2d = (pos // 8 + direction[0], pos % 8 + direction[1])
            pos_1d = pos_2d[0] * 8 + pos_2d[1]
            while True:
                if (pos_2d[1] < 0 or pos_2d[0] < 0) or (pos_2d[1] > 7 or pos_2d[0] > 7):
                    break
                if pos_1d not in allowed_moves_by_piece:
                    break
                elif (
                    state.board[pos_1d] == 0 or state.board[pos_1d] == 6 * state.turn
                ):  # if king is opposite color, the files or rank is still in attack
                    dig_hor_moves.append(pos_1d)
                    pos_1d = pos_1d + direction[0] * 8 + direction[1]
                    continue
                elif state.board[pos_1d] != 0:
                    # blocked by piece, last attacked pos
                    dig_hor_moves.append(pos_1d)
                    break
        attacked_poisitions = dig_hor_moves

    else:
        attacked_poisitions = allowed_moves_by_piece
    return attacked_poisitions


class Chess:
    def __init__(self, fen: str = None):
        self.state = State(fen_string=fen)
        self.optimizer = Optimizer(self.state)
        self.move_combination = []

    def __repr__(self):
        """Nice representation of board state"""
        v = [ChessUtils.PIECE_DICT_INV[piece] for piece in self.state.board]
        repr = self.build_representation(v)

        player_name = "White" if self.state.turn == 1 else "Black"
        return (
            f"Player to move: {player_name}\n"
            + f"Move count: {self.state.full_move_number}\n"
            + repr
        )

    def fen(self):
        return "not implemented"

    def build_representation(self, v):
        """Build representation of board with v values
        v indexes:
        --------------------------------------------
        8 || 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |
        -------------------------------------------
        7 || 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 |
        --------------------------------------------
        6 || 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 |
        -------------------------------------------
        5 || 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 |
        -------------------------------------------
        4 || 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
        -------------------------------------------
        3 || 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
        -------------------------------------------
        2 ||  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
        -------------------------------------------
        1 ||  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
        -------------------------------------------
          ||  a |  b |  c |  d |  e |  f |  g |  h |"""
        hline = "-----------------------------------\n"
        pline = "{} || {} | {} | {} | {} | {} | {} | {} | {} |\n"
        repr = "".join(
            [
                pline.format((8 - index), *v[row - 8 : row]) + hline
                for index, row in enumerate(range(64, 0, -8))
            ]
        )
        repr = repr + hline + "  || a | b | c | d | e | f | g | h\n"
        return repr

    def print_allowed_moves(self, allowed_moves, pos):
        # TODO: duplicated functionality
        board_allowed = np.zeros(64)
        if len(allowed_moves) > 1:
            board_allowed[allowed_moves] = 1
        v = ["O" if sq == 1 else " " for sq in board_allowed]
        v[pos] = ChessUtils.PIECE_DICT_INV[self.state.board[pos]]
        repr = self.build_representation(v)

        print("Allowed moves\n" + repr)

    def move(self, move: Union[str, List[int]], check_allowed_moves: bool = False):

        pos_i, pos_f, promoted_piece = self.convert_move_to_ints(move)
        piece = self.state.board[pos_i]
        piece_color = 1 if piece > 0 else -1
        if check_allowed_moves:
            allowed_moves = get_allowed_moves(
                board=self.state.board,
                pos=pos_i,
                en_passant_allowed=self.state.en_passant_allowed,
                castling_rights=self.state.castling_rights,
                state=self.state,
                optimizer=self.optimizer,
            )
        else:
            allowed_moves = None
        self.validate(pos_i, pos_f, allowed_moves)
        # Realize move
        if promoted_piece is not None:
            self.state.board[pos_f] = (
                ChessUtils.PIECE_DICT[promoted_piece] * -piece_color
            )  # lower case is negative already
        else:
            self.state.board[pos_f] = self.state.board[pos_i]
        self.state.board[pos_i] = 0
        # en passant
        if (pos_f in self.state.en_passant_allowed) and abs(piece) == 1:
            # if movement is en passant, remove the pawn
            self.state.board[pos_f - 8 * piece_color] = 0
        # castling
        if abs(piece) == 6 and (pos_f - pos_i) == 2:  # king side
            # if movement is castling, move also rook
            self.state.board[pos_f - 1] = self.state.board[pos_f + 1]
            self.state.board[pos_f + 1] = 0
        elif abs(piece) == 6 and (pos_f - pos_i) == -2:  # queenside
            # if movement is castling, move also rook
            self.state.board[pos_f + 1] = self.state.board[pos_f - 2]
            self.state.board[pos_f - 2] = 0
        self.state.full_move_number += 1 if self.state.turn == 1 else 0
        # Update state that is not related to the board:
        # 1. turn
        self.state.turn = -self.state.turn
        # 2. en passant allowed
        self.update_en_passant_rights(piece, pos_i, pos_f)

        # 3. castling allowed
        self.update_castling_rights(piece, piece_color, pos_i)

        # 5. and so on
        if self.optimizer is not None:
            self.optimizer.update(self.state)
        # Bonus, keeping track of move combinations for debugging
        self.move_combination.append(move)

        return self

    def update_en_passant_rights(self, piece, pos_i, pos_f):
        self.state.en_passant_allowed = []
        if piece == 1 and (pos_f - pos_i) == 16:
            self.state.en_passant_allowed.append(pos_f - 8)
        elif piece == -1 and (pos_f - pos_i) == -16:
            self.state.en_passant_allowed.append(pos_f + 8)

    def update_castling_rights(self, piece, piece_color, pos_i):
        if abs(piece) == 6:
            if piece_color == 1 and 1 in self.state.castling_rights:
                self.state.castling_rights.remove(1)
            elif piece_color == -1 and 3 in self.state.castling_rights:
                self.state.castling_rights.remove(3)
        elif abs(piece) == 4:
            if piece_color == 1 and pos_i == 63 and 2 in self.state.castling_rights:
                self.state.castling_rights.remove(2)
            elif piece_color == -1 and pos_i == 56 and 0 in self.state.castling_rights:
                self.state.castling_rights.remove(0)

    def convert_move_to_ints(self, move: Union[str, List[int]]) -> Tuple[int]:
        """Convert move to ints"""
        promoted_piece = None
        if len(move) == 5:
            promoted_piece: str = move[4]
            move = move[:4]
        if (
            isinstance(move, List)
            and len(move) == 2
            and max(move) < 64
            and min(move) >= 0
        ):
            pos_i, pos_f = move[0], move[1]
        elif isinstance(move, str) and len(move) == 4:
            pos_i = ChessUtils.POSITION_DICT[move[0:2]]
            pos_f = ChessUtils.POSITION_DICT[move[2:4]]
        else:
            raise ValueError(
                "Invalid move. Must be a list of two ints or a string of length 4: [0,1] corresponds to ´a1a2´ "
            )
        return pos_i, pos_f, promoted_piece

    def validate(self, pos_i, pos_f, allowed_moves):
        if self.state.board[pos_i] == 0:
            raise ValueError("Invalid movement, initial position is empty.")
        color = 1 if self.state.board[pos_i] > 0 else -1
        if color != self.state.turn:
            raise ValueError("Invalid movement, this player cannot move this pice.")
        if DEBUG:
            self.print_allowed_moves(allowed_moves=allowed_moves, pos=pos_i)
        if allowed_moves is not None and pos_f not in allowed_moves:
            raise ValueError(
                f"Invalid movement, piece: {ChessUtils.PIECE_DICT_INV[self.state.board[pos_i]]} cannot move to that position. Allowed moves: {allowed_moves,[ChessUtils.POSITION_DICT_INV[pos] for pos in allowed_moves]}"
            )
        if DEBUG:
            print(
                f"It can move to {ChessUtils.POSITION_DICT_INV[pos_f], pos_f}. From availbale positions: {allowed_moves, [ChessUtils.POSITION_DICT_INV[pos] for pos in allowed_moves]}"
            )


def get_allowed_moves_by_piece(pos, piece, move_directions_offset):
    """Get allowed moves based only on piece related moves."""

    # we have to take into account limits of the board in the side, which complicates things if we use a 1D array for the board
    pos_2d = np.unravel_index(pos, (8, 8))
    empty_board = np.zeros((8, 8))

    piece = piece if abs(piece) == 1 else abs(piece)

    for offset in move_directions_offset[piece]:
        allowed_x = pos_2d[0] + offset[0]
        allowed_y = pos_2d[1] + offset[1]
        if (allowed_x < 0 or allowed_y < 0) or (allowed_x > 7 or allowed_y > 7):
            # out of bounds
            continue

        empty_board[allowed_x, allowed_y] = 1
    return empty_board.ravel().nonzero()[0]


def check_if_positions_are_attacked(optimizer: Optimizer, positions: list):
    """Check if a position is attacked by a piece of the opposite color."""
    for pos in positions:
        if pos in optimizer.attacked_map:
            return True
    return False


def get_allowed_moves(
    board, pos, en_passant_allowed, castling_rights, state, optimizer
):
    piece = board[pos]
    color = 1 if piece > 0 else -1
    # 1
    allowed_moves_by_piece = get_allowed_moves_by_piece(
        pos, piece, ChessUtils.MOVE_DIRECTIONS_OFFSET
    )
    # 2 remove moves where end position is own piece
    allowed_moves = np.array(
        [move for move in allowed_moves_by_piece if board[move] * color <= 0]
    )
    # 3 remove moves where there is a piece in trajectory (only bishop, queen and rook)
    if abs(piece) in [1, 3, 4, 5]:
        blocked_illegal_moves = get_blocked_illegal_moves(board, pos, allowed_moves)
        # remove them
        allowed_moves = np.array(
            [move for move in allowed_moves if move not in blocked_illegal_moves]
        )
    # 4 remove moves for pawns if en passant is not allowed
    if abs(piece) == 1:
        # only move to corners if en passant is allowed or there is a opposite piece there
        allowed_moves = get_pawn_moves(
            board, pos, allowed_moves, color, en_passant_allowed
        )
    # 5 Checks
    # if next possible move is to take king, ilegal move: inefficient but work
    allowed_moves = get_check_illegal_moves_optimized(
        state, pos, allowed_moves, optimizer
    )

    # 6 Castling
    if abs(piece) == 6:
        # in this case we dont want allowe by default the two files move of the king, we treat castling separately
        castle_types_allowed = get_castle_possibilities(
            board, color, castling_rights, optimizer
        )
        castling_positions_allowed = [
            ChessUtils.CASTLING_POS[castle_type_allowed][abs(piece)][1]
            for castle_type_allowed in castle_types_allowed
            if pos == ChessUtils.CASTLING_POS[castle_type_allowed][abs(piece)][0]
        ]
        allowed_moves = np.concatenate(
            (allowed_moves, np.array(castling_positions_allowed))
        )

    return allowed_moves


def get_check_illegal_moves_optimized(
    state: State, pos: int, allowed_moves: list, optimizer: Optimizer
):
    # the whole point is to avoid another depth
    allowed_moves_optimized = [
        pos_f for pos_f in allowed_moves if optimizer.is_move_legal(pos_f, pos, state)
    ]
    return allowed_moves_optimized


def get_castle_possibilities(board, color, castling_rights, optimizer) -> List[int]:
    """Get castle possibilities."""
    allowed_castle_moves = []

    castling_rights_per_color = (
        [cast for cast in castling_rights if cast in [0, 1]]
        if color == 1
        else [cast for cast in castling_rights if cast in [2, 3]]
    )
    for castle_type in castling_rights_per_color:
        if castle_type == 0:  # Queenside
            # space must be empty, king should not be in check and rook not attacked
            if all(
                board[[0, 1, 2, 3, 4]] == np.array([4, 0, 0, 0, 6])
            ) and not check_if_positions_are_attacked(optimizer, [2, 3, 4]):
                allowed_castle_moves.append(0)
        elif castle_type == 1:  # Kingside
            if all(
                board[[4, 5, 6, 7]] == np.array([6, 0, 0, 4])
            ) and not check_if_positions_are_attacked(optimizer, [4, 5, 6]):
                allowed_castle_moves.append(1)

        elif castle_type == 2:
            if all(
                board[[56, 57, 58, 59, 60]] == np.array([-4, 0, 0, 0, -6])
            ) and not check_if_positions_are_attacked(optimizer, [58, 59, 60]):
                allowed_castle_moves.append(2)
        elif castle_type == 3:
            if all(
                board[[60, 61, 62, 63]] == np.array([-6, 0, 0, -4])
            ) and not check_if_positions_are_attacked(optimizer, [60, 61, 62]):
                allowed_castle_moves.append(3)
    return allowed_castle_moves


def get_blocked_illegal_moves(
    board: np.array, pos: int, allowed_moves: np.array
) -> List[int]:
    # TODO: this way of doing it is not efficient, revisit
    get_blocked_illegal_moves = []
    for move in allowed_moves:
        index_trajectory = get_index_trajectory(pos_i=pos, pos_f=move)
        if abs(board[index_trajectory]).sum() > 0:
            get_blocked_illegal_moves.append(move)
    return get_blocked_illegal_moves


def get_pawn_moves(board, pos, allowed_moves, color, en_passant_allowed):
    # TODO: it feels duplicated
    rank = pos // 8
    allowed_push_moves_offset = [8, 16, -8, -16] if rank in [1, 6] else [8, -8]
    # TODO: definetely duplicated
    subset_of_en_passant_allowed = (
        list(range(16, 24)) if color == -1 else list(range(40, 48))
    )
    en_passant_allowed_per_color = [
        move for move in en_passant_allowed if move in subset_of_en_passant_allowed
    ]

    diagonal_moves = [
        pos + diagonal_index
        for diagonal_index in [7, 9, -7, -9]
        if (pos + diagonal_index) in allowed_moves
    ]
    allowed_diagonal_moves = [
        move
        for move in diagonal_moves
        if ((board[move] * color < 0) or (move in en_passant_allowed_per_color))
    ]
    push_moves = [
        pos + push_index
        for push_index in allowed_push_moves_offset
        if (pos + push_index) in allowed_moves
    ]
    allowed_push_moves = [move for move in push_moves if board[move] == 0]
    return np.array(allowed_diagonal_moves + allowed_push_moves)


def get_index_trajectory(pos_i: int, pos_f: int) -> list:
    """Get index of trajectory between two positions"""
    # TODO: is there a simpler way to do this?
    index_trajectory = []
    if pos_i == pos_f:
        return index_trajectory

    # pos_2d_i= np.unravel_index(pos_i, (8,8))
    # same as
    pos_2d_i = (pos_i // 8, pos_i % 8)
    pos_2d_f = (pos_f // 8, pos_f % 8)

    # get direction
    direction = np.array([pos_2d_f[0] - pos_2d_i[0], pos_2d_f[1] - pos_2d_i[1]])
    # make it unit vector
    direction[0] = 0 if direction[0] == 0 else direction[0] / abs(direction[0])
    direction[1] = 0 if direction[1] == 0 else direction[1] / abs(direction[1])

    # get index of trajectory
    index_trajectory.append(pos_i)
    cont = 0
    while index_trajectory[-1] != pos_f:
        index_trajectory.append(index_trajectory[-1] + direction[0] * 8 + direction[1])
        cont += 1
        if cont > 10:
            print("error")
    return index_trajectory[1:-1]


def get_allowed_moves_in_state(state, optimizer=None):
    pieces_positions = np.where(state.board * state.turn > 0)[0]
    allowed_complete_moves = []
    for pos_i in pieces_positions:
        allowed_moves = get_allowed_moves(
            board=state.board,
            pos=pos_i,
            en_passant_allowed=state.en_passant_allowed,
            castling_rights=state.castling_rights,
            state=state,
            optimizer=optimizer,
        )
        for move in allowed_moves:
            allowed_complete_moves.append(
                ChessUtils.POSITION_DICT_INV[pos_i] + ChessUtils.POSITION_DICT_INV[move]
            )

    # moves that could be a promotion
    promoting_moves = [
        move
        for move in allowed_complete_moves
        if (move[1] in ["2", "7"] and move[3] in ["1", "8"])
        and abs(state.board[ChessUtils.POSITION_DICT[move[0:2]]]) == 1
    ]
    allowed_intermediate_moves = [
        move for move in allowed_complete_moves if move not in promoting_moves
    ]

    final_allowed_complete_moves = allowed_intermediate_moves + [
        move + piece_prom
        for move in promoting_moves
        for piece_prom in ["q", "r", "b", "n"]
    ]

    return final_allowed_complete_moves
