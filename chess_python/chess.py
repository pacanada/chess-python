from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numba

from chess_python.utils import parse_fen

class ChessUtils:
    KNIGHT_OFFSETS = [[2, 1], [2, -1], [-2, 1], [-2, -1], [1, -2], [1, 2], [-1, 2], [-1, -2]]
    BISHOP_OFFSETS = [[i, i] for i in range(-7, 8) if i!=0] + [[i, -i] for i in range(-7, 8) if i!=0]
    ROOK_OFFSETS = [[i, 0] for i in range(-7, 8) if i!=0] + [[0, i] for i in range(-7, 8) if i!=0]
    QUEEN_OFFSETS = BISHOP_OFFSETS + ROOK_OFFSETS
    KING_OFFSETS = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    MOVE_DIRECTIONS_OFFSET = {
        1: [[1, 0], [1, 1], [1, -1], [2, 0]],
        -1: [[-1, 0], [-1, -1], [-1, 1], [-2, 0]],  # inverted pawn moves
        2: KNIGHT_OFFSETS,
        -2: KNIGHT_OFFSETS,
        3: BISHOP_OFFSETS,
        -3: BISHOP_OFFSETS,
        4: ROOK_OFFSETS,
        -4: ROOK_OFFSETS,
        5: QUEEN_OFFSETS,
        -5: QUEEN_OFFSETS,
        6: KING_OFFSETS,
        -6: KING_OFFSETS
    }
    ATTACKED_OFFSET = MOVE_DIRECTIONS_OFFSET.copy()
    ATTACKED_OFFSET[1] = [[1, 1], [1, -1]]
    ATTACKED_OFFSET[-1] = [[-1, -1], [-1, 1]]
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
    CASTLING_PER_COLOR = {1: [0, 1], -1: [2, 3]}
    # castling type, piece_type, allowed_moves (init, final)
    CASTLING_POS = {
        0: {6: [4, 2], 4: [0, 3]},
        1: {6: [4, 6], 4: [7, 5]},
        2: {6: [60, 58], 4: [56, 59]},
        3: {6: [60, 62], 4: [59, 61]},
    }
    CASTLING_UTILS = {
        0: {
            "square_indexes": [0, 1, 2, 3, 4],
            "squares_layout": [4, 0, 0, 0, 6],
            "positions_should_not_attacked": [2, 3, 4],
        },
        1: {
            "square_indexes": [4, 5, 6, 7],
            "squares_layout": [6, 0, 0, 4],
            "positions_should_not_attacked": [4, 5, 6],
        },
        2: {
            "square_indexes": [56, 57, 58, 59, 60],
            "squares_layout": [-4, 0, 0, 0, -6],
            "positions_should_not_attacked": [58, 59, 60],
        },
        3: {
            "square_indexes": [60, 61, 62, 63],
            "squares_layout": [-6, 0, 0, -4],
            "positions_should_not_attacked": [60, 61, 62],
        },
    }


class State:
    def __init__(self, fen_string: Optional[str]):
        """Class to keep track of the state of the game."""
        # TODO: castling and on passant can be encoded in a smarter way
        self.fen_string = fen_string
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

    def __deepcopy__(self, memo: Dict[int, object]):
        state = type(self)(None)
        state.board = self.board.copy()
        state.turn = self.turn
        state.castling_rights = deepcopy(self.castling_rights)
        state.en_passant_allowed = self.en_passant_allowed
        state.half_move_clock = self.half_move_clock
        state.full_move_number = self.full_move_number
        return state


class Optimizer:
    """In this class we keep track of positions attacked by enemy, and pin pieces"""

    def __init__(self, state: State):
        self.attacked_map_dict: Dict[int, np.array] = {}  # convenient function
        self.pin_map_dict: Dict[int, np.array] = {}
        self.pin_map = self.update_pin_map(state)
        self.attacked_map = self.update_attacked_map(state)
        self.positions_atacking_king = self.get_positions_of_attacking_pieces(
            np.where(state.board == state.turn * 6)[0][0]
        )

    def update(self, state: State):
        self.pin_map = self.update_pin_map(state)
        self.attacked_map = self.update_attacked_map(state)
        self.positions_atacking_king = self.get_positions_of_attacking_pieces(
            np.where(state.board == state.turn * 6)[0][0]
        )

    def update_pin_map(self, state: State):
        pin_positions = []
        # only the ones that can attack in diagonal or direct more than one square
        king_pos = np.where(state.board == state.turn * 6)[0][0]
        enemy_positions = np.where(
            (state.board * state.turn <= -3) & (state.board * state.turn >= -5)
        )[0]
        for pos in enemy_positions:
            if king_pos in get_allowed_moves_by_piece(
                pos, ChessUtils.MOVE_DIRECTIONS_OFFSET[state.board[pos]]
            ):
                # only if piece can attack king
                index_trajectory = get_index_trajectory(pos_i=pos, pos_f=king_pos)
            else:
                continue

            # if there is only one piece in the way, and is own, append to pin pieces
            is_only_one_piece = sum(state.board[index_trajectory] != 0) == 1
            is_two_opposite_pawns_in_rank_45 = king_pos // 8 in [3,4] and sum(state.board[index_trajectory])==0 and sum(abs(state.board[index_trajectory]))==2
            is_own_piece = sum(state.board[index_trajectory]) * state.turn > 0
            if is_two_opposite_pawns_in_rank_45:
                # pinning en passant target pawn as well (to be used in is_en_passant_discover_check)
                pin_pos = np.array(index_trajectory)[
                        state.board[index_trajectory]*state.turn == -1
                    ][0]
                pin_positions.append(pin_pos)
                # enemy which is doing the pin
                self.pin_map_dict[pin_pos] = (pos, index_trajectory)

            if is_only_one_piece and is_own_piece:
                pin_pos = np.array(index_trajectory)[
                    state.board[index_trajectory] != 0
                ][0]
                pin_positions.append(pin_pos)
                # enemy which is doing the pin
                self.pin_map_dict[pin_pos] = (pos, index_trajectory)

        return pin_positions

    def update_attacked_map(self, state: State) -> np.array:
        # initialize to zero
        self.attacked_map_dict = {}

        attacked_map: List[List[int]] = []
        enemy_positions = np.where(state.board * state.turn < 0)[0]
        for pos in enemy_positions:
            piece = state.board[pos]
            direct_attacks = get_direct_attacks(pos, piece, state)
            self.attacked_map_dict[pos] = direct_attacks
            attacked_map = attacked_map + direct_attacks
        # probably duplicated
        return attacked_map

    def is_move_legal(self, pos_i: int, pos_f: int, state: State) -> bool:  # noqa C901
        # TODO: reduce cyclomatic complexity
        king_pos = np.where(state.board == state.turn * 6)[0][0]
        is_king_in_check = king_pos in self.attacked_map
        is_piece_a_king = abs(state.board[pos_i]) == 6
        if is_king_in_check:
            if is_piece_a_king:
                return pos_f not in self.attacked_map
            else:
                if (
                    pos_f in self.positions_atacking_king
                    and len(self.positions_atacking_king) < 2
                ) and pos_i not in self.pin_map:
                    return True
                # also where the piece block check block
                if abs(state.board[self.positions_atacking_king[0]]) in [3, 4, 5]:
                    # only in the case of bishop, rook or queen attck can be block
                    possible_blocking_trajectories = get_index_trajectory(
                        pos_i=king_pos, pos_f=self.positions_atacking_king[0]
                    )
                else:
                    possible_blocking_trajectories = []

                if (
                    len(self.positions_atacking_king) < 2
                    and pos_f in possible_blocking_trajectories
                ):
                    return True
                else:
                    return False
        else:
            if is_piece_a_king:
                return pos_f not in self.attacked_map
            else:
                if pos_i in self.pin_map:
                    # if it can move along the direction of the pin yes
                    if pos_f in self.pin_map_dict[pos_i][1]:
                        return True
                    if pos_f == self.pin_map_dict[pos_i][0]:
                        # taking enemy piece that is doing the pin
                        return True
                    else:
                        return False
                elif is_en_passant_discovering_check_move(pos_i, pos_f, state, self.pin_map):
                    return False
                else:
                    return True

    def get_positions_of_attacking_pieces(self, target_pos: int):
        return [
            pos
            for pos in self.attacked_map_dict.keys()
            if target_pos in self.attacked_map_dict[pos]
        ]


def is_en_passant_discovering_check_move(pos_i: int, pos_f: int, state: State, pin_map):
    capture_pawn_pos = pos_f + 8 if state.turn == -1 else pos_f - 8
    return abs(state.board[pos_i]) == 1 and pos_f in state.en_passant_allowed and capture_pawn_pos in pin_map


def get_direct_attacks(pos:int, piece:int, state: State) -> List[int]:
    # TODO: get directs attacks is similar to get_allowed_moves_by_piece, reuse somehow?

    allowed_moves_by_piece = get_allowed_moves_by_piece(
        pos, ChessUtils.ATTACKED_OFFSET[piece]
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
        attacked_positions = dig_hor_moves

    else:
        attacked_positions = allowed_moves_by_piece
    return attacked_positions


class Chess:
    def __init__(self, fen: Optional[str] = None, run_optimizer=True):
        self.state = State(fen_string=fen)
        self.optimizer = Optimizer(self.state) if run_optimizer else None
        self.move_combination: List[str] = []

    def __deepcopy__(self, memo: Dict[int, object]):
        """Creates a deepcopy of the board."""
        chess = type(self)(None, False)
        chess.state = deepcopy(self.state)
        chess.optimizer = self.optimizer
        chess.move_combination = self.move_combination
        return chess

    def __repr__(self):
        """Nice representation of board state"""
        v = [ChessUtils.PIECE_DICT_INV[piece] for piece in self.state.board]
        repr_str = self.build_representation(v)

        player_name = "White" if self.state.turn == 1 else "Black"
        return (
            f"Player to move: {player_name}\n"
            + f"Move count: {self.state.full_move_number}\n"
            + repr_str
        )

    def legal_moves(self):
        return get_allowed_moves_in_state(self.state, self.optimizer)

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
        repr_str = "".join(
            [
                pline.format((8 - index), *v[row - 8 : row]) + hline
                for index, row in enumerate(range(64, 0, -8))
            ]
        )
        repr_str = repr_str + hline + "  || a | b | c | d | e | f | g | h\n"
        return repr_str

    def print_allowed_moves(self, allowed_moves, pos):
        # TODO: duplicated functionality
        board_allowed = np.zeros(64)
        if len(allowed_moves) > 1:
            board_allowed[allowed_moves] = 1
        v = ["O" if sq == 1 else " " for sq in board_allowed]
        v[pos] = ChessUtils.PIECE_DICT_INV[self.state.board[pos]]
        repr_str = self.build_representation(v)

        print("Allowed moves\n" + repr_str)

    def move(self, move: str, check_allowed_moves: bool = False):

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

    def convert_move_to_ints(
        self, move: Union[str, List]
    ) -> Tuple[int, int, Optional[str]]:
        """Convert move to ints"""

        if len(move) == 5:
            # TODO: consistency for types in this var
            promoted_piece: Optional[str] = move[4]
            move = move[:4]
        else:
            promoted_piece = None

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
        if allowed_moves is not None and pos_f not in allowed_moves:
            raise ValueError(
                f"Invalid movement, piece: {ChessUtils.PIECE_DICT_INV[self.state.board[pos_i]]} cannot move to that position. Allowed moves: {allowed_moves,[ChessUtils.POSITION_DICT_INV[pos] for pos in allowed_moves]}"
            )
def get_allowed_moves_by_piece(pos: int, offsets: List[List[int]]):
    """Get allowed moves based only on piece related moves."""
    # TODO: should probably optimize this, since it is called many times

    # we have to take into account limits of the board in the side, which complicates things if we use a 1D array for the board
    pos_2d = (pos // 8, pos % 8)
    list_allowed: List[int] = []

    for offset in offsets:
        allowed_x = pos_2d[0] + offset[0]
        allowed_y = pos_2d[1] + offset[1]
        if (allowed_x < 0 or allowed_y < 0) or (allowed_x > 7 or allowed_y > 7):
            # out of bounds
            continue
        # transform to 1d
        list_allowed.append(allowed_x*8+allowed_y)

    return list_allowed


def check_if_positions_are_attacked(attacked_map: List, positions: list):
    """Check if a position is attacked by a piece of the opposite color."""
    return any(pos in attacked_map  for pos in positions)


def get_allowed_moves(
    board, pos, en_passant_allowed, castling_rights, state, optimizer
):
    piece = board[pos]
    color = 1 if piece > 0 else -1
    # 1
    allowed_moves_by_piece = get_allowed_moves_by_piece(
        pos, ChessUtils.MOVE_DIRECTIONS_OFFSET[piece]
    )
    # 2 remove moves where end position is own piece
    allowed_moves = [
        move for move in allowed_moves_by_piece if board[move] * color <= 0
    ]
    # 3 remove moves where there is a piece in trajectory (only bishop, queen and rook)
    if abs(piece) in [1, 3, 4, 5]:
        blocked_illegal_moves = get_blocked_illegal_moves(board, pos, allowed_moves)
        # remove them
        allowed_moves = [
            move for move in allowed_moves if move not in blocked_illegal_moves
        ]
    # 4 remove moves for pawns if en passant is not allowed
    if abs(piece) == 1:
        # only move to corners if en passant is allowed or there is a opposite piece there
        allowed_moves = get_pawn_moves(
            board, pos, allowed_moves, color, en_passant_allowed
        )
    # 5 Checks
    # if next possible move is to take king, ilegal move: inefficient but work
    allowed_moves = get_check_illegal_moves(
        state, pos, allowed_moves, optimizer
    )

    # 6 Castling
    if abs(piece) == 6:
        # in this case we dont want allowe by default the two files move of the king, we treat castling separately
        # TODO: probably duplicated with get_castl_possibilities?
        castle_types_allowed = get_castle_possibilities(
            board, color, castling_rights, optimizer
        )
        castling_positions_allowed = [
            ChessUtils.CASTLING_POS[castle_type_allowed][abs(piece)][1]
            for castle_type_allowed in castle_types_allowed
            if pos == ChessUtils.CASTLING_POS[castle_type_allowed][abs(piece)][0]
        ]
        allowed_moves = allowed_moves + castling_positions_allowed

    return allowed_moves


def get_check_illegal_moves(
    state: State, pos: int, allowed_moves: list, optimizer: Optimizer
):
    # the whole point is to avoid another depth
    allowed_moves_optimized = [
        pos_f for pos_f in allowed_moves if optimizer.is_move_legal(pos_i=pos, pos_f=pos_f, state=state)
    ]
    return allowed_moves_optimized


def get_castle_possibilities(board, color, castling_rights, optimizer) -> List[int]:
    """Get castle possibilities."""
    allowed_castle_moves = []

    castling_rights_per_color = [
        cast for cast in castling_rights if cast in ChessUtils.CASTLING_PER_COLOR[color]
    ]
    for castle_type in castling_rights_per_color:
        square_indexes = ChessUtils.CASTLING_UTILS[castle_type]["square_indexes"]
        squares_layout = ChessUtils.CASTLING_UTILS[castle_type]["squares_layout"]
        positions = ChessUtils.CASTLING_UTILS[castle_type][
            "positions_should_not_attacked"
        ]
        if all(
            board[square_indexes] == squares_layout
        ) and not check_if_positions_are_attacked(optimizer.attacked_map, positions):
            allowed_castle_moves.append(castle_type)

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
    return allowed_diagonal_moves + allowed_push_moves

def get_index_trajectory(pos_i: int, pos_f: int) -> List[int]:
    """Get index of trajectory between two positions"""
    # TODO: there is no check if trajectory can be computed
    # Not super readable but way faster way of doing it
    if pos_i % 8 == pos_f % 8:
        # vertical
        return list(range(pos_i, pos_f, 8*(1 if pos_i<pos_f else -1)))[1:]
    elif pos_i // 8 == pos_f // 8:
        # horizontal
        return list(range(pos_i, pos_f, 1*(1 if pos_i<pos_f else -1)))[1:]
    else:
        # diagonal
        if pos_i % 8 > pos_f % 8:
            return list(range(pos_i, pos_f, 7 if pos_i<pos_f else -9))[1:]
        else:
            return list(range(pos_i, pos_f, 9 if pos_i<pos_f else -7))[1:]
        
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
