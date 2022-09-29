from copy import deepcopy
import time
from typing import List, Literal, Tuple
from chess_python.chess import Chess, ChessUtils

PIECE_VALUES = {
        0: 0, 1: 1, 2: 3, 3:3, 4:5, 5:9, 6:0,
        -1: -1, -2: -3, -3:-3, -4:-5, -5:-9, -6:0,
    }
PIECE_VALUES_ABS = {
        0: 0, 1: 1, 2: 3, 3:3, 4:5, 5:9, 6:0,
    }

MOVE_ORDER = True


class ClassicEvaluator:
    def evaluate(self, state):
        # Simple count
        base_evaluation = sum([PIECE_VALUES[piece] for piece in state.board])
        return base_evaluation

class MoveOrderer:
    """ Priority for captures"""
    def order(self, board, moves: List[str]):
        def rank(move: str):
            # way more things can be done here
            v_end = PIECE_VALUES_ABS[abs(board[ChessUtils.POSITION_DICT[move[2:4]]])] 
            v_ini = PIECE_VALUES_ABS[abs(board[ChessUtils.POSITION_DICT[move[:2]]])]
            if v_end == 0:
                return 0
            else:
                return v_end-v_ini + 8
        moves_ordered = sorted(moves, key=lambda move: rank(move), reverse=True )
        return moves_ordered
# CACHE = {}
def minimax_ab(node:Chess, depth, alpha, beta, maximize):
    """Working"""
    # if node.state in CACHE.keys():
    #     return CACHE[node.state]
     
    if depth == 0:
        return ClassicEvaluator().evaluate(node.state)

    legal_moves = MoveOrderer().order(node.state.board, node.legal_moves())

    if maximize:
        value=-float("inf")
        for move in legal_moves:
            # no need to update the optimizer if is second to last node
            child = deepcopy(node).move(move, False, True if depth!=1 else False)
            value = max(value, minimax_ab(child, depth-1, alpha, beta, False))
            if value >= beta:
                break
            alpha = max(alpha, value)
        return value
    else:
        value=+float("inf")
        for move in legal_moves:
            child = deepcopy(node).move(move, False, True if depth!=1 else False)
            value = min(value, minimax_ab(child, depth-1, alpha, beta, True))
            if value <= alpha:
                break
            beta = min(beta, value)
    # CACHE[node.state] = value
    return value

class Agent:
    """Base class to make a recommendation of best move"""
    def __init__(self, color: Literal[1,-1]):
        self.color = color

    def recommend(self, node: Chess, depth: int, order:bool=False):
        list_moves: List[Tuple[str, float]] = []
        maximize = self.color == -1
        for move in node.legal_moves():
            ti = time.time()
            value = minimax_ab(node=deepcopy(node).move(move), depth=depth, alpha=-float("inf"), beta=float("inf"), maximize=maximize)
            tf = time.time()
            list_moves.append((move, value, f"{(tf-ti):.2f}"))
        if order:
            list_moves = sorted(list_moves, key=lambda item: item[1], reverse=True)

        return list_moves
#fen = "3qr2k/pbpp2pp/1p5N/3Q2b1/2P1P3/P7/1PP2PPP/R4RK1 w - - 0 1"
fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" # answer e2e3
fen = "8/2p5/3p4/KP5r/1R3p2/4P1k1/6P1/8 w - - 1 2" # answer e3f4
fen = "8/8/2pp4/KP5r/1R3P2/6k1/6P1/8 w - - 0 3" # b4b1
fen = "8/8/2pp4/KP5r/5k2/8/6P1/1R6 w - - 0 4"
chess = Chess(fen)
agent = Agent(color=1)
t0 = time.time()
recommended_moves = agent.recommend(chess, 6, True)
t1 = time.time()
print(recommended_moves[0][0], recommended_moves[0][1], t1-t0)
#print(CACHE)