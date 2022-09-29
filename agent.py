from copy import deepcopy
import time
from typing import List, Literal, Tuple
from chess_python.chess import Chess

class State:
    def __hash__(self):
        pass

class ClassicEvaluator:
    PIECE_VALUE = {
        0: 0, 1: 1, 2: 3, 3:3, 4:5, 5:9, 6:0,
        -1: -1, -2: -3, -3:-3, -4:-5, -5:-9, -6:0,
    }
    def evaluate(self, state):
        # Simple count
        base_evaluation = sum([self.PIECE_VALUE[piece] for piece in state.board])
        return base_evaluation

def minimax_ab(node:Chess, depth, alpha, beta, maximize):
    """Working"""
    if depth == 0:
        return ClassicEvaluator().evaluate(node.state)
    legal_moves = node.legal_moves()
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
        legal_moves = node.legal_moves()
        for move in legal_moves:
            child = deepcopy(node).move(move, False, True if depth!=1 else False)
            value = min(value, minimax_ab(child, depth-1, alpha, beta, True))
            if value <= alpha:
                break
            beta = min(beta, value)

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
fen = "3qr2k/pbpp2pp/1p5N/3Q2b1/2P1P3/P7/1PP2PPP/R4RK1 w - - 0 1"
chess = Chess(fen)
agent = Agent(color=1)
t0 = time.time()
recommended_moves = agent.recommend(chess, 3, True)
t1 = time.time()
print(recommended_moves[0][0])