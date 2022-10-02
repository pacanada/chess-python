import time
from copy import deepcopy
from typing import Dict, List, Literal, Tuple

from chess_python.chess import Chess, ChessUtils

PIECE_VALUES = {
    0: 0,
    1: 1,
    2: 3,
    3: 3,
    4: 5,
    5: 9,
    6: 0,
    -1: -1,
    -2: -3,
    -3: -3,
    -4: -5,
    -5: -9,
    -6: 0,
}
PIECE_VALUES_ABS = {
    0: 0,
    1: 1,
    2: 3,
    3: 3,
    4: 5,
    5: 9,
    6: 0,
}


class ClassicEvaluator:
    def evaluate(self, state):
        # Simple count
        base_evaluation = sum([PIECE_VALUES[piece] for piece in state.board])
        return base_evaluation


class MoveOrderer:
    """Priority for captures"""

    def order(self, board: List[int], moves: List[str]):
        def rank(move: str):
            # way more things can be done here
            v_end = PIECE_VALUES_ABS[abs(board[ChessUtils.POSITION_DICT[move[2:4]]])]
            v_ini = PIECE_VALUES_ABS[abs(board[ChessUtils.POSITION_DICT[move[:2]]])]
            if v_end == 0:
                return 0
            else:
                return v_end - v_ini + 10

        moves_ordered = sorted(moves, key=lambda move: rank(move), reverse=True)
        return moves_ordered


class Agent:
    """Base class to make a recommendation of best move"""

    def __init__(self, depth: int, color: Literal[1, -1], alpha_beta:bool = True, move_ordering:bool=True, use_transpositions:bool=False):
        self.color = color
        self.nodes_visited = 0
        self.alpha_beta = alpha_beta
        self.move_ordering = move_ordering 
        self.use_transpositions = use_transpositions
        self.depth = depth
        self.transpositions: Dict[int, Dict[int, float]]  = {i:{} for i in range(depth+1)}
        self.transpositions_found=0
        

    def minimax(self, node: Chess, depth, maximize):
        if depth == 0:
            self.nodes_visited+=1
            return ClassicEvaluator().evaluate(node.state)
        if maximize:
            maxeva = -float("inf")
            for move in node.legal_moves():
                child = deepcopy(node).move(move)
                eva = self.minimax(child, depth - 1, False)
                maxeva = max(eva, maxeva)
            return maxeva
        else:
            mineva = +float("inf")
            for move in node.legal_moves():
                child = deepcopy(node).move(move)
                eva = self.minimax(child, depth - 1, True)
                mineva = min(eva, mineva)

            return mineva

    def minimax_ab(self, node: Chess, depth, alpha, beta, maximize):
        """Working. As class method to keep track of metrics"""
        if self.use_transpositions:
            node_hash = hash(node.state)
            if node_hash in self.transpositions[depth].keys():
                self.transpositions_found +=1
                return self.transpositions[depth][node_hash]

        if depth == 0:
            self.nodes_visited+=1
            value = ClassicEvaluator().evaluate(node.state)
            if self.use_transpositions:
                # not working properly: is the hash dimension not enough?
                self.transpositions[0][hash(node.state)]=value
            return value

        if self.move_ordering:
            legal_moves = MoveOrderer().order(board=node.state.board, moves=node.legal_moves())
        else:
            legal_moves = node.legal_moves()

        if maximize:
            value = -float("inf")
            node_hash = hash(hash(node.state))
            for move in legal_moves:

                # no need to update the optimizer if is second to last node
                child = deepcopy(node).move(move, False, True if depth != 1 else False)
                value = max(value, self.minimax_ab(child, depth - 1, alpha, beta, False))
                if value >= beta:
                    break
                alpha = max(alpha, value)

        else:
            value = +float("inf")
            node_hash = hash(hash(node.state))
            for move in legal_moves:
                child = deepcopy(node).move(move, False, True if depth != 1 else False)
                value = min(value, self.minimax_ab(child, depth - 1, alpha, beta, True))
                if value <= alpha:
                    break
                beta = min(beta, value)
        # caching
        if self.use_transpositions:
            self.transpositions[depth][node_hash]=value
        return value

    def recommend(self, node: Chess, order: bool = False):
        list_moves: List[Tuple[str, float, str]] = []
        if self.move_ordering:
            legal_moves = MoveOrderer().order(board=node.state.board, moves=node.legal_moves())
        else:
            legal_moves = node.legal_moves()
        maximize = self.color == -1
        for move in legal_moves:
            ti = time.time()
            if self.alpha_beta:
            
                value = self.minimax_ab(
                    node=deepcopy(node).move(move),
                    depth=self.depth,
                    alpha=-float("inf"),
                    beta=float("inf"),
                    maximize=maximize,
                )
            else:
                value = self.minimax(node=deepcopy(node).move(move), depth=self.depth, maximize=maximize)
            tf = time.time()
            list_moves.append((move, value, f"{(tf-ti):.2f}"))
        if order:
            list_moves = sorted(list_moves, key=lambda item: item[1], reverse=self.color == 1)

        return list_moves


# fen = "3qr2k/pbpp2pp/1p5N/3Q2b1/2P1P3/P7/1PP2PPP/R4RK1 w - - 0 1"
# fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" # answer e2e3
# fen = "8/2p5/3p4/KP5r/1R3p2/4P1k1/6P1/8 w - - 1 2" # answer e3f4
# fen = "8/8/2pp4/KP5r/1R3P2/6k1/6P1/8 w - - 0 3" # b4b1
# fen = "8/8/2pp4/KP5r/5k2/8/6P1/1R6 w - - 0 4"
# chess = Chess(fen)
# agent = Agent(color=1)
# t0 = time.time()
# recommended_moves = agent.recommend(chess, 6, True)
# t1 = time.time()
# print(recommended_moves[0][0], recommended_moves[0][1], t1-t0)
# print(CACHE)
