# Introduction

Python **implementation of the chess game** in less than 1000 lines of code.

For engines/agents to play against check my other project
[chess-ai](https://github.com/pacanada/chess-ai)

There is also a work in progress for the cli based game (`python -m chess_python.game`).

# Usage

```
pip install chess-python-pip
```

```python
#examples/example_01.py
from chess_python.chess import Chess
chess = Chess(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0") # Default
print(chess)
print(chess.legal_moves())
chess.move("e2e4")
print(chess)
```

outputs:

```
Player to move: White
Move count: 0
8 || r | n | b | q | k | b | n | r |
-----------------------------------
7 || p | p | p | p | p | p | p | p |
-----------------------------------
6 ||   |   |   |   |   |   |   |   |
-----------------------------------
5 ||   |   |   |   |   |   |   |   |
-----------------------------------
4 ||   |   |   |   |   |   |   |   |
-----------------------------------
3 ||   |   |   |   |   |   |   |   |
-----------------------------------
2 || P | P | P | P | P | P | P | P |
-----------------------------------
1 || R | N | B | Q | K | B | N | R |
-----------------------------------
-----------------------------------
  || a | b | c | d | e | f | g | h

['b1a3', 'b1c3', 'g1f3', 'g1h3', 'a2a3', 'a2a4', 'b2b3', 'b2b4', 'c2c3', 'c2c4', 'd2d3', 'd2d4', 'e2e3', 'e2e4', 'f2f3', 'f2f4', 'g2g3', 'g2g4', 'h2h3', 'h2h4']
Player to move: Black
Move count: 1
8 || r | n | b | q | k | b | n | r |
-----------------------------------
7 || p | p | p | p | p | p | p | p |
-----------------------------------
6 ||   |   |   |   |   |   |   |   |
-----------------------------------
5 ||   |   |   |   |   |   |   |   |
-----------------------------------
4 ||   |   |   |   | P |   |   |   |
-----------------------------------
3 ||   |   |   |   |   |   |   |   |
-----------------------------------
2 || P | P | P | P |   | P | P | P |
-----------------------------------
1 || R | N | B | Q | K | B | N | R |
-----------------------------------
-----------------------------------
  || a | b | c | d | e | f | g | h
```

# DEV section:

## Performance tracking

Using `python3 -m cProfile -o prof.txt tree.py -h` for profiling and `snakeviz prof.txt` to
visualize.

Perft(3) initial position `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` (8902
positions, reference time 0.05 s with
[`python-chess`](https://python-chess.readthedocs.io/en/latest/):

- 1.25 s (first)
- 0.79 s (moving `get_positions_attacking_pieces` to optimizer initialization and update, not when
  checking if move is legal)
- 0.70 s removing `np.array` from list of moves (does not make a lot of sense)
- 0.55 s removing more np.arrays
- 0.51 s removing np.unravel
- 0.47 s using only lists in `get_allowed_moves_by_piece`
- 0.39 s with revamped `get_index_trajectory`
- 0.35 with custom `deepcopy`
- 0.18 s with using list for board instead of numpy!!
- 0.15 s further tweaks

## TODO:

- [x] Include tests for perft in different positions
- [x] Include utils if there is a mismatch in positions tree with reference implementation
- [ ] Improve performance:
  - [x] Keep track of index where there are pieces in optimizer level
  - [ ] Move and unmake move (implenting unmake also requires using a copy of the board, which
        performance wise does not improve anything)
- [ ] Improve overall code quality (clarity, choose right data structure for the job):
  - Public vs private functions
  - cyclomatic complexity in `is_legal_move`
- [x] Automate release with github action to pip
- [ ] Explore pypy
- [x] Explore deepcopy for tree generation, it takes a lot of time (only copying board improves
      performance)
- [ ] Simplify castling
