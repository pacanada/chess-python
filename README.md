# Introduction

Python implementation of chess game in less than 1000 lines of code.

There is also a work in progress for the cli based game (`python -m chess_python.game`).

Disclaimer: it is missing api for checkmate and stalemates

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
- 0.70 s removing np.array from list of moves (does not make a lot of sense)
- 0.55 s removing more np.arrays

## TODO:

- [x] Include tests for perft in different positions
- [ ] Include utils if there is a mismatch in positions tree with reference implementation
- [ ] Improve performance
- [ ] Improve overall code quality (clarity, choose right data structure for the job)
- [x] Automate release with github action to pip
- [ ] Explore pypy
