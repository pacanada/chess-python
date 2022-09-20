# Introduction

## Performance

Using `python3 -m cProfile -o prof.txt tree.py -h` for profiling and `snakeviz prof.txt` to
visualize.

Perft(3) initial positions `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` (8902
positions, reference time 0.05 with `python-chess`:

    - 1.25 s (first)
    - 0.79 s (moving`get_positions_attacking_pieces` to optimizer initialization and update, not when checking if move

is legal) - 0.70 removing np.array from list of moves (does not make a lot of sense)

# TODO:

    - [X] Include tests for perft in different positions
    - [ ] Utils if there is a mismatch in positions
    - [ ] Improve performance
    - [ ] Improve overall code quality (clarity, choose right data structure for the job)
    - [ ] Automate release with github action to pip
    - [ ] Explore pypy
