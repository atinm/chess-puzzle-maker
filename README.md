# Chess puzzle maker

This is a command-line program that creates chess puzzles from positions with clear sequences of best moves.
It looks for positions where a player can:

* Checkmate the opponent in a forced sequence
* Convert a position into a material advantage
* Equalize a losing position

Give it a PGN with any number of games or positions and it will look for positions to convert into puzzles:

`./make_puzzles.py --pgn games.pgn`

Or give it a position (FEN) and it will try to create a puzzle:

`./make_puzzles.py --fen "6rr/1k3p2/1pb1p1np/p1p1P2R/2P3R1/2P1B3/P1BK1PP1/8 b - - 5 26"`

For a list of options:

`./make_puzzles.py -h`


## Requirements

This requires Python 3 and a UCI-compatible chess engine such as Stockfish.
`
Install the required python libraries:

```
python3 -m venv venv
. venv/bin/activate
pip3 install -r requirements.txt
```

Make sure you have a version of Stockfish available in your `$PATH` or local directory:

* You can install a Stockfish binary using Homebrew if you're on macOS (`brew install stockfish`)
* Or install an old version of Stockfish using apt if you're on Ubuntu Linux (`sudo apt install stockfish`)
* Or download an official Stockfish release from the [Stockfish website](https://stockfishchess.org/download/)
* Or run `./build-stockfish.sh` to compile the latest [official Stockfish development build](https://github.com/official-stockfish/Stockfish)
* Or run `inv update-stockfish` to get the latest multi-variant Stockfish fork used by Lichess


## Creating puzzles

Created puzzles are printed in PGN format to standard output
while errors and log messages are printed to standard error.

If you want to write created puzzles to a file, redirect standard output:

`./make_puzzles.py --pgn games.pgn >> puzzles.pgn`


## How it works

It scans the moves of a game for mistakes, represented by large swings in position evaluation.

<img src="https://user-images.githubusercontent.com/208617/70076652-3af78380-15cd-11ea-969b-217789c5401b.png" width=340 />

For each of these positions, it looks for move sequences where the player can make one clear best move.

<img src="https://user-images.githubusercontent.com/208617/70076756-742ff380-15cd-11ea-828a-44f0ed12b78d.png" width=340 />

If it finds that the position could become a puzzle, it will output the puzzle as a PGN.


## Example PGN output

```
[FEN "6rr/1k3p2/1pb1p1np/p1p1P2R/2P3R1/2P1B3/P1BK1PP1/8 b - - 5 26"]
[PuzzleCategory "Material"]
[PuzzleEngine "Stockfish 2018-11-29 64 Multi-Variant"]
[PuzzleWinner "Black"]
[SetUp "1"]

26... Nxe5   { Nxe5  [-1.68] }
27.   Rxg8   { Rxg8  [-1.66] Rgh4 [-2.51]  Re4  [-3.51] }
27... Nxc4+  { Nxc4+ [-1.78] Rxg8 [ 3.82]  Nf3+ [ 4.79] }
28.   Kd3    { Kd3   [-1.55] Ke2  [-1.83]  Kd1  [-2.85] }
28... Nb2+   { Nb2+  [-1.78] Rxg8 [ 2.11]  Ne5+ [ 3.44] }
29.   Ke2    { Kd2   [-1.64] Ke2  [-1.74]               }
```

Each move in the puzzle is annotated with the best 3 candidate moves
considered by the chess engine.


## Example commands

To scan a PGN for positions that might be candidate puzzles without
investigating any of the positions:

`./make_puzzles.py --scan-only --pgn games.pgn`

To start at the n-th PGN in a PGN file with lots of games:

`./make_puzzles.py --start-index 1234 --pgn games.pgn`

To fetch a Lichess game and save it as a PGN:

`inv fetch-lichess -g 12345`

To fetch all games from a Lichess tournament and save the games to PGN:

`inv fetch-lichess -t 67890`

You can run the whole test suite with:

`inv test`


## Acknowledgements

This program is based on:

* [Python-Puzzle-Creator](https://github.com/clarkerubber/Python-Puzzle-Creator) by [@clarkerubber](https://github.com/clarkerubber)
* [pgn-tactics-generator](https://github.com/vitogit/pgn-tactics-generator) by [@vitogit](https://github.com/vitogit)
