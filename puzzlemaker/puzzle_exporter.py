import chess
from chess.pgn import Game

class PuzzleExporter(object):
    """ Exports a puzzle to a PGN file
    """
    def __init__(self, puzzle):
        self.puzzle = puzzle

    def export(self, pgn_headers=None) -> Game:
        """ pgn_headers - PGN headers to include in the exported PGN
        """
        if pgn_headers:
            for h in pgn_headers:
                if h == "FEN":
                    continue
                self.puzzle.game.headers[h] = pgn_headers[h]

        return self.puzzle.game

    def to_pgn(self, pgn_headers=None) -> str:
        return str(self.export(pgn_headers)).replace("}", "}\n")
