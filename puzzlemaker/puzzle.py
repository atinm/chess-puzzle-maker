from collections import namedtuple
from typing import List, Literal, Optional
import chess
from chess import Move, Color, PieceType, square_rank, square_file, Board, SquareSet, Piece, PieceType, square_distance
from chess import KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN
from chess import WHITE, BLACK
import chess.pgn
from chess.pgn import ChildNode, Game

from puzzlemaker.puzzle_position import PuzzlePosition
from puzzlemaker.puzzle_exporter import PuzzleExporter
from puzzlemaker.logger import log, log_board, log_move
from puzzlemaker.colors import Color
from puzzlemaker.analysis import AnalysisEngine, AnalyzedMove
import puzzlemaker.utils
from puzzlemaker.utils import material_difference, material_diff
from puzzlemaker.constants import MIN_PLAYER_MOVES
from puzzlemaker.analysis import AnalysisEngine
from puzzlemaker.version import __version__

TagKind = Literal[
    "advancedPawn",
    "advantage",
    "attackingF2F7",
    "attraction",
    "backRankMate",
    "bishopEndgame",
    "capturingDefender",
    "castling",
    "clearance",
    "coercion",
    "crushing",
    "defensiveMove",
    "discoveredAttack",
    "deflection",
    "doubleCheck",
    "equality",
    "enPassant",
    "exposedKing",
    "fork",
    "hangingPiece",
    "interference",
    "intermezzo",
    "kingsideAttack",
    "knightEndgame",
    "long",
    "mate",
    "mateIn5",
    "mateIn4",
    "mateIn3",
    "mateIn2",
    "mateIn1",
    "oneMove",
    "overloading",
    "pawnEndgame",
    "pin",
    "promotion",
    "queenEndgame",
    "queensideAttack",
    "quietMove",
    "rookEndgame",
    "queenRookEndgame",
    "sacrifice",
    "short",
    "simplification",
    "skewer",
    "smotheredMate",
    "trappedPiece",
    "underPromotion",
    "veryLong",
    "xRayAttack",
    "zugzwang"
]

class Puzzle(object):
    """ initial_board [chess.Board]:
          the board before the first move in the puzzle

        initial_move [chess.uci.Move]:
          the first move in the puzzle

        initial_score [chess.uci.Score]:
          the initial score before the first move of the puzzle

        initial_position [PuzzlePosition]:
          the first position of the puzzle
          uses the initial move if there is one
          otherwise, uses the best analyzed move

        positions [list(PuzzlePosition)]:
          list of all positions included in the puzzle

        analyzed_moves [list(AnalyzedMove)]:
          list of analyzed possible first moves

        check_ambiguity [Boolean]:
          if true, don't generate new positions when the best move is ambiguous
    """
    def __init__(self, initial_board, initial_move=None):
        self.game = None
        self.initial_score = None
        self.initial_board = initial_board.copy()
        self.initial_move = initial_move
        self.initial_position = None
        self.final_score = None
        self.positions = []
        self.analyzed_moves = []
        self.mainline = []
        self.pov = None

    def _analyze_best_initial_move(self, depth) -> Move:
        log(Color.BLACK, "Evaluating best initial move (depth %d)..." % depth)
        best_move = AnalysisEngine.best_move(self.initial_board, depth)
        if best_move.move:
            self.analyzed_moves.append(best_move)
            log_move(self.initial_board, best_move.move, best_move.score, show_uci=True)
        self.initial_score = best_move.score
        return best_move.move

    def _analyze_initial_moves(self, depth):
        """ get the score of the position before the initial move
            also get the score of the position after the initial move
        """
        best_move = self._analyze_best_initial_move(depth)
        if not self.initial_move:
            return
        elif self.initial_move == best_move:
            log(Color.BLACK, "The move played was the best move")
        else:
            log(Color.BLACK, "Evaluating played initial move (depth %d)..." % depth)
            analyzed_move = AnalysisEngine.evaluate_move(self.initial_board, self.initial_move, depth)
            self.analyzed_moves.append(analyzed_move)
            log_move(self.initial_board, self.initial_move, analyzed_move.score, show_uci=True)

    def _set_initial_position(self):
        initial_move = self.initial_move
        if not initial_move and len(self.analyzed_moves) > 0:
            initial_move = self.analyzed_moves[0].move
        self.initial_position = PuzzlePosition(self.initial_board, initial_move)

    def _player_moves_first(self) -> bool:
        """ Determines if the player makes the first move in this puzzle
            If yes, the first move is the best move (i.e. winning position, checkmate puzzle)
            If no, the first move is a blunder/mistake by the opponent
        """
        if self.initial_score.is_mate() and not self.initial_move:
            # it's a checkmate puzzle without a move sequence
            white_to_move = self.initial_board.turn
            if white_to_move and self.initial_score.mate() > 0:
                return True
            elif not white_to_move and self.initial_score.mate() < 0:
                return True
        elif self.initial_move and self.initial_position.is_mate():
            # player just moved and has a winning position to checkmate the opponent
            white_just_moved = not self.initial_position.board.turn
            if self.initial_score.is_mate():
                if white_just_moved and self.initial_score.mate() > 0:
                    return True
                elif not white_just_moved and self.initial_score.mate() < 0:
                    return True
        # assume the initial move is a blunder/mistake by the opponent
        if self.initial_move:
            return False
        else:
            return True

    def _calculate_final_score(self, depth):
        """ Get the score of the final board position in the puzzle
            after the last move is made
        """
        final_score = self.positions[-1].score
        if final_score:
            self.final_score = final_score
        else:
            self.final_score = AnalysisEngine.score(self.positions[-1].board, depth)

    def generate(self, depth):
        """ Generate new positions for the puzzle until a final position is reached
        """
        log_board(self.initial_board)
        self._analyze_initial_moves(depth)
        self._set_initial_position()
        position = self.initial_position
        position.evaluate(depth)
        self.player_moves_first = self._player_moves_first()
        is_player_move = not self.player_moves_first
        while True:
            self.positions.append(position)
            if position.is_final(is_player_move):
                log_str = "Not going deeper: "
                if position.is_ambiguous():
                    log_str += "ambiguous"
                elif position.board.is_game_over():
                    log_str += "game over"
                log(Color.YELLOW, log_str)
                break
            else:
                log_str = "Going deeper..."
                if is_player_move is not None:
                    if is_player_move:
                        if len(position.candidate_moves) == 1:
                            log_str += " only one move"
                        else:
                            log_str += " one clear best move"
                    else:
                        log_str += " not player move"
                log(Color.DIM, log_str)
            position = PuzzlePosition(position.board, position.best_move)
            position.evaluate(depth)
            is_player_move = not is_player_move
        self._calculate_final_score(depth)
        if self.is_complete():
            log(Color.GREEN, "Puzzle is complete")
        else:
            log(Color.RED, "Puzzle incomplete")

    def to_pgn(self, pgn_headers=None) -> chess.pgn.Game:
        return PuzzleExporter(self).to_pgn(pgn_headers)

    def advanced_pawn(self) -> bool:
        for node in self.mainline[1::2]:
            if puzzlemaker.utils.is_very_advanced_pawn_move(node):
                return True
        return False

    def double_check(self) -> bool:
        for node in self.mainline[1::2]:
            if len(node.board().checkers()) > 1:
                return True
        return False

    def sacrifice(self) -> bool:
        # down in material compared to initial position, after moving
        diffs = [material_diff(n.board(), self.pov) for n in self.mainline]
        initial = diffs[0]
        for d in diffs[1::2][1:]:
            if d - initial <= -2:
                return not any(n.move.promotion for n in self.mainline[::2][1:])
        return False

    def x_ray(self) -> bool:
        for node in self.mainline[1::2][1:]:
            if not puzzlemaker.utils.is_capture(node):
                continue
            prev_op_node = node.parent
            assert isinstance(prev_op_node, ChildNode)
            if prev_op_node.move.to_square != node.move.to_square or puzzlemaker.utils.moved_piece_type(prev_op_node) == KING:
                continue
            prev_pl_node = prev_op_node.parent
            assert isinstance(prev_pl_node, ChildNode)
            if prev_pl_node.move.to_square != prev_op_node.move.to_square:
                continue
            if prev_op_node.move.from_square in SquareSet.between(node.move.from_square, node.move.to_square):
                return True

        return False

    def fork(self) -> bool:
        for node in self.mainline[1::2][:-1]:
            if puzzlemaker.utils.moved_piece_type(node) is not KING:
                board = node.board()
                if puzzlemaker.utils.is_in_bad_spot(board, node.move.to_square):
                    continue
                nb = 0
                for (piece, square) in puzzlemaker.utils.attacked_opponent_squares(board, node.move.to_square, self.pov):
                    if piece.piece_type == PAWN:
                        continue
                    if (
                        puzzlemaker.utils.king_values[piece.piece_type] > puzzlemaker.utils.king_values[puzzlemaker.utils.moved_piece_type(node)] or (
                            puzzlemaker.utils.is_hanging(board, piece, square) and
                            square not in board.attackers(not self.pov, node.move.to_square)
                        )
                    ):
                        nb += 1
                if nb > 1:
                    return True
        return False

    def hanging_piece(self) -> bool:
        to = self.mainline[1].move.to_square
        captured = self.mainline[0].board().piece_at(to)
        if self.mainline[0].board().is_check() and (not captured or captured.piece_type == PAWN):
            return False
        if captured and captured.piece_type != PAWN:
            if puzzlemaker.utils.is_hanging(self.mainline[0].board(), captured, to):
                op_move = self.mainline[0].move
                op_capture = self.game.board().piece_at(op_move.to_square)
                if op_capture and puzzlemaker.utils.values[op_capture.piece_type] >= puzzlemaker.utils.values[captured.piece_type] and op_move.to_square == to:
                    return False
                if len(self.mainline) < 3:
                    return True
                if material_diff(self.mainline[3].board(), self.pov) >= material_diff(self.mainline[1].board(), self.pov):
                    return True
        return False

    def trapped_piece(self) -> bool:
        for node in self.mainline[1::2][1:]:
            square = node.move.to_square
            captured = node.parent.board().piece_at(square)
            if captured and captured.piece_type != PAWN:
                prev = node.parent
                assert isinstance(prev, ChildNode)
                if prev.move.to_square == square:
                    square = prev.move.from_square
                if puzzlemaker.utils.is_trapped(prev.parent.board(), square):
                    return True
        return False

    def overloading(self) -> bool:
        return False

    def discovered_attack(self) -> bool:
        if self.discovered_check():
            return True
        for node in self.mainline[1::2][1:]:
            if puzzlemaker.utils.is_capture(node):
                between = SquareSet.between(node.move.from_square, node.move.to_square)
                assert isinstance(node.parent, ChildNode)
                if node.parent.move.to_square == node.move.to_square:
                    return False
                prev = node.parent.parent
                assert isinstance(prev, ChildNode)
                if (prev.move.from_square in between and
                    node.move.to_square != prev.move.to_square and
                    node.move.from_square != prev.move.to_square and
                    not puzzlemaker.utils.is_castling(prev)
                ):
                    return True
        return False

    def discovered_check(self) -> bool:
        for node in self.mainline[1::2]:
            board = node.board()
            checkers = board.checkers()
            if checkers and not node.move.to_square in checkers:
                return True
        return False

    def quiet_move(self) -> bool:
        for node in self.mainline:
            if (
                # on player move, not the last move of the puzzle
                node.turn() != self.pov and not node.is_end() and
                # no check given or escaped
                not node.board().is_check() and not node.parent.board().is_check() and
                # no capture made or threatened
                not puzzlemaker.utils.is_capture(node) and not puzzlemaker.utils.attacked_opponent_pieces(node.board(), node.move.to_square, self.pov) and
                # no advanced pawn push
                not puzzlemaker.utils.is_advanced_pawn_move(node) and
                puzzlemaker.utils.moved_piece_type(node) != KING
            ):
                return True
        return False

    def defensive_move(self) -> bool:
        # like quiet_move, but on last move
        # at least 3 legal moves
        if self.mainline[-2].board().legal_moves.count() < 3:
            return False
        node = self.mainline[-1]
        # no check given, no piece taken
        if node.board().is_check() or puzzlemaker.utils.is_capture(node):
            return False
        # no piece attacked
        if puzzlemaker.utils.attacked_opponent_pieces(node.board(), node.move.to_square, self.pov):
            return False
        # no advanced pawn push
        return not puzzlemaker.utils.is_advanced_pawn_move(node)

    def check_escape(self) -> bool:
        for node in self.mainline[1::2]:
            if node.board().is_check() or puzzlemaker.utils.is_capture(node):
                return False
            if node.parent.board().legal_moves.count() < 3:
                return False
            if node.parent.board().is_check():
                return True
        return False

    def attraction(self) -> bool:
        for node in self.mainline[1:]:
            if node.turn() == self.pov:
                continue
            # 1. player moves to a square
            first_move_to = node.move.to_square
            opponent_reply = puzzlemaker.utils.next_node(node)
            # 2. opponent captures on that square
            if opponent_reply and opponent_reply.move.to_square == first_move_to:
                attracted_piece = puzzlemaker.utils.moved_piece_type(opponent_reply)
                if attracted_piece in [KING, QUEEN, ROOK]:
                    attracted_to_square = opponent_reply.move.to_square
                    next_node = puzzlemaker.utils.next_node(opponent_reply)
                    if next_node:
                        attackers = next_node.board().attackers(self.pov, attracted_to_square)
                        # 3. player attacks that square
                        if next_node.move.to_square in attackers:
                            # 4. player checks on that square
                            if attracted_piece == KING:
                                return True
                            n3 = puzzlemaker.utils.next_next_node(next_node)
                            # 4. or player later captures on that square
                            if n3 and n3.move.to_square == attracted_to_square:
                                return True
        return False

    def deflection(self) -> bool:
        for node in self.mainline[1::2][1:]:
            captured_piece = node.parent.board().piece_at(node.move.to_square)
            if captured_piece or node.move.promotion:
                capturing_piece = puzzlemaker.utils.moved_piece_type(node)
                if captured_piece and puzzlemaker.utils.king_values[captured_piece.piece_type] > puzzlemaker.utils.king_values[capturing_piece]:
                    continue
                square = node.move.to_square
                prev_op_move = node.parent.move
                assert(prev_op_move)
                grandpa = node.parent.parent
                assert isinstance(grandpa, ChildNode)
                prev_player_move = grandpa.move
                prev_player_capture = grandpa.parent.board().piece_at(prev_player_move.to_square)
                if (
                    (not prev_player_capture or puzzlemaker.utils.values[prev_player_capture.piece_type] < puzzlemaker.utils.moved_piece_type(grandpa)) and
                    square != prev_op_move.to_square and square != prev_player_move.to_square and
                    (prev_op_move.to_square == prev_player_move.to_square or grandpa.board().is_check()) and
                    (
                        square in grandpa.board().attacks(prev_op_move.from_square) or
                        node.move.promotion and
                            square_file(node.move.to_square) == square_file(prev_op_move.from_square) and
                            node.move.from_square in grandpa.board().attacks(prev_op_move.from_square)
                    ) and
                    (not square in node.parent.board().attacks(prev_op_move.to_square))
                ):
                    return True
        return False

    def exposed_king(self) -> bool:
        if self.pov:
            pov = self.pov
            board = self.mainline[0].board()
        else:
            pov = not self.pov
            board = self.mainline[0].board().mirror()
        king = board.king(not pov)
        assert king is not None
        if chess.square_rank(king) < 5:
            return False
        squares = SquareSet.from_square(king - 8)
        if chess.square_file(king) > 0:
            squares.add(king - 1)
            squares.add(king - 9)
        if chess.square_file(king) < 7:
            squares.add(king + 1)
            squares.add(king - 7)
        for square in squares:
            if board.piece_at(square) == Piece(PAWN, not pov):
                return False
        for node in self.mainline[1::2][1:-1]:
            if node.board().is_check():
                return True
        return False

    def skewer(self) -> bool:
        for node in self.mainline[1::2][1:]:
            prev = node.parent
            assert isinstance(prev, ChildNode)
            capture = prev.board().piece_at(node.move.to_square)
            if capture and puzzlemaker.utils.moved_piece_type(node) in puzzlemaker.utils.ray_piece_types and not node.board().is_checkmate():
                between = SquareSet.between(node.move.from_square, node.move.to_square)
                op_move = prev.move
                assert op_move
                if (op_move.to_square == node.move.to_square or not op_move.from_square in between):
                    continue
                if (
                    puzzlemaker.utils.king_values[puzzlemaker.utils.moved_piece_type(prev)] > puzzlemaker.utils.king_values[capture.piece_type] and
                    puzzlemaker.utils.is_in_bad_spot(prev.board(), node.move.to_square)
                ):
                    return True
        return False

    def self_interference(self) -> bool:
        # intereference by opponent piece
        for node in self.mainline[1::2][1:]:
            prev_board = node.parent.board()
            square = node.move.to_square
            capture = prev_board.piece_at(square)
            if capture and puzzlemaker.utils.is_hanging(prev_board, capture, square):
                grandpa = node.parent.parent
                assert grandpa
                init_board = grandpa.board()
                defenders = init_board.attackers(capture.color, square)
                defender = defenders.pop() if defenders else None
                defender_piece = init_board.piece_at(defender) if defender else None
                if defender and defender_piece and defender_piece.piece_type in puzzlemaker.utils.ray_piece_types:
                    if node.parent.move and node.parent.move.to_square in SquareSet.between(square, defender):
                        return True
        return False

    def interference(self) -> bool:
        # intereference by player piece
        for node in self.mainline[1::2][1:]:
            prev_board = node.parent.board()
            square = node.move.to_square
            capture = prev_board.piece_at(square)
            assert node.parent.move
            if capture and square != node.parent.move.to_square and puzzlemaker.utils.is_hanging(prev_board, capture, square):
                assert node.parent
                assert node.parent.parent
                assert node.parent.parent.parent
                init_board = node.parent.parent.parent.board()
                defenders = init_board.attackers(capture.color, square)
                defender = defenders.pop() if defenders else None
                defender_piece = init_board.piece_at(defender) if defender else None
                if defender and defender_piece and defender_piece.piece_type in puzzlemaker.utils.ray_piece_types:
                    interfering = node.parent.parent
                    if interfering.move and interfering.move.to_square in SquareSet.between(square, defender):
                        return True
        return False

    def intermezzo(self) -> bool:
        for node in self.mainline[1::2][1:]:
            if puzzlemaker.utils.is_capture(node):
                capture_move = node.move
                capture_square = node.move.to_square
                op_node = node.parent
                assert isinstance(op_node, ChildNode)
                prev_pov_node = node.parent.parent
                assert isinstance(prev_pov_node, ChildNode)
                if not op_node.move.from_square in prev_pov_node.board().attackers(not self.pov, capture_square):
                    if prev_pov_node.move.to_square != capture_square:
                        prev_op_node = prev_pov_node.parent
                        assert isinstance(prev_op_node, ChildNode)
                        return (
                            prev_op_node.move.to_square == capture_square and
                            puzzlemaker.utils.is_capture(prev_op_node) and
                            capture_move in prev_op_node.board().legal_moves
                        )
        return False

    # the pinned piece can't attack a player piece
    def pin_prevents_attack(self) -> bool:
        for node in self.mainline[1::2]:
            board = node.board()
            for square, piece in board.piece_map().items():
                if piece.color == self.pov:
                    continue
                pin_dir = board.pin(piece.color, square)
                if pin_dir == chess.BB_ALL:
                    continue
                for attack in board.attacks(square):
                    attacked = board.piece_at(attack)
                    if attacked and attacked.color == self.pov and not attack in pin_dir and (
                            puzzlemaker.utils.values[attacked.piece_type] > puzzlemaker.utils.values[piece.piece_type] or
                            puzzlemaker.utils.is_hanging(board, attacked, attack)
                        ):
                        return True
        return False

    # the pinned piece can't escape the attack
    def pin_prevents_escape(self) -> bool:
        for node in self.mainline[1::2]:
            board = node.board()
            for pinned_square, pinned_piece in board.piece_map().items():
                if pinned_piece.color == self.pov:
                    continue
                pin_dir = board.pin(pinned_piece.color, pinned_square)
                if pin_dir == chess.BB_ALL:
                    continue
                for attacker_square in board.attackers(self.pov, pinned_square):
                    if attacker_square in pin_dir:
                        attacker = board.piece_at(attacker_square)
                        assert(attacker)
                        if puzzlemaker.utils.values[pinned_piece.piece_type] > puzzlemaker.utils.values[attacker.piece_type]:
                            return True
                        if (puzzlemaker.utils.is_hanging(board, pinned_piece, pinned_square) and
                            pinned_square not in board.attackers(not self.pov, attacker_square) and
                            [m for m in board.pseudo_legal_moves if m.from_square == pinned_square and m.to_square not in pin_dir]
                        ):
                            return True
        return False

    def attacking_f2_f7(self) -> bool:
        for node in self.mainline[1::2]:
            square = node.move.to_square
            if node.parent.board().piece_at(node.move.to_square) and square in [chess.F2, chess.F7]:
                king = node.board().piece_at(chess.E8 if square == chess.F7 else chess.E1)
                return king is not None and king.piece_type == KING and king.color != self.pov
        return False

    def kingside_attack(self) -> bool:
        return self.side_attack(7, [6, 7], 20)

    def queenside_attack(self) -> bool:
        return self.side_attack(0, [0, 1, 2], 18)

    def side_attack(self, corner_file: int, king_files: List[int], nb_pieces: int) -> bool:
        back_rank = 7 if self.pov else 0
        init_board = self.mainline[0].board()
        king_square = init_board.king(not self.pov)
        if (
            not king_square or
            square_rank(king_square) != back_rank or
            square_file(king_square) not in king_files or
            len(init_board.piece_map()) < nb_pieces or # no endgames
            not any(node.board().is_check() for node in self.mainline[1::2])
        ):
            return False
        score = 0
        corner = chess.square(corner_file, back_rank)
        for node in self.mainline[1::2]:
            corner_dist = square_distance(corner, node.move.to_square)
            if node.board().is_check():
                score += 1
            if puzzlemaker.utils.is_capture(node) and corner_dist <= 3:
                score += 1
            elif corner_dist >= 5:
                score -= 1
        return score >= 2

    def clearance(self) -> bool:
        for node in self.mainline[1::2][1:]:
            board = node.board()
            if not node.parent.board().piece_at(node.move.to_square):
                piece = board.piece_at(node.move.to_square)
                if piece and piece.piece_type in puzzlemaker.utils.ray_piece_types:
                    prev = node.parent.parent
                    assert prev
                    prev_move = prev.move
                    assert prev_move
                    assert isinstance(node.parent, ChildNode)
                    if (not prev_move.promotion and
                        prev_move.to_square != node.move.from_square and
                        prev_move.to_square != node.move.to_square and
                        not node.parent.board().is_check() and
                        (not board.is_check() or puzzlemaker.utils.moved_piece_type(node.parent) != KING)):
                        if (prev_move.from_square == node.move.to_square or
                            prev_move.from_square in SquareSet.between(node.move.from_square, node.move.to_square)):
                            if prev.parent and not prev.parent.board().piece_at(prev_move.to_square) or puzzlemaker.utils.is_in_bad_spot(prev.board(), prev_move.to_square):
                                return True
        return False

    def en_passant(self) -> bool:
        for node in self.mainline[1::2]:
            if (puzzlemaker.utils.moved_piece_type(node) == PAWN and
                square_file(node.move.from_square) != square_file(node.move.to_square) and
                not node.parent.board().piece_at(node.move.to_square)
            ):
                return True
        return False

    def castling(self) -> bool:
        for node in self.mainline[1::2]:
            if puzzlemaker.utils.is_castling(node):
                return True
        return False

    def promotion(self) -> bool:
        for node in self.mainline[1::2]:
            if node.move.promotion:
                return True
        return False

    def under_promotion(self) -> bool:
        for node in self.mainline[1::2]:
            if node.move.promotion and node.move.promotion != QUEEN:
                return True
        return False

    def capturing_defender(self) -> bool:
        for node in self.mainline[1::2][1:]:
            board = node.board()
            capture = node.parent.board().piece_at(node.move.to_square)
            assert isinstance(node.parent, ChildNode)
            if board.is_checkmate() or (
                capture and
                puzzlemaker.utils.moved_piece_type(node) != KING and
                puzzlemaker.utils.values[capture.piece_type] <= puzzlemaker.utils.values[puzzlemaker.utils.moved_piece_type(node)] and
                puzzlemaker.utils.is_hanging(node.parent.board(), capture, node.move.to_square) and
                node.parent.move.to_square != node.move.to_square
            ):
                prev = node.parent.parent
                assert isinstance(prev, ChildNode)
                if not prev.board().is_check() and prev.move.to_square != node.move.from_square:
                    assert prev.parent
                    init_board = prev.parent.board()
                    defender_square = prev.move.to_square
                    defender = init_board.piece_at(defender_square)
                    if (defender and
                        defender_square in init_board.attackers(defender.color, node.move.to_square) and
                        not init_board.is_check()):
                        return True
        return False

    def smothered_mate(self) -> bool:
        board = self.game.end().board()
        king_square = board.king(not self.pov)
        assert king_square is not None
        for checker_square in board.checkers():
            piece = board.piece_at(checker_square)
            assert piece
            if piece.piece_type == KNIGHT:
                for escape_square in [s for s in chess.SQUARES if square_distance(s, king_square) == 1]:
                    blocker = board.piece_at(escape_square)
                    if not blocker or blocker.color == self.pov:
                        return False
                return True
        return False

    def back_rank_mate(self) -> bool:
        node = self.game.end()
        board = node.board()
        king = board.king(not self.pov)
        assert king is not None
        assert isinstance(node, ChildNode)
        back_rank = 7 if self.pov else 0
        if board.is_checkmate() and square_rank(king) == back_rank:
            squares = SquareSet.from_square(king + (-8 if self.pov else 8))
            if self.pov:
                if chess.square_file(king) < 7:
                    squares.add(king - 7)
                if chess.square_file(king) > 0:
                    squares.add(king - 9)
            else:
                if chess.square_file(king) < 7:
                    squares.add(king + 9)
                if chess.square_file(king) > 0:
                    squares.add(king + 7)
            for square in squares:
                piece = board.piece_at(square)
                if piece is None or piece.color == self.pov or board.attackers(self.pov, square):
                    return False
            return any(square_rank(checker) == back_rank for checker in board.checkers())
        return False

    def piece_endgame(self, piece_type: PieceType) -> bool:
        for board in [self.mainline[i].board() for i in [0, 1]]:
            if not board.pieces(piece_type, WHITE) and not board.pieces(piece_type, BLACK):
                return False
            for piece in board.piece_map().values():
                if not piece.piece_type in [KING, PAWN, piece_type]:
                    return False
        return True

    def queen_rook_endgame(self) -> bool:
        def test(board: Board) -> bool:
            pieces = board.piece_map().values()
            return (
                len([p for p in pieces if p.piece_type == QUEEN]) == 1 and
                any(p.piece_type == ROOK for p in pieces) and
                all(p.piece_type in [QUEEN, ROOK, PAWN, KING] for p in pieces)
            )
        return all(test(self.mainline[i].board()) for i in [0, 1])

    def mate_in(self) -> Optional[TagKind]:
        if not self.game.end().board().is_checkmate():
            return None
        moves_to_mate = len(self.mainline) // 2
        if moves_to_mate == 1:
            return "mateIn1"
        elif moves_to_mate == 2:
            return "mateIn2"
        elif moves_to_mate == 3:
            return "mateIn3"
        elif moves_to_mate == 4:
            return "mateIn4"
        return "mateIn5"

    def category(self) -> Optional[str]:
        """ Mate     - win by checkmate
            Material - gain a material advantage
            Equalize - equalize a losing position
        """
        tags : List[TagKind] = []

        self.pov = chess.WHITE if self.winner() == "White" else chess.BLACK

        initial_cp = self.initial_score.score()
        final_cp = self.final_score.score()
        mate_tag = self.mate_in()
        if mate_tag:
            tags.append(mate_tag)
            tags.append("mate")
        else:
            if initial_cp is not None and final_cp is not None:
                # going from a disadvantage to an equal position
                if abs(initial_cp) > 200 and abs(final_cp) < 90:
                    tags.append("equality")
                # otherwise, the puzzle is only complete if the score changed
                # significantly after the initial position and was converted
                # into a material advantage
                initial_material_diff = material_difference(self.positions[0].initial_board)
                final_material_diff = material_difference(self.positions[-1].board)
                if abs(final_material_diff - initial_material_diff) > 0.1:
                    if abs(final_cp - initial_cp) > 600:
                        tags.append("crushing")
                    elif abs(final_cp - initial_cp) > 200:
                        tags.append("advantage")
                    elif not self.initial_move:
                        # a puzzle from a position, not a sequence of moves
                        if final_cp > 200 or final_cp < -200:
                            tags.append("advantage")

        # we have game, we can do more categorization
        if self.game != None:
            if "mate" in tags:
                if self.smothered_mate():
                    tags.append("smotheredMate")
                elif self.back_rank_mate():
                    tags.append("backRankMate")

            if self.attraction():
                tags.append("attraction")

            if self.deflection():
                tags.append("deflection")
            elif self.overloading():
                tags.append("overloading")

            if self.advanced_pawn():
                tags.append("advancedPawn")

            if self.double_check():
                tags.append("doubleCheck")

            if self.quiet_move():
                tags.append("quietMove")

            if self.defensive_move() or self.check_escape():
                tags.append("defensiveMove")

            if self.sacrifice():
                tags.append("sacrifice")

            if self.x_ray():
                tags.append("xRayAttack")

            if self.fork():
                tags.append("fork")

            if self.hanging_piece():
                tags.append("hangingPiece")

            if self.trapped_piece():
                tags.append("trappedPiece")

            if self.discovered_attack():
                tags.append("discoveredAttack")

            if self.exposed_king():
                tags.append("exposedKing")

            if self.skewer():
                tags.append("skewer")

            if self.self_interference() or self.interference():
                tags.append("interference")

            if self.intermezzo():
                tags.append("intermezzo")

            if self.pin_prevents_attack() or self.pin_prevents_escape():
                tags.append("pin")

            if self.attacking_f2_f7():
                tags.append("attackingF2F7")

            if self.clearance():
                tags.append("clearance")

            if self.en_passant():
                tags.append("enPassant")

            if self.castling():
                tags.append("castling")

            if self.promotion():
                tags.append("promotion")
                if self.under_promotion():
                    tags.append("underPromotion")

            if self.capturing_defender():
                tags.append("capturingDefender")

            if self.piece_endgame(PAWN):
                tags.append("pawnEndgame")
            elif self.piece_endgame(QUEEN):
                tags.append("queenEndgame")
            elif self.piece_endgame(ROOK):
                tags.append("rookEndgame")
            elif self.piece_endgame(BISHOP):
                tags.append("bishopEndgame")
            elif self.piece_endgame(KNIGHT):
                tags.append("knightEndgame")
            elif self.queen_rook_endgame():
                tags.append("queenRookEndgame")

            if "backRankMate" not in tags and "fork" not in tags:
                if self.kingside_attack():
                    tags.append("kingsideAttack")
                elif self.queenside_attack():
                    tags.append("queensideAttack")

        # we should have either a mate or a material advantage change
        if (any(x in ["mate", "equality", "crushing", "advantage"] for x in tags)):
            if len(self.mainline) == 2:
                tags.append("oneMove")
            elif len(self.mainline) <= 4:
                tags.append("short")
            elif len(self.mainline) >= 8:
                tags.append("veryLong")
            else:
                tags.append("long")

            return " ".join(tags)
        else:
            return None

    def winner(self) -> Optional[str]:
        """ Find the winner of the puzzle based on the move sequence
        """
        position = self.positions[-2]
        if position.score.mate() == 1:
            return "White"
        elif position.score.mate() == -1:
            return "Black"
        initial_cp = self.initial_score.score()
        final_cp = self.final_score.score()
        if initial_cp is not None and final_cp is not None:
            # evaluation change favors white
            if final_cp - initial_cp > 100:
                return "White"
            # evaluation change favors black
            elif final_cp - initial_cp < -100:
                return "Black"
            # evaluation equalized after initially favoring black
            elif initial_cp < 0 and abs(final_cp) < 50:
                return "White"
            # evaluation equalized after initially favoring white
            elif initial_cp > 0 and abs(final_cp) < 50:
                return "Black"
        if not self.initial_move and final_cp:
            # a puzzle from a position, not a sequence of moves
            if final_cp > 100:
                return "White"
            elif final_cp < -100:
                return "Black"

    def _score_to_str(self, score) -> str:
        if score.is_mate():
            return "mate in %d" % score.mate()
        else:
            return score.cp

    def _candidate_moves_annotations(self, candidate_moves) -> str:
        """ Returns the candidate moves with evaluations for PGN comments
        """
        comment = ""
        for candidate_move in candidate_moves:
            comment += candidate_move.move_san
            comment += " (%s) " % self._score_to_str(candidate_move.score)
        return comment.strip()

    def is_complete(self) -> bool:
        """ Verify that this sequence of moves represents a complete puzzle
            Incomplete if too short or if the puzzle could not be categorized
        """
        n_player_moves = 1 if self.player_moves_first else 0
        n_player_moves += int((len(self.positions) - 1) / 2)
        if n_player_moves < MIN_PLAYER_MOVES:
            return False
        fen = self.initial_board.fen()
        board = chess.Board(fen)
        self.game = Game().from_board(board)
        game_node = self.game
        game_node.comment = "score: %s -> %s" % (
            self._score_to_str(self.initial_score),
            self._score_to_str(self.final_score)
        )
        comment = self._candidate_moves_annotations(self.analyzed_moves)
        for position in self.positions:
            game_node = game_node.add_variation(
                chess.Move.from_uci(position.initial_move.uci())
            )
            if comment:
                game_node.comment = comment
            comment = self._candidate_moves_annotations(position.candidate_moves)
        self.mainline = list(self.game.mainline())
        puzzle_winner = self.winner()
        if puzzle_winner:
            self.game.headers['PuzzleWinner'] = puzzle_winner
        self.game.headers['PuzzleEngine'] = AnalysisEngine.name()
        self.game.headers['PuzzleMakerVersion'] = __version__
        category = self.category()
        if category:
            self.game.headers['PuzzleCategory'] = category
            return True
        return False
