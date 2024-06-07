import copy
import math
import random
from math import log, sqrt, inf
from random import randrange
import numpy as np
from rich.table import Table
from rich.progress import track
from rich.console import Console
from rich.progress import Progress

import classes.logic as logic

# When implementing a new strategy add it to the `str2strat`
# dictionary at the end of the file


class PlayerStrat:
    def __init__(self, _board_state, player):
        self.root_state = _board_state
        self.player = player

    def start(self):
        """
        This function select a tile from the board.

        @returns    (x, y) A tuple of integer corresponding to a valid
                    and free tile on the board.
        """
        raise NotImplementedError


class Node(object):
    """
    This class implements the main object that you will manipulate : nodes.
    Nodes include the state of the game (i.e. the 2D board), children (i.e. other children nodes), a list of
    untried moves, etc...
    """
    def __init__(self, board, move=(None, None),
                 wins=0, visits=0, children=None):
        # Save the #wins:#visited ratio
        self.state = board
        self.move = move
        self.wins = wins
        self.visits = visits
        self.children = children or []
        self.parent = None
        self.untried_moves = logic.get_possible_moves(board)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class Random(PlayerStrat):
    def start(self):
        """
        Implement the start function for the custom strategy.

        @returns    (x, y) A tuple of integers corresponding to a valid
                    and free tile on the board.
        """
        
        
        valid_moves = logic.get_possible_moves(self.root_state)
        chosen_move = random.choice(valid_moves)
        return chosen_move

class MiniMax(PlayerStrat):
    def __init__(self, _board_state, player, depth=100000):
        super().__init__(_board_state, player)
        self.depth = depth

    def check_winner(self, board):
        for i in [1,2]:
            is_winner = logic.is_game_over(i, board)
            if is_winner is not None:
                return is_winner
        return None
    
    def start(self):
        _, move = self.minimax(np.copy(self.root_state), True)
        return move

    def minimax(self, board, max_player):
        if self.check_winner(board) is not None:
            return self.evaluate(board), None
        #print(board)
        valid_moves = logic.get_possible_moves(board)

        if max_player:
            return self.maximize(board, valid_moves)
        else:
            return self.minimize(board, valid_moves)

    def maximize(self, board, valid_moves):
        max_eval = -inf
        best_move = None
        for move in valid_moves:
            new_board = self.apply_move(board, move, self.player)
            eval, _ = self.minimax(new_board, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move

    def minimize(self, board, valid_moves):
        min_eval = inf
        best_move = None
        for move in valid_moves:
            new_board = self.apply_move(board, move, 3 - self.player)
            eval, _ = self.minimax(new_board, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move

    def apply_move(self, board, move, player):
        new_board = np.copy(board)
        new_board[move[0]][move[1]] = player 
        return new_board

    def evaluate(self, board):
        winner = self.check_winner(board)
        if winner == self.player:
            return 1
        else:
            return -1



class MiniMaxAlphaBeta(PlayerStrat):
    def __init__(self, _board_state, player, depth=3):
        super().__init__(_board_state, player)
        self.depth = depth

    def check_winner(self, board):
        for player in [1, 2]:
            is_winner = logic.is_game_over(player, board)
            if is_winner is not None:
                return is_winner
        return None

    def start(self):
        _, move = self.alphabeta(np.copy(self.root_state), True, self.depth, -inf, inf)
        return move

    def alphabeta(self, board, max_player, depth, alpha, beta):
        winner = self.check_winner(board)
        if winner is not None or depth == 0:
            return self.evaluate(board), None

        valid_moves = logic.get_possible_moves(board)

        if max_player:
            return self.maximize(board, valid_moves, depth, alpha, beta)
        else:
            return self.minimize(board, valid_moves, depth, alpha, beta)

    def maximize(self, board, valid_moves, depth, alpha, beta):
        max_eval = -inf
        best_move = None
        for move in valid_moves:
            new_board = self.apply_move(board, move, self.player)
            eval, _ = self.alphabeta(new_board, False, depth - 1, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Élagage alpha-beta
        return max_eval, best_move

    def minimize(self, board, valid_moves, depth, alpha, beta):
        min_eval = inf
        best_move = None
        for move in valid_moves:
            new_board = self.apply_move(board, move, 3 - self.player)
            eval, _ = self.alphabeta(new_board, True, depth - 1, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Élagage alpha-beta
        return min_eval, best_move

    def apply_move(self, board, move, player):
        new_board = np.copy(board)
        new_board[move[0]][move[1]] = player
        return new_board

    def get_possible_moves_for_player(self, board: np.ndarray, player: int) -> list:
        """
        @return   All the coordinates of nodes where the specified player can play.
        """
        (x, y) = np.where(board == 0)
        return list(zip(x, y)) if player == 1 else []

    def evaluate(self, board):
        # Fonction d'évaluation ajustée
        score = 0

        # Mobilité
        player_moves = len(self.get_possible_moves_for_player(board, self.player))
        opponent_moves = len(self.get_possible_moves_for_player(board, 3 - self.player))
        score += 0.5 * (player_moves - opponent_moves)

        # Contrôle du centre
        center_coordinates = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for coord in center_coordinates:
            if board[coord[0]][coord[1]] == self.player:
                score += 2
            elif board[coord[0]][coord[1]] == 3 - self.player:
                score -= 2

        # Occupation du territoire
        player_tiles = np.count_nonzero(board == self.player)
        opponent_tiles = np.count_nonzero(board == 3 - self.player)
        score += 1.0 * (player_tiles - opponent_tiles)

        # Évitement des bords
        for i in range(board.shape[0]):
            if board[i][0] == self.player:
                score -= 0.3
            elif board[i][0] == 3 - self.player:
                score += 0.3

            if board[0][i] == self.player:
                score -= 0.3
            elif board[0][i] == 3 - self.player:
                score += 0.3

            if board[i][-1] == self.player:
                score -= 0.3
            elif board[i][-1] == 3 - self.player:
                score += 0.3

            if board[-1][i] == self.player:
                score -= 0.3
            elif board[-1][i] == 3 - self.player:
                score += 0.3

        return score
    

class MiniMaxAlphaBeta2(PlayerStrat):
    def __init__(self, _board_state, player, depth=3):
        super().__init__(_board_state, player)
        self.depth = depth

    def check_winner(self, board):
        for player in [1, 2]:
            is_winner = logic.is_game_over(player, board)
            if is_winner is not None:
                return is_winner
        return None

    def start(self):
        _, move = self.alphabeta(np.copy(self.root_state), True, self.depth, -inf, inf)
        return move

    def alphabeta(self, board, max_player, depth, alpha, beta):
        winner = self.check_winner(board)
        if winner is not None or depth == 0:
            return self.evaluate(board), None

        valid_moves = logic.get_possible_moves(board)

        if max_player:
            return self.maximize(board, valid_moves, depth, alpha, beta)
        else:
            return self.minimize(board, valid_moves, depth, alpha, beta)

    def maximize(self, board, valid_moves, depth, alpha, beta):
        max_eval = -inf
        best_move = None
        for move in valid_moves:
            new_board = self.apply_move(board, move, self.player)
            eval, _ = self.alphabeta(new_board, False, depth - 1, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Élagage alpha-beta
        return max_eval, best_move

    def minimize(self, board, valid_moves, depth, alpha, beta):
        min_eval = inf
        best_move = None
        for move in valid_moves:
            new_board = self.apply_move(board, move, 3 - self.player)
            eval, _ = self.alphabeta(new_board, True, depth - 1, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Élagage alpha-beta
        return min_eval, best_move

    def apply_move(self, board, move, player):
        new_board = np.copy(board)
        new_board[move[0]][move[1]] = player
        return new_board

    def get_possible_moves_for_player(self, board: np.ndarray, player: int) -> list:
        """
        @return   All the coordinates of nodes where the specified player can play.
        """
        (x, y) = np.where(board == 0)
        return list(zip(x, y)) if player == 1 else []

    def evaluate(self, board):
        # Fonction d'évaluation améliorée
        score = 0

        # Mobilité pondérée
        player_moves = len(self.get_possible_moves_for_player(board, self.player))
        opponent_moves = len(self.get_possible_moves_for_player(board, 3 - self.player))
        weighted_mobility = 0.5 * player_moves - 0.2 * opponent_moves
        score += weighted_mobility

        # Contrôle du centre pondéré
        center_coordinates = [(1, 1), (1, 2), (2, 1), (2, 2)]
        center_weight = 2.0
        for coord in center_coordinates:
            if board[coord[0]][coord[1]] == self.player:
                score += center_weight
            elif board[coord[0]][coord[1]] == 3 - self.player:
                score -= center_weight

        # Occupation des coins
        corner_weight = 3.0
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        for coord in corners:
            if board[coord[0]][coord[1]] == self.player:
                score += corner_weight
            elif board[coord[0]][coord[1]] == 3 - self.player:
                score -= corner_weight

        # Évitement des bords
        border_penalty = 0
        for i in range(board.shape[0]):
            if board[i][0] == self.player:
                score -=border_penalty
            elif board[i][0] == 3 - self.player:
                score +=border_penalty

            if board[0][i] == self.player:
                score -=border_penalty
            elif board[0][i] == 3 - self.player:
                score +=border_penalty

            if board[i][-1] == self.player:
                score -=border_penalty
            elif board[i][-1] == 3 - self.player:
                score +=border_penalty

            if board[-1][i] == self.player:
                score -=border_penalty
            elif board[-1][i] == 3 - self.player:
                score +=border_penalty

        return score


str2strat: dict[str, PlayerStrat] = {
        "human": None,
        "random": Random,
        "minimax": MiniMax,
        "elagage": MiniMaxAlphaBeta,
        "elagage2": MiniMaxAlphaBeta2
}