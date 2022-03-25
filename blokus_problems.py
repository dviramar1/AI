import math
import random
from statistics import mean

from board import Board
from search import SearchProblem, ucs
import util

BIG_NUMBER = 1000000


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        return get_covered_corners(state, self) == 4

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return sum(action.piece.get_num_tiles() for action in actions)


def tiles_distance(p1, p2):
    return max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))


def is_in_board(position, problem):
    return 0 <= position[0] <= problem.board.board_w - 1 and 0 <= position[1] <= problem.board.board_h - 1


def is_tile_player_corner(state, p):
    neighbors = [(p[0], p[1]), (p[0], p[1] + 1), (p[0] + 1, p[1]), (p[0] + 1, p[1] + 1)]
    neighbors_num = sum(map(lambda pos: state.get_position(*pos) == 0, neighbors))
    return neighbors_num == 1


def get_player_corners(state, problem):
    points_w = problem.board.board_w - 1
    points_h = problem.board.board_h - 1

    player_corners = []
    for x in range(points_w - 1):
        for y in range(points_h - 1):
            curr_point = (x, y)
            if is_tile_player_corner(state, curr_point):
                player_corners.append(curr_point)
    return player_corners


def is_legal_next_pos(state, problem, pos):
    straight_neighbors = [(pos[0] + 1, pos[1]), (pos[0] - 1, pos[1]), (pos[0], pos[1] + 1), (pos[0], pos[1] - 1)]
    diagonal_neighbors = [(pos[0] + 1, pos[1] + 1), (pos[0] + 1, pos[1] - 1), (pos[0] - 1, pos[1] + 1),
                          (pos[0] - 1, pos[1] - 1)]

    straight_neighbors_num = sum(
        map(lambda pos: is_in_board(pos, problem) and state.get_position(*pos) == 0, straight_neighbors))
    diagonal_neighbors_num = sum(
        map(lambda pos: is_in_board(pos, problem) and state.get_position(*pos) == 0, diagonal_neighbors))

    is_legal_next_pos = straight_neighbors_num == 0 and diagonal_neighbors_num > 0
    return is_legal_next_pos


def get_legal_next_positions(state, problem):
    w = problem.board.board_w
    h = problem.board.board_h

    legal_positions = [(x, y) for x in range(w) for y in range(h) if is_legal_next_pos(state, problem, (x, y))]
    return legal_positions


def get_corners_dists(state, problem):
    w = problem.board.board_w
    h = problem.board.board_h

    board_corners = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]
    legal_next_positions = get_legal_next_positions(state, problem)

    if len(board_corners) == 0 or len(legal_next_positions) == 0:  # TODO: fix
        return [min(h / 2, w / 2) for _ in range(len(board_corners))]

    are_corners_covered = [state.get_position(*corner) == 0 for corner in board_corners]

    corners_dists = []
    for board_corner, is_covered in zip(board_corners, are_corners_covered):
        if is_covered:
            corner_dist = 0
        else:
            corner_dists = [tiles_distance(board_corner, next_pos) + 1 for next_pos in legal_next_positions]
            corner_dist = min(corner_dists)
        corners_dists.append(corner_dist)
    return corners_dists


def max_distance_heuristic(state, problem):
    corners_dists = get_corners_dists(state, problem)
    return max(corners_dists)


def min_distance_heuristic(state, problem):
    corners_dists = get_corners_dists(state, problem)
    corners_dists = filter(lambda dist: dist != 0, corners_dists)
    return min(corners_dists)


def mean_distance_heuristic(state, problem):
    corners_dists = get_corners_dists(state, problem)
    return mean(corners_dists)


def sum_distance_na_heuristic(state, problem):  # TODO: warning: not admissible
    corners_dists = get_corners_dists(state, problem)
    return sum(corners_dists)


def get_covered_corners(state, problem):
    corners = [(0, 0), (0, problem.board.board_h - 1), (problem.board.board_w - 1, 0),
               (problem.board.board_w - 1, problem.board.board_h - 1)]
    covered_corners = sum(state.get_position(*corner) == 0 for corner in corners)
    return covered_corners


def covered_corners_heuristic(state, problem):
    covered_corners = get_covered_corners(state, problem)
    return 4 - covered_corners


def is_near_corner_covered(state, problem):
    corners = [(0, 0), (0, problem.board.board_h - 1), (problem.board.board_w - 1, 0),
               (problem.board.board_w - 1, problem.board.board_h - 1)]
    corners_neighbors = [
        [(corner[0] + 1, corner[1]), (corner[0] - 1, corner[1]), (corner[0], corner[1] + 1), (corner[0], corner[1] - 1)]
        for corner in corners]
    corners_neighbors = [[pos for pos in cor_neigh if is_in_board(pos, problem)] for cor_neigh in
                         corners_neighbors]

    for corner, corner_neighbors in zip(corners, corners_neighbors):
        if state.get_position(*corner) != 0:
            for neighbor in corner_neighbors:
                if state.get_position(*neighbor) == 0:
                    return True
    return False


def has_no_legal_moves(state: Board):
    return state.get_legal_moves(0) == []


def is_fail_state(state, problem):
    return has_no_legal_moves(state) or is_near_corner_covered(state, problem)


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """

    detect_fails = True

    if problem.is_goal_state(state):
        return 0

    if detect_fails:
        if is_fail_state(state, problem):
            return BIG_NUMBER

    return 0.3 * covered_corners_heuristic(state, problem) + 0.7 * mean_distance_heuristic(state, problem)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = targets.copy()
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        for target in self.targets:
            if state.get_position(*target) != 0:
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return sum(action.piece.get_num_tiles() for action in actions)


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
