from copy import deepcopy
import math
import random
from statistics import mean

from board import Board
from search import SearchProblem, ucs, astar
import util

BIG_NUMBER = 1000000


# general methods

def tiles_distance(p1, p2):
    return max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))


def is_in_board(position, state: Board):
    return 0 <= position[0] <= state.board_w - 1 and 0 <= position[1] <= state.board_h - 1


def is_legal_next_pos(state: Board, pos):
    x = pos[0]
    y = pos[1]
    return state.check_tile_legal(0, x, y) and state.check_tile_attached(0, x, y)


def get_legal_next_positions(state: Board):
    w = state.board_w
    h = state.board_h

    legal_positions = [(x, y) for x in range(w) for y in range(h) if is_legal_next_pos(state, (x, y))]
    return legal_positions


def get_dist_from_positions(state: Board, positions):
    legal_next_positions = get_legal_next_positions(state)

    if len(positions) == 0 or len(legal_next_positions) == 0:
        return None

    are_positions_covered = [state.get_position(*pos) == 0 for pos in positions]

    corners_dists = []
    for board_corner, is_covered in zip(positions, are_positions_covered):
        if is_covered:
            corner_dist = 0
        else:
            corner_dists = [tiles_distance(board_corner, next_pos) + 1 for next_pos in legal_next_positions]
            corner_dist = min(corner_dists)
        corners_dists.append(corner_dist)
    return corners_dists


def has_no_legal_moves(state: Board):
    return state.get_legal_moves(0) == []


def get_min_target_dists(state: Board, targets):
    targets = list(filter(lambda target: state.get_position(*target) != 0, targets))
    if len(targets) == 1:
        return BIG_NUMBER
    min_distance = tiles_distance(targets[0], targets[1])
    for i in range(1, len(targets)):
        for j in range(i):
            if i != j:
                distance = tiles_distance(targets[i], targets[j])
                if distance < min_distance:
                    min_distance = distance
    return min_distance


def get_sum_of_smallest_k(state, pieces_num, max_size):
    pieces_sizes = [piece.get_num_tiles() for piece in state.piece_list.pieces]
    small_sizes = [size for size in pieces_sizes if size <= max_size]
    big_sizes = [size for size in pieces_sizes if size > max_size]

    smallest_k_sum = sum(sorted(small_sizes)[:pieces_num])
    value = smallest_k_sum

    if len(big_sizes) > 0:
        smallest_big = min(big_sizes)
        value = min(value, smallest_big)

    return value


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
        self.min_targets_dist = min(board_w, board_h) - 2
        self.corners = [(0, 0), (0, self.board.board_h - 1), (self.board.board_w - 1, 0),
                        (self.board.board_w - 1, self.board.board_h - 1)]

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


def get_corners_dists(state, problem: BlokusCornersProblem):
    return get_dist_from_positions(state, problem.corners)


def max_distance_corners_heuristic(state, problem: BlokusCornersProblem):
    corners_dists = get_corners_dists(state, problem)
    if corners_dists is not None:
        return max(corners_dists)
    else:
        w = problem.board.board_w
        h = problem.board.board_h
        return max((h - 1) / 2, (w - 1) / 2)


def get_covered_corners(state, problem: BlokusCornersProblem):
    covered_corners = sum(state.get_position(*corner) == 0 for corner in problem.corners)
    return covered_corners


def num_of_corners_to_cover(state, problem: BlokusCornersProblem):
    covered_corners = get_covered_corners(state, problem)
    return 4 - covered_corners


def is_near_point_covered(state: Board, problem: SearchProblem, point):
    point_neighbors = [(point[0] + 1, point[1]), (point[0] - 1, point[1]), (point[0], point[1] + 1),
                       (point[0], point[1] - 1)]
    point_neighbors = filter(lambda pos: is_in_board(pos, state), point_neighbors)

    if state.get_position(*point) != 0:
        for neighbor in point_neighbors:
            if state.get_position(*neighbor) == 0:
                return True
    return False


def is_near_corner_covered(state, problem: BlokusCornersProblem):
    for corner in problem.corners:
        if is_near_point_covered(state, problem, corner):
            return True
    return False


def is_corner_fail_state(state, problem: BlokusCornersProblem):
    return has_no_legal_moves(state) or is_near_corner_covered(state, problem)


def small_pieces_corners_heuristic(state, problem: BlokusCornersProblem):
    corners_to_cover = num_of_corners_to_cover(state, problem)
    min_dist = get_min_target_dists(state, problem.corners)
    smallest_pieces_sum = get_sum_of_smallest_k(state, corners_to_cover, min_dist + 1)
    return smallest_pieces_sum


def blokus_corners_heuristic(state, problem: BlokusCornersProblem):
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
        if is_corner_fail_state(state, problem):
            return BIG_NUMBER

    smallest_pieces_value = small_pieces_corners_heuristic(state, problem)
    max_dist_value = max_distance_corners_heuristic(state, problem)
    value = max(smallest_pieces_value, max_dist_value)

    return value


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = targets.copy()
        self.standardized_targets = [target[::-1] for target in targets]
        self.expanded = 0


    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        for target in self.standardized_targets:
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


def covered_targets_heuristic(state: Board, problem: BlokusCoverProblem):
    return len(problem.targets) - sum(state.get_position(*target) == 0 for target in problem.standardized_targets)


def get_targets_dists(state, problem: BlokusCoverProblem, targets):
    return get_dist_from_positions(state, targets)


def max_distance_cover_heuristic(state, problem: BlokusCoverProblem):
    targets_dists = get_targets_dists(state, problem, problem.standardized_targets)
    if targets_dists is not None:
        return max(targets_dists)
    else:
        return 0


def get_num_targets_to_cover(state, problem):
    num_targets = 0
    for target in problem.standardized_targets:
        if state.get_position(*target) != 0:
            num_targets += 1
    return num_targets


def small_pieces_cover_heuristic(state, problem: BlokusCoverProblem):
    targets_to_cover = get_num_targets_to_cover(state, problem)
    min_dist = get_min_target_dists(state, problem.standardized_targets)
    smallest_pieces_sum = get_sum_of_smallest_k(state, targets_to_cover, min_dist + 1)
    return smallest_pieces_sum


def is_near_target_covered(state, problem: BlokusCoverProblem):
    for target in problem.standardized_targets:
        if is_near_point_covered(state, problem, target):
            return True
    return False


def is_cover_fail_state(state, problem: BlokusCoverProblem):
    return has_no_legal_moves(state) or is_near_target_covered(state, problem)


def blokus_cover_heuristic(state, problem):
    detect_fails = True

    if problem.is_goal_state(state):
        return 0

    if detect_fails:
        if is_cover_fail_state(state, problem):
            return BIG_NUMBER

    smallest_pieces = small_pieces_cover_heuristic(state, problem)
    max_dist_value = max_distance_cover_heuristic(state, problem)
    value = max(smallest_pieces, max_dist_value)

    return value


class MiniBlokusCoverProblem(BlokusCoverProblem):
    def __init__(self, prev_actions, board_w, board_h, piece_list, starting_point, targets, blacklist):
        super().__init__(board_w, board_h, piece_list, starting_point, targets)
        for action in prev_actions:
            self.board.add_move(0, action)
        self.blacklist = blacklist

    def is_goal_state(self, state):
        for target in self.standardized_targets:
            if state.get_position(*target) != 0:
                return False
        return not any(is_near_point_covered(state, self, target) for target in self.blacklist)


def is_near_target_blacklist_covered(state: Board, problem: MiniBlokusCoverProblem):
    for target in problem.standardized_targets:
        if is_near_point_covered(state, problem, target):
            return True

    for target in problem.blacklist:
        if is_near_point_covered(state, problem, target):
            return True

    return False


def is_closest_fail_state(state, problem: MiniBlokusCoverProblem):
    return has_no_legal_moves(state) or is_near_target_blacklist_covered(state, problem)


def closest_location_heuristic(state: Board, problem: MiniBlokusCoverProblem):
    detect_fails = True

    if detect_fails:
        if is_closest_fail_state(state, problem):
            return BIG_NUMBER

    pieces_sizes = [piece.get_num_tiles() for piece in state.piece_list.pieces]
    smallest_pieces = min(pieces_sizes)
    max_dist_value = max_distance_cover_heuristic(state, problem)
    value = max(smallest_pieces, max_dist_value)

    return value


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = targets.copy()
        self.standardized_targets = [target[::-1] for target in targets]
        self.expanded = 0
        self.starting_point = starting_point

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def closest_target(self, targets, back_trace):
        """
        Finds the closest target in the given target list
        """

        board = self.board.__copy__()
        for action in back_trace:
            board.add_move(0, action)
        dists = get_dist_from_positions(board, targets)
        if dists is None:
            dists = [tiles_distance(self.starting_point[::-1], target) for target in targets]
        return min(zip(targets, dists), key=lambda x: x[1])[0]

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

        back_trace = []
        targets = deepcopy(self.standardized_targets)
        while targets != []:
            target = self.closest_target(targets, back_trace)
            targets = list(filter(lambda x: x != target, targets))
            mini_problem = MiniBlokusCoverProblem(back_trace, self.board.board_w, self.board.board_h,
                                                  self.board.piece_list, starting_point=self.starting_point,
                                                  targets=[target[::-1]],
                                                  blacklist=list(filter(lambda x: x != target, self.standardized_targets)))
            actions_to_add = astar(mini_problem, closest_location_heuristic)
            self.expanded += mini_problem.expanded
            back_trace += actions_to_add
        return back_trace


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
