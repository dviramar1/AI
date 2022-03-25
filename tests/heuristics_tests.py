import random

from blokus_problems import BlokusCornersProblem, get_corners_dists, get_legal_next_positions
from pieces import PieceList


def get_random_state(problem):
    start_state = problem.get_start_state()

    curr_state = start_state
    for i in range(5):
        curr_state = random.choice(problem.get_successors(curr_state))[0]

    return curr_state


if __name__ == '__main__':
    piece_list = PieceList('tiny_set_2.txt')
    problem = BlokusCornersProblem(8, 8, piece_list, (0, 0))

    state = get_random_state(problem)
    player_nexts = get_legal_next_positions(state, problem)
    corners_dists = get_corners_dists(state, problem)

    print(state)
    print(player_nexts)
    print(corners_dists)
    print("make sense?")
