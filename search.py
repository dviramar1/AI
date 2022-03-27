"""
In search.py, you will implement generic search algorithms
"""
import os
from dataclasses import dataclass
from os import system
from time import sleep
from tkinter import W

from typing import List

import util
from board import Board


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


@dataclass
class SearchItem:
    state: Board
    actions: List

    def __iter__(self):
        yield self.state
        yield self.actions


def print_state(state):
    state = state.state
    new_rows = []
    for row in state:
        new_row = [" " if item == -1 else "X" for item in row]
        new_rows.append(new_row)
    print('\n'.join(map(str, new_rows)))
    print()
    sleep(0.3)


def _generic_search(problem: SearchProblem, fringe, use_cost=False, heuristic=None):
    """
    Implements a generic search using the given fringe

    Args:
        problem (SearchProblem): The problem to search on
        fringe (object): A fringe to keep the element in
    """

    if use_cost:
        priority = heuristic(problem.get_start_state(), problem) if heuristic else 0
        fringe.push(SearchItem(problem.get_start_state(), []), priority)
    else:
        fringe.push(SearchItem(problem.get_start_state(), []))
    closed = []

    while not fringe.isEmpty():
        curr_state, curr_actions = fringe.pop()

        if problem.is_goal_state(curr_state):
            return curr_actions

        elif curr_state not in closed:
            for (state, action, cost) in problem.get_successors(curr_state):
                if use_cost:
                    priority = cost
                    if heuristic:
                        priority += heuristic(state, problem)
                    fringe.push(SearchItem(state, curr_actions + [action]), priority)
                else:
                    fringe.push(SearchItem(state, curr_actions + [action]))
            closed += [curr_state]


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

	print("Start:", problem.get_start_state().state)
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    return _generic_search(problem, util.Stack())


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    return _generic_search(problem, util.Queue())


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    return _generic_search(problem, util.PriorityQueue(), use_cost=True)


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    return _generic_search(problem, util.PriorityQueue(), use_cost=True, heuristic=heuristic)


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
