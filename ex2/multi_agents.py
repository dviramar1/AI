import math
import random
from enum import auto, Enum
from time import sleep

import numpy as np
import abc
import util
from game import Agent, Action
from typing import Callable, List, Tuple

from ex2.game_state import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        state_score = successor_game_state.score

        big_number_1 = 10 ** 5
        big_number_2 = 10 ** 4

        bonus_mapping = {Action.RIGHT: 2 * big_number_1, Action.DOWN: 2 * big_number_1,
                         Action.LEFT: big_number_1, Action.UP: 0}
        direction_score = bonus_mapping[action]

        if board[3, 3] == max_tile:
            corner_score = big_number_2
        else:
            corner_score = 0

        return state_score + direction_score + corner_score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinimaxPhase(Enum):
    min = auto()
    max = auto()


class MinmaxAgent(MultiAgentSearchAgent):

    def minimax(self, game_state: GameState, depth: int, phase: MinimaxPhase):
        if depth == 0:  # TODO: terminal mode
            return self.evaluation_function(game_state), None

        best_value = -math.inf if phase == MinimaxPhase.max else math.inf
        legal_actions = game_state.get_agent_legal_actions() if MinimaxPhase.max else game_state.get_opponent_legal_actions()  # TODO: fix
        for action in legal_actions:
            successor_game_state = game_state.generate_successor(action=action)  # TODO: agent index?
            next_phase = MinimaxPhase.min if phase == MinimaxPhase.max else MinimaxPhase.max
            new_value, _ = self.minimax(successor_game_state, depth - 1, next_phase)
            is_better = new_value > best_value if phase == MinimaxPhase.max else new_value < best_value
            if is_better:
                best_value = new_value
                best_action = action

        return best_value, best_action

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """

        _, best_action = self.minimax(game_state, self.depth, MinimaxPhase.max)
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()


class ExpectimaxPhase(Enum):
    expect = auto()
    max = auto()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def expectimax(self, game_state: GameState, depth: int, phase: ExpectimaxPhase):
        if depth == 0:  # TODO: or terminal node
            return self.evaluation_function(game_state), None

        if phase == ExpectimaxPhase.expect:
            legal_actions = game_state.get_opponent_legal_actions()
            weighted_average = 0
            for action in legal_actions:
                action_probability = 1 / (len(legal_actions))  # "assume the board response uniformly at random"
                successor_game_state = game_state.generate_successor(action=action, agent_index=1)
                action_value, _ = self.expectimax(successor_game_state, depth - 1, ExpectimaxPhase.max)
                weighted_average += action_probability * action_value

            return weighted_average, None

        elif phase == ExpectimaxPhase.max:
            best_value = -math.inf
            best_action = None
            legal_actions = game_state.get_agent_legal_actions()
            for action in legal_actions:
                successor_game_state = game_state.generate_successor(action=action)
                action_value, _ = self.expectimax(successor_game_state, depth - 1, ExpectimaxPhase.expect)
                if action_value > best_value:
                    best_value = action_value
                    best_action = action

            return best_value, best_action

        else:
            raise Exception(f"phase {phase} not supported here.")

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        _, best_action = self.expectimax(game_state, self.depth, ExpectimaxPhase.max)
        return best_action


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
