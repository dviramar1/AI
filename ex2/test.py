from audioop import minmax
from re import M
from multi_agents import AlphaBetaAgent
from game_state import GameState
import math
from enum import Enum, auto

class TestState(GameState):

    def __init__(self, node='a'):
        super(TestState, self).__init__()
        self.nodes = {'a': 0,
                      'b': 0, 'c': 5,
                      'd': 2, 'e': 1, 'f': 0, 'g': 10}
        self.edges = {'a': ['b', 'c'], 'b': ['d', 'e'], 'c': ['f', 'g']}
        self.node = node

    def get_agent_legal_actions(self):
        if self.node in self.edges.keys():
            return self.edges[self.node]
        return []

    def get_opponent_legal_actions(self):
        if self.node in self.edges.keys():
            return self.edges[self.node]
        return []

    def generate_successor(self, agent_index=0, action=0):
        return TestState(node=action)

class MinimaxPhase(Enum):
    min = auto()
    max = auto()

class MinmaxAgent():

    def __init__(self, evaluation_function, depth=2):
        self.evaluation_function = evaluation_function
        self.depth = depth

    def max_phase(self, game_state: GameState, depth: int):
        best_value = -math.inf
        best_action = None
        legal_actions = game_state.get_agent_legal_actions()
        for action in legal_actions:
            successor_game_state = game_state.generate_successor(action=action, agent_index=0)
            new_value, _ = self.minimax(successor_game_state, depth - 1, MinimaxPhase.min)
            if new_value > best_value:
                best_value = new_value
                best_action = action

        return best_value, best_action

    def min_phase(self, game_state: GameState, depth: int):
        best_value = math.inf
        best_action = None
        legal_actions = game_state.get_opponent_legal_actions()
        for action in legal_actions:
            successor_game_state = game_state.generate_successor(action=action, agent_index=1)
            new_value, _ = self.minimax(successor_game_state, depth - 1, MinimaxPhase.max)
            if new_value < best_value:
                best_value = new_value
                best_action = action

        return best_value, best_action

    def minimax(self, game_state: GameState, depth: int, phase: MinimaxPhase):
        if depth == 0 or game_state.done:
            return self.evaluation_function(game_state), None

        strategy_of_phase = {MinimaxPhase.max: self.max_phase, MinimaxPhase.min: self.min_phase}
        return strategy_of_phase[phase](game_state, depth)

class AlphaBetaAgent(object):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def __init__(self, evaluation_function, depth=2):
        self.evaluation_function = evaluation_function
        self.depth = depth

    def max_phase(self, game_state: GameState, depth: int, alpha: float, beta: float):
        value = -math.inf
        legal_actions = game_state.get_agent_legal_actions()

        print ('max')
        print (game_state.node)
        print (legal_actions)

        for action in legal_actions:
            successor_game_state = game_state.generate_successor(action=action, agent_index=0)
            new_value, _ = self.alphabeta(successor_game_state, depth - 1, alpha, beta, MinimaxPhase.min)
            value = max(value, new_value)
            print (action, value)
            print (alpha, beta)
            if value >= beta:
                break
            alpha = max(alpha, value)
            print(alpha, beta)
            print()

        return value, action

    def min_phase(self, game_state: GameState, depth: int, alpha: float, beta: float):
        value = math.inf
        legal_actions = game_state.get_opponent_legal_actions()

        print ('min')
        print (game_state.node)
        print (legal_actions)

        for action in legal_actions:
            successor_game_state = game_state.generate_successor(action=action, agent_index=1)
            new_value, _ = self.alphabeta(successor_game_state, depth - 1, alpha, beta, MinimaxPhase.max)
            value = min(value, new_value)
            print (action, value)
            print (alpha, beta)
            if value <= alpha:
                break
            beta = min(beta, value)
            print(alpha, beta)
            print()

        return value, action

    # TODO fail hard vs fail soft?
    def alphabeta(self, game_state: GameState, depth: int, alpha: float, beta: float, phase: MinimaxPhase):
        if depth == 0 or game_state.done:
            print ('done: ', game_state.node)
            return self.evaluation_function(game_state), None

        strategy_of_phase = {MinimaxPhase.max: self.max_phase, MinimaxPhase.min: self.min_phase}
        return strategy_of_phase[phase](game_state, depth, alpha, beta)

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        best_value, action = self.alphabeta(game_state, self.depth, -math.inf, math.inf, MinimaxPhase.max)
        return action

def main():
    state = TestState()
    agent = AlphaBetaAgent(evaluation_function=lambda x:x.nodes[x.node], depth=2)
    # agent = MinmaxAgent(evaluation_function=lambda x:x.nodes[x.node], depth=2)

    print (agent.alphabeta(state, 2, -math.inf, math.inf, MinimaxPhase.max))
    # print (agent.minimax(state, 2, MinimaxPhase.max))

if __name__ == '__main__':
    main()