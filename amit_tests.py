import util
from search import dfs, bfs


class TestSearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def __init__(self):
        self.tree = {"a": ["b", "c"], "b": ["d", "e"], "c": ["f", "g"]}

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return "a"

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        return state == "d"

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        if state in self.tree:
            nodes = self.tree[state]
        else:
            nodes = []
        return list(zip(nodes, nodes, [1] * len(nodes)))

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


if __name__ == '__main__':
    dfs_actions = dfs(TestSearchProblem())
    print(dfs_actions)
    bfs_actions = bfs(TestSearchProblem())
    print(bfs_actions)
