# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent
from mdp import MarkovDecisionProcess
import numpy as np

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp: MarkovDecisionProcess, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0

    "*** YOUR CODE HERE ***"
    self.learn_values()

  def learn_values(self):
    for i in range(self.iterations):
      new_values = util.Counter()
      for state in self.mdp.getStates():
        if not self.mdp.isTerminal(state):
          new_values[state] = max(self.get_expected_reward(state, action) + self.discount * self.getQValue(state, action) 
                                  for action in self.mdp.getPossibleActions(state))
      self.values = new_values

    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  def get_expected_reward(self, state, action):
    states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
    return sum(prob * self.mdp.getReward(state, action, next_state) for next_state, prob in states_and_probs)


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
    return sum(prob * self.values[next_state] for next_state, prob in states_and_probs)

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    if (self.mdp.isTerminal(state)):
      return None
    actions = self.mdp.getPossibleActions(state)
    max_index = np.argmax([self.getQValue(state, action) for action in actions])
    return actions[max_index]


  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
