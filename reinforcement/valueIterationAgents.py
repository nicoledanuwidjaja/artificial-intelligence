# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Run set number of iterations
        for _ in range(self.iterations):
            # keep track of each iteration's results
            currValues = util.Counter() 
            for s in self.mdp.getStates():
                # Receive best action at each state
                action = self.getAction(s)
                if action is not None: currValues[s] = self.getQValue(s, action)
            self.values = currValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        value = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for transition in transitions:
             value += transition[1]*(self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[transition[0]])
        return value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state) # N/W/S/E
        actionValues = [] # store (action, value)
        
        for a in actions:
            # initialize value and accumulate for each transition state
            qValue = self.getQValue(state, a)
            actionValues.append((a, qValue))
        return max(actionValues, key=lambda i:i[1], default=(None, 0))[0]
                

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Run set number of iterations
        for i in range(self.iterations):
            # keep track of each iteration's results
            states = self.mdp.getStates()
            ithState = states[i % len(states)] # update ith value at each iteration
            # Receive best action at each state
            action = self.getAction(ithState)
            if action is not None: self.values[ithState] = self.getQValue(ithState, action)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # compute predecessors of all states
        def getPredecessors():
            map = {}
            # map through all states
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    nextActions = self.mdp.getPossibleActions(state)
                    for a in nextActions:
                        nextStates = self.mdp.getTransitionStatesAndProbs(state, a)
                        for nextState, _ in nextStates:
                            # add current state as predecessor for nextState (s)
                            # { state, set(...predecessor) } key value
                            preds = map.get(nextState)
                            if preds is None:
                                preds = set()
                            preds.add(state)
                            map[nextState] = preds
            return map

        # map of (state, set(predecessors))
        state = getPredecessors()
        pq = util.PriorityQueue() # min-heap of {state, theta}

        # keep track of each iteration's results
        states = self.mdp.getStates()
        for s in states:
            # get highest q-value from all possible actions from s
            # Receive best action at each state
            if not self.mdp.isTerminal(s):
                action = self.getAction(s)
                highestQValue = self.getQValue(s, action)
                diff = abs(self.values[s] - highestQValue)
                # prioritize updating states with higher error (theta)
                pq.update(s, -diff) 

        for _ in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            if not self.mdp.isTerminal(s):
                action = self.getAction(s)
                self.values[s] = self.getQValue(s, action)

            for pred in state.get(s):
                # get highest q-value from all possible actions from s
                # Receive best action at each state
                if not self.mdp.isTerminal(pred):
                    action = self.getAction(pred)
                    highestQValue = self.getQValue(pred, action)
                    diff = abs(self.values[pred] - highestQValue)
                    if diff > self.theta:
                        pq.update(pred, -diff)



