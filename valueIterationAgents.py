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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for k in range(self.iterations):
            newIteration = self.values.copy()  # to ensure that we do not alter self.values during current iteration
            for state in self.mdp.getStates():
                qvalues = [float("-inf")]
                if not self.mdp.isTerminal(state):
                    for action in self.mdp.getPossibleActions(state):
                        qvalues += [self.computeQValueFromValues(state, action)]
                    newIteration[state] = max(qvalues)  # return max of list of qvalues
            self.values = newIteration

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
        sum = 0
        trans = self.mdp.getTransitionStatesAndProbs(state, action)
        for item in trans:  # find qvalue by summing over transition states
            sum += (item[1] * (self.mdp.getReward(state, action, item[0]) + self.discount * self.values[item[0]]))
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # find maximum qvalue of possible actions
        options = [(float("-inf"), None)]
        for action in self.mdp.getPossibleActions(state):
            options.append((self.computeQValueFromValues(state, action), action))
        maximum = max(options, key=lambda x: x[0])
        return maximum[1]

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
        for k in range(self.iterations):
            states = self.mdp.getStates()
            state = states[k % len(states)]  # cyclic
            qvalues = [float("-inf")]
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    qvalues += [self.computeQValueFromValues(state, action)]
                self.values[state] = max(qvalues)


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
        predecessors = dict()
        for state in self.mdp.getStates():
            predecessors[state] = set()
        for state in self.mdp.getStates():
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                trans = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, prob in trans:
                    if prob > 0:
                        predecessors[nextState].add(state)

        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
            qvalues = [float("-inf")]
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    qvalues += [self.computeQValueFromValues(state, action)]
                maxq = max(qvalues)
                diff = abs(self.values[state] - maxq)
                pq.push(state, -diff)

        for k in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            if not self.mdp.isTerminal(s):
                qvalues = [float("-inf")]
                for action in self.mdp.getPossibleActions(s):
                    qvalues += [self.computeQValueFromValues(s, action)]
                self.values[s] = max(qvalues)
            for p in predecessors[s]:
                qvalues = [float("-inf")]
                for action in self.mdp.getPossibleActions(p):
                    qvalues += [self.computeQValueFromValues(p, action)]
                maxq = max(qvalues)
                diff = abs(self.values[p] - maxq)
                if diff > self.theta:
                    pq.update(p, -diff)
