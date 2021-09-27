# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    actions = []
    actionMap = {}
    stateMap = {}
    visited = []

    start = problem.getStartState()
    s = util.Stack()
    s.push(start)

    while (not s.isEmpty()):
        curr = s.pop()
        visited.append(curr)

        if (problem.isGoalState(curr)):
            # convert state mapping to actions
            while (curr != None):
                if curr == start:
                    return actions
                action = actionMap[curr]
                actions.insert(0, action)
                curr = stateMap[curr]
        
        # add successors to stack
        for succ in problem.getSuccessors(curr):
            nextState = succ[0]
            action = succ[1]
            if (nextState not in visited):
                s.push(nextState)
                stateMap[nextState] = curr
                actionMap[nextState] = action
                
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    actions = []
    actionMap = {}
    stateMap = {}
    visited = []

    start = problem.getStartState()
    q = util.Queue()
    q.push(start)

    while (not q.isEmpty()):
        curr = q.pop()
        visited.append(curr)

        if (problem.isGoalState(curr)):
            # convert state mapping to actions
            while (curr != None):
                if curr == start:
                    return actions
                action = actionMap[curr]
                actions.insert(0, action)
                curr = stateMap[curr]
        
        for succ in problem.getSuccessors(curr):
            nextState = succ[0]
            action = succ[1]
            if (nextState not in visited and nextState not in q.list):
                q.push(nextState)
                stateMap[nextState] = curr
                actionMap[nextState] = action

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    visited = []

    pq = util.PriorityQueue()
    start = (problem.getStartState(), []) # (states, [paths]), sorted by priority of cost
    pq.push(start, 0)

    while (not pq.isEmpty()):
        # choose lowest-cost node in frontier
        curr = pq.pop()
        curr_state = curr[0] # current state
        curr_path = curr[1] # list of actions

        if (problem.isGoalState(curr_state)):
            return curr_path

        if (curr_state not in visited):
            visited.append(curr_state)
            # functions as priority queue and hash table
            for succ in problem.getSuccessors(curr_state):
                next_state = succ[0]
                next_action = succ[1]

                if (next_state not in visited):
                    next_path = curr_path + [next_action]
                    total_cost = problem.getCostOfActions(next_path)
                    new_element = (next_state, next_path)
                    pq.update(new_element, total_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    """Search the node of least total cost first."""
    visited = []

    pq = util.PriorityQueue()
    start = (problem.getStartState(), []) # (states, [paths]), sorted by priority of cost
    pq.push(start, 0)

    while (not pq.isEmpty()):
        # choose lowest-cost node in frontier
        curr = pq.pop()
        curr_state = curr[0] # current state
        curr_path = curr[1] # list of actions

        if (problem.isGoalState(curr_state)):
            return curr_path

        if (curr_state not in visited):
            visited.append(curr_state)
            # functions as priority queue and hash table
            for succ in problem.getSuccessors(curr_state):
                next_state = succ[0]
                next_action = succ[1]

                if (next_state not in visited):
                    next_path = curr_path + [next_action]
                    h = heuristic(next_state, problem)
                    total_cost = problem.getCostOfActions(next_path) + h
                    new_element = (next_state, next_path)
                    pq.update(new_element, total_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
