# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        foodLeft = successorGameState.getFood().asList()

        # Check if successor state completes game
        if not foodLeft:
            print("Complete!")
            return 99999

        # Get closest food
        foodDistances = [manhattanDistance(newPos, food) for food in foodLeft]
        bestDistance = min(foodDistances, default=0)
        # bestFoods = [i for i in range(len(foodDistances)) if foodDistances[i] == bestDistance]
        # randomBest = foodLeft[random.choice(bestFoods)]
        
        # Get reciprocal value
        value = 1 / bestDistance

        # If move makes pacman adjacent to ghost --> deprioritize immensely
        newGhostStates = successorGameState.getGhostStates()
        for s in newGhostStates:
            if (manhattanDistance(newPos, s.configuration.pos) == 1):
                value = value - 1000 # some absurdly huge value

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore() + value


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # Calculates maximum utility for Pacman
        def calculatePacman(gameState, action, currDepth):
            # Keeps track of overall game state's max utility
            utility = -99999
            finalMove = None

            # Check if game is over and calculate utility
            if currDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            # Generate best action for pacman (Max)
            legalPacmanActions = gameState.getLegalActions(0)
            for action in legalPacmanActions:
                newUtility, _ = calculateGhost(gameState.generateSuccessor(0, action), action, currDepth, 1)
                if newUtility > utility:
                    utility, finalMove = newUtility, action
            return utility, finalMove


        # Calculates maximum utility for a ghost (minimum for Pacman)
        def calculateGhost(gameState, action, currDepth, ghostNum):
            # Keeps track of overall game state's min utility
            utility = 99999
            finalMove = None

            # Check if game is over and calculate utility
            if currDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            numOfAgents = gameState.getNumAgents()
            # Generate best action for ghost (Min)
            legalGhostActions = gameState.getLegalActions(ghostNum)
            for action in legalGhostActions:
                newUtility = utility
                # Switch to Pacman Max function or recurse for next ghost
                if ghostNum == numOfAgents - 1:
                    newUtility, _ = calculatePacman(gameState.generateSuccessor(ghostNum, action), action, currDepth + 1)
                else:
                    newUtility, _ = calculateGhost(gameState.generateSuccessor(ghostNum, action), action, currDepth, ghostNum + 1)
                if newUtility < utility:
                    utility, finalMove = newUtility, action
            
            return utility, finalMove

        _, action = calculatePacman(gameState, None, 0)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Calculates maximum utility for Pacman
        def calculatePacman(gameState, action, currDepth, alpha, beta):
            # Keeps track of overall game state's max utility
            utility = -99999
            finalMove = None

            # Check if game is over and calculate utility
            if currDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            # Generate best action for pacman (Max)
            legalPacmanActions = gameState.getLegalActions(0)
            for action in legalPacmanActions:
                newUtility, _ = calculateGhost(gameState.generateSuccessor(0, action), action, 
                currDepth, 1, alpha, beta)
                if newUtility > utility:
                    utility, finalMove = newUtility, action 
                # Update alpha and check pruning condition
                if utility > beta:
                    return utility, finalMove
                alpha = max(alpha, utility)

            return utility, finalMove


        # Calculates maximum utility for a ghost (minimum for Pacman)
        def calculateGhost(gameState, action, currDepth, ghostNum, alpha, beta):
            # Keeps track of overall game state's min utility
            utility = 99999
            finalMove = None

            # Check if game is over and calculate utility
            if currDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            numOfAgents = gameState.getNumAgents()
            # Generate best action for ghost (Min)
            legalGhostActions = gameState.getLegalActions(ghostNum)
            for action in legalGhostActions:
                newUtility = utility
                # Switch to Pacman Max function or recurse for next ghost
                if ghostNum == numOfAgents - 1:
                    newUtility, _ = calculatePacman(gameState.generateSuccessor(ghostNum, action), action, currDepth + 1, alpha, beta)
                else:
                    newUtility, _ = calculateGhost(gameState.generateSuccessor(ghostNum, action), action, currDepth, ghostNum + 1, alpha, beta)
                    
                if newUtility < utility:
                    utility, finalMove = newUtility, action 
                
                # Update beta and check pruning condition
                if utility < alpha:
                    return utility, finalMove
                beta = min(beta, utility)
            
            return utility, finalMove

        alpha = -99999
        beta = 99999
        _, action = calculatePacman(gameState, None, 0, alpha, beta)
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        # Calculates maximum utility for Pacman
        def calculatePacman(gameState, currDepth):
            # Keeps track of overall game state's max utility
            utility = -99999
            finalMove = None

            # Check if game is over and calculate utility
            if currDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            # Generate best action for pacman (Max)
            legalPacmanActions = gameState.getLegalActions(0)
            for action in legalPacmanActions:
                newUtility = calculateGhost(gameState.generateSuccessor(0, action), currDepth, 1)
                if newUtility > utility:
                    utility, finalMove = newUtility, action
            return utility, finalMove


        # Calculates maximum utility for a ghost (minimum for Pacman)
        def calculateGhost(gameState, currDepth, ghostNum):
            # Keeps track of overall game state's min utility
            utility = 99999

            # Check if game is over and calculate utility
            if currDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            numOfAgents = gameState.getNumAgents()
            # Randomly select action for ghost (Min) by calculating expected value
            legalGhostActions = gameState.getLegalActions(ghostNum)
            
            costs = []
            for action in legalGhostActions:
                successor = gameState.generateSuccessor(ghostNum, action)
                cost = 0
                # Switch to Pacman Max function or recurse for next ghost 
                if ghostNum == numOfAgents - 1:
                        cost, _ = calculatePacman(successor, currDepth + 1)
                else:
                    cost = calculateGhost(successor, currDepth, ghostNum + 1)
                costs.append(cost)

            # calculate expected value
            expectedValue = sum(costs) / len(costs)
            return expectedValue   

        _, action = calculatePacman(gameState, 0)
        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    currPacmanPos = currentGameState.getPacmanPosition()
    currGhostStates = currentGameState.getGhostStates()
    foodGrid = currentGameState.getFood().asList()
    foodLeft = len(foodGrid)
    capsulesLeft = len(currentGameState.getCapsules())

    # Get closest food
    foodDistances = [manhattanDistance(currPacmanPos, food) for food in foodGrid]
    bestDistance = min(foodDistances, default=0)
    bestFoods = [i for i in range(len(foodDistances)) if foodDistances[i] == bestDistance]
    # Distance to closest food
    randomBest = manhattanDistance(currPacmanPos, foodGrid[random.choice(bestFoods)])

    # Check ghost distance from Pacman
    ghostDistance = 0
    for g in currGhostStates:
        ghostDistance = manhattanDistance(currPacmanPos, g.getPosition())
        
        # If pacman is adjacent to ghost --> run away and deprioritize FOOD immensely
        if ghostDistance == 1:
            randomBest = 99999
            
    # Get reciprocal values
    closestValue = (1 / (randomBest + 1) * 1000)
    foodLeftValue = (1 / (foodLeft + 1) * 1000000)
    distValue = ((1 / bestDistance) * 100)
    capsuleValue = (1 / (capsulesLeft + 1) * 10)
    
    return distValue + closestValue + foodLeftValue + capsuleValue + ghostDistance
    

    

# Abbreviation
better = betterEvaluationFunction
