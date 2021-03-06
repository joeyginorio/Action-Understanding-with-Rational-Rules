# Joey Velez-Ginorio
# Gridworld Implementation
# ---------------------------------

from mdp import MDP
from grid import Grid
from scipy.stats import uniform
from scipy.stats import beta
from scipy.stats import expon
import numpy as np
import random
import pyprind
import matplotlib.pyplot as plt

class GridWorld(MDP):
	"""
		 Defines a gridworld environment to be solved by an MDP!
	"""
	def __init__(self, grid, goalVals, discount=.99, tau=.01, epsilon=.001):

		MDP.__init__(self, discount=discount, tau=tau, epsilon=epsilon)

		self.goalVals = goalVals
		self.grid = grid

		self.setGridWorld()
		self.valueIteration()
		self.extractPolicy()


	def isTerminal(self, state):
		"""
			Specifies terminal conditions for gridworld.
		"""
		return True if tuple(self.scalarToCoord(state)) in self.grid.objects.values() else False

	def isObstacle(self, sCoord):
		""" 
			Checks if a state is a wall or obstacle.
		"""
		if tuple(sCoord) in self.grid.walls:
			return True

		if sCoord[0] > (self.grid.row - 1) or sCoord[0] < 0:
			return True

		if sCoord[1] > (self.grid.col - 1) or sCoord[1] < 0:
			return True

		return False

	def takeAction(self, sCoord, action):
		"""
			Receives an action value, performs associated movement.
		"""
		if action is 0:
			return self.up(sCoord)

		if action is 1:
			return self.down(sCoord)

		if action is 2:
			return self.left(sCoord)

		if action is 3:
			return self.right(sCoord)

		if action is 4:
			return sCoord

		if action is 5:
			return self.upleft(sCoord)

		if action is 6:
			return self.upright(sCoord)

		if action is 7:
			return self.downleft(sCoord)

		if action is 8:
			return self.downright(sCoord)


	def up(self, sCoord):
		"""
			Move agent up, uses state coordinate.
		"""
		newCoord = np.copy(sCoord)
		newCoord[0] -= 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord
	
	def upright(self, sCoord):
		"""
			Move agent up and right, uses state coordinate.
		"""
		newCoord = np.copy(sCoord)
		newCoord[0] -= 1
		newCoord[1] += 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord

	def upleft(self, sCoord):
		"""
			Move agent up and left, uses state coordinate.
		"""
		newCoord = np.copy(sCoord)
		newCoord[0] -= 1
		newCoord[1] -= 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord


	def down(self, sCoord):
		"""
			Move agent down, uses state coordinate.
		"""
		newCoord = np.copy(sCoord)
		newCoord[0] += 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord

	def downleft(self, sCoord):
		"""
			Move agent down, uses state coordinate.
		"""
		newCoord = np.copy(sCoord)
		newCoord[0] += 1
		newCoord[1] -= 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord

	def downright(self, sCoord):
		"""
			Move agent down, uses state coordinate.
		"""
		newCoord = np.copy(sCoord)
		newCoord[0] += 1
		newCoord[1] += 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord

	def left(self, sCoord):
		"""
			Move agent left, uses state coordinate.
		"""
		newCoord = np.copy(sCoord)
		newCoord[1] -= 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord

	def right(self, sCoord):
		"""
			Move agent right, uses state coordinate.
		"""

		newCoord = np.copy(sCoord)
		newCoord[1] += 1

		# Check if action takes you to a wall/obstacle
		if not self.isObstacle(newCoord):
			return newCoord

		# You hit a wall, return original coord
		else:
			return sCoord

	def coordToScalar(self, sCoord):
		""" 
			Convert state coordinates to corresponding scalar state value.
		"""
		return sCoord[0]*(self.grid.col) + sCoord[1]

	def scalarToCoord(self, scalar):
		"""
			Convert scalar state value into coordinates.
		"""
		return np.array([scalar / self.grid.col, scalar % self.grid.col])

	def getPossibleActions(self, sCoord):
		"""
			Will return a list of all possible actions from a current state.
		"""
		possibleActions = list()

		if self.up(sCoord) is not sCoord:
			possibleActions.append(0)

		if self.down(sCoord) is not sCoord:
			possibleActions.append(1)

		if self.left(sCoord) is not sCoord:
			possibleActions.append(2)

		if self.right(sCoord) is not sCoord:
			possibleActions.append(3)
		
		if self.upleft(sCoord) is not sCoord:
			possibleActions.append(5)

		if self.upright(sCoord) is not sCoord:
			possibleActions.append(6)

		if self.downleft(sCoord) is not sCoord:
			possibleActions.append(7)

		if self.downright(sCoord) is not sCoord:
			possibleActions.append(8)
		
		return possibleActions
	def setGridWorld(self):
		"""
			Initializes states, actions, rewards, transition matrix.
		"""

		# Possible coordinate positions + Death State
		self.s = np.arange(self.grid.row*self.grid.col + 1)

		# 4 Actions {Up, Down, Left, Right}
		self.a = np.arange(9)

		# Reward Zones
		self.r = np.zeros(len(self.s))
		for i in range(len(self.grid.objects)):
			self.r[self.coordToScalar(self.grid.objects.values()[i])] = self.goalVals[i]

		self.r_sa = np.zeros([len(self.s),len(self.a)])
		for i in range(len(self.s)):
			for j in range(len(self.a)):
				if j <= 4:
					self.r_sa[i][j] = self.r[self.coordToScalar(self.takeAction(self.scalarToCoord(i),j))]-1.0
				else:
					self.r_sa[i][j] = self.r[self.coordToScalar(self.takeAction(self.scalarToCoord(i),j))]-np.sqrt(2)

		self.r = self.r_sa


		# Transition Matrix
		self.t = np.zeros([len(self.s),len(self.a),len(self.s)])

		for state in range(len(self.s)):
			possibleActions = self.getPossibleActions(self.scalarToCoord(state))

			if self.isTerminal(state):

				for i in range(len(self.a)):

					if i == 4:
						self.t[state][4][state]=1.0

					else:
						self.t[state][i][len(self.s)-1] = 1.0

				continue

			for action in self.a:

				# Up
				if action == 0:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 0)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0


				if action == 1:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 1)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0

				if action == 2:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 2)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0

				if action == 3:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 3)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0

				if action == 4:
					self.t[state][action][state] = 1.0

				if action == 5:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 5)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0

				if action == 6:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 6)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0

				if action == 7:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 7)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0

				if action == 8:

					currentState = self.scalarToCoord(state)

					nextState = self.takeAction(currentState, 8)
					self.t[state][action][self.coordToScalar(nextState)] = 1.0




	def simulate(self, state):

		""" 
			Runs the solver for the MDP, conducts value iteration, extracts policy,
			then runs simulation of problem.

			NOTE: Be sure to run value iteration (solve values for states) and to
		 	extract some policy (fill in policy vector) before running simulation
		"""
		
		# Run simulation using policy until terminal condition met
		
		actions = ['up', 'down', 'left', 'right']
		count = 0

		while not self.isTerminal(state):

			# Determine which policy to use (non-deterministic)
			policy = self.policy[np.where(self.s == state)[0][0]]
			p_policy = self.policy[np.where(self.s == state)[0][0]] / \
						self.policy[np.where(self.s == state)[0][0]].sum()

			# Get the parameters to perform one move
			stateIndex = np.where(self.s == state)[0][0]
			policyChoice = np.random.choice(policy, p=p_policy)
			actionIndex = np.random.choice(np.array(np.where(self.policy[state][:] == policyChoice)).ravel())
			
			# print actionIndex
			if actionIndex <= 3:
				count += 1
			else:
				count += np.sqrt(2)

			# Take an action, move to next state
			nextState = self.takeAction(self.scalarToCoord(int(stateIndex)), int(actionIndex))
			nextState = self.coordToScalar(nextState)

			# print "In state: {}, taking action: {}, moving to state: {}".format(
				# self.scalarToCoord(state), actions[actionIndex], self.scalarToCoord(nextState))

			# End game if terminal state reached
			state = int(nextState)
			# if self.isTerminal(state):

				# print "Terminal state: {} has been reached. Simulation over.".format(self.scalarToCoord(state))
		
		return count










