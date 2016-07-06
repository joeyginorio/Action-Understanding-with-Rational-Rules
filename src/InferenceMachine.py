# Joey Velez-Ginorio
# Gridworld Implementation
# ---------------------------------

from GridWorld import GridWorld
from Hypothesis import Hypothesis
from Grid import Grid
from scipy.stats import uniform
from scipy.stats import beta
from scipy.stats import expon
import numpy as np
import copy
import pyprind
import matplotlib.pyplot as plt


class InferenceMachine():
	"""

		Conducts inference over a hypothesis space defined via
		objects and primitives. 

		Can be used to generate all aspects of Bayes' Rule:
			- Prior
			- Likelihood
			- Posterior
		
	"""

	def __init__(self, samples, grid, start, action,discount=.99, tau=.01, epsilon=.01):
		self.sims = list()

		# Key elements of Bayes' Rule
		self.likelihoods = list()
		self.posteriors = list()
		self.prior = None

		# Modify the GridWorld solver
		self.discount = discount
		self.tau = tau
		self.epsilon = epsilon

		self.grid = grid

		self.states = list()
		self.actions = list()

		# Generate separate grids, one for each goal in the original map
		# This will be used later to generate subpolicies for each goal
		# e.g. A policy just for going to A, one just for B, etc.
		objectGrids = np.empty([len(self.grid),len(self.grid[0].objects)], dtype=object)
		for i in range(len(self.grid)):

			for j in range(len(self.grid[0].objects)):
				objectGrids[i][j] = copy.deepcopy(self.grid[i])
				objValue = objectGrids[i][j].objects[self.grid[i].objects.keys()[j]] 
				objectGrids[i][j].objects.clear()
				objectGrids[i][j].objects[self.grid[i].objects.keys()[j]] = objValue

		self.objectGrids = objectGrids

		# Simulates the MDP's needed to begin the inference process
		# In this case, one for each object in the map
		# Subpolicies for each objectGrid done here.
		self.buildBiasEngine()
		self.inferSummary(samples,start,action)

		maxH = np.argwhere(self.posteriors[len(self.posteriors)-1] == np.amax(
			self.posteriors[len(self.posteriors)-1]))
		maxH = maxH.flatten().tolist()

		print "\n"
		for i,index in enumerate(maxH):
			print "Max Hypothesis {}: {}".format(i,self.hypotheses[index])



	def getStateActionVectors(self,gridIndex,start,actions):
		"""
			Generates the state vectors resulting from starting in some 
			state and taking a series of actions. Useful for testing 
			inference over state,action sequences (only have to come
			up with start state and actions, instead of having to manually
			calculate both state,action pairs each time).
		"""

		states = list()
		states.append(start)

		# Cycle through all actions, store the states they take you to
		for i in np.arange(0,len(actions)):

			# State that an action in current state takes me to
			nextState = self.sims[gridIndex][0].takeAction(self.sims[gridIndex][0].scalarToCoord(states[i]),actions[i])
			states.append(self.sims[gridIndex][0].coordToScalar(nextState))

		self.states.append(states)
		self.actions.append(actions)


	def getPolicySwitch(self, gridIndex, hypothesis, states):
		"""
			Generates a vector detailing, according to a hypothesis, when
			to switch policies when iterating across a vector of states.
		"""

		# State location of all goals/object in map
		goalStates = [self.sims[gridIndex][0].coordToScalar(goalCoord) for goalCoord in
					self.grid[gridIndex].objects.values()]

		# Create a dict, mapping goals->state_index
		goalIndex = dict()
		for i in range(len(goalStates)):
			goalIndex[self.grid[gridIndex].objects.keys()[i]] = goalStates[i]

		# Initialize policySwitch vector
		switch = np.empty(len(states), dtype=str)

		# Iterate across states, if you hit current goalState, switch to next goalState
		# as your objective.
		# Essentially, if goals are 'ABC', stay in A until you hit A, then make B the goal
		switchCount = 0
		for i, state in enumerate(states):
			if state == goalIndex[hypothesis[switchCount]] and switchCount + 1 < len(hypothesis):
				switchCount += 1

			switch[i] = hypothesis[switchCount]

		return switch


	def inferSummary(self, samples, start, actions):
		"""
			Provide the prior, likelihood, and posterior distributions 
			for a set of hypotheses. 

			Utilizes Bayes' Rule, P(H|D) ~ P(D|H)P(H)

		"""
		h = Hypothesis(self.grid[0])
		h.sampleHypotheses(samples)
		self.hypotheses = h.hypotheses
		self.primHypotheses = h.primHypotheses

		# Add starting object to map

		self.H = list()
		for i in range(len(self.grid)):

			self.grid[i].objects['S'] = tuple(self.sims[i][0].scalarToCoord(start[i]))

			# Initialize the hypotheis generator
			self.H.append(Hypothesis(self.grid[i]))

			# Setup the distance matrix, for calculating cost of hypotheses
			self.H[i].buildDistanceMatrix()

		evalHypotheses = list()
		# For each hypotheses, evaluate it and return the minCost graphString
		for i in range(len(self.H)):

			bufferHypotheses = list()
			for j in range(len(h.hypotheses)):
				bufferHypotheses.append(self.H[i].evaluate(h.evalHypotheses[j]))

			evalHypotheses.append(bufferHypotheses)

		# Remove the 'S' node from each graph, no longer needed
		# since cost of each graph has been computed

		evalHypothesesCost = evalHypotheses
		self.hypCost = np.copy(evalHypothesesCost)

		for i in range(len(evalHypothesesCost)):
			for j in range(len(evalHypothesesCost[i])):
				evalHypothesesCost[i][j] = abs(evalHypothesesCost[i][j] - max(evalHypothesesCost[i][j]))
				evalHypothesesCost[i][j] = evalHypothesesCost[i][j] - max(evalHypothesesCost[i][j])
				evalHypothesesCost[i][j] = np.exp(evalHypothesesCost[i][j]/self.tau)
				evalHypothesesCost[i][j] /= np.sum(evalHypothesesCost[i][j])

		self.evalHypotheses = h.evalHypotheses
		self.evalHypothesesSM = evalHypothesesCost

		for i in range(len(actions)):

			# Get state,action vectors to conduct inference over
			self.getStateActionVectors(i,start[i],actions[i])

			# Get policySwitch vector to know when to follow which policy
			self.policySwitch = list()
			for j in range(len(self.evalHypotheses)):

				buff = list()
				for k in range(len(self.evalHypotheses[j])):

					##### HERE IS WHERE IT IS MESSED UP
					buff.append(self.getPolicySwitch(i,self.evalHypotheses[j][k], self.states[i]))
				self.policySwitch.append(buff)
				
			# Compute the likelihood for all hypotheses
			self.inferLikelihood(i,self.states[i], self.actions[i], self.policySwitch)
			


		self.inferPrior()

		
		for i in range(len(self.likelihoods)):

			likelihood = 1
			for j in range(i+1):
				likelihood *= np.array(self.likelihoods[j])

			self.inferPosterior(likelihood)


	def inferPrior(self):
		"""

		"""
		self.prior = [1/(i) for i in self.primHypotheses]
		self.prior /= np.sum(self.prior)

	def buildBiasEngine(self):
		""" 
			Simulates the GridWorlds necessary to conduct inference.
		"""

		# Builds/solves gridworld for each objectGrid, generating policies for
		# each object in grid. One for going only to A, another just for B, etc.
		for i in range(len(self.objectGrids)):

			simsBuffer = list()
			for j in range(len(self.objectGrids[0])):
				simsBuffer.append(GridWorld(self.objectGrids[i][j], [10], self.discount, self.tau, self.epsilon))

			self.sims.append(simsBuffer)
		

	def inferLikelihood(self, gridIndex, states, actions, policySwitch):
		"""
			Uses inference engine to inferBias predicated on an agents'
			actions and current state.
		"""

		likelihood = list()

		
		for i in range(len(policySwitch)):
			
			p_sum = 0
			for k in range(len(policySwitch[i])):

				p = 1
				for j in range(len(policySwitch[0][0])-1):

					if states[j] == self.sims[gridIndex][0].coordToScalar(self.grid[gridIndex].objects[self.policySwitch[i][k][j]]):
						p *= self.sims[gridIndex][self.grid[gridIndex].objects.keys().index(policySwitch[i][k][j])].policy[self.sims[gridIndex][0].s[len(self.sims[gridIndex][0].s)-1]][actions[j]]
					else:
						p *= self.sims[gridIndex][self.grid[gridIndex].objects.keys().index(policySwitch[i][k][j])].policy[states[j]][actions[j]]


				p *= self.evalHypothesesSM[gridIndex][i][k]
				p_sum += p

			likelihood.append(p_sum)

		self.likelihoods.append(likelihood)


	def inferPosterior(self, likelihood):
		"""
			Uses inference engine to compute posterior probability from the 
			likelihood and prior (beta distribution).
		"""

		posterior = likelihood * self.prior
		posterior /= posterior.sum()
		self.posteriors.append(posterior)


#################### Testing ############################

test1 = True
test2 = False
test3 = False
test4 = False


if test1:
	""" Test 1 """
	# Testing 'A'

	testGrid = Grid('testGrid')
	testGrid2 = Grid('testGrid2')
	start = [8,10]
	actions = [[0,0],[3,3,0,0,3]]

	infer = InferenceMachine(100, [testGrid,testGrid2], start, actions)

if test2:
	""" Test 2 """
	# Testing Or(A,B)

	testGrid = Grid('testGrid')
	testGrid2 = Grid('testGrid2')
	start = [8,10]
	actions = [[0,0],[0,0]]

	infer = InferenceMachine(100, [testGrid,testGrid2], start, actions)

if test3:
	""" Test 3 """
	# Testing 'Then(A,B)'

	testGrid = Grid('testGrid')
	testGrid2 = Grid('testGrid2')
	start = [8,9]
	actions = [[0,0,3],[0,2,2]]

	infer = InferenceMachine(1000, [testGrid,testGrid2], start, actions)

if test4:
	""" Test 4 """
	# Testing 'And(A,B)'

	testGrid = Grid('testGrid')
	testGrid2 = Grid('testGrid2')
	start = [8,10]
	actions = [[0,0,3],[0,0,3]]

	infer = InferenceMachine(1000, [testGrid,testGrid2], start, actions)

# top 10 hypotheses