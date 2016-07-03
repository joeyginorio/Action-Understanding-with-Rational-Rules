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

	def __init__(self, grid, discount=.99, tau=.01, epsilon=.01):
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
		objectGrids = np.empty(len(self.grid.objects), dtype=object)
		for i in range(len(self.grid.objects)):
			objectGrids[i] = copy.deepcopy(self.grid)
			objValue = objectGrids[i].objects[self.grid.objects.keys()[i]] 
			objectGrids[i].objects.clear()
			objectGrids[i].objects[self.grid.objects.keys()[i]] = objValue

		self.objectGrids = objectGrids

		# Simulates the MDP's needed to begin the inference process
		# In this case, one for each object in the map
		# Subpolicies for each objectGrid done here.
		self.buildBiasEngine()


	def getStateActionVectors(self,start,actions):
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
			nextState = self.sims[0].takeAction(self.sims[0].scalarToCoord(states[i]),actions[i])
			states.append(self.sims[0].coordToScalar(nextState))

		self.states.append(states)
		self.actions.append(actions)


	def getPolicySwitch(self, hypothesis, states):
		"""
			Generates a vector detailing, according to a hypothesis, when
			to switch policies when iterating across a vector of states.
		"""

		# State location of all goals/object in map
		goalStates = [self.sims[0].coordToScalar(goalCoord) for goalCoord in
					self.grid.objects.values()]

		# Create a dict, mapping goals->state_index
		goalIndex = dict()
		for i in range(len(goalStates)):
			goalIndex[self.grid.objects.keys()[i]] = goalStates[i]

		# Initialize policySwitch vector
		switch = np.empty(len(states), dtype=str)
		switchCount = 0

		# Iterate across states, if you hit current goalState, switch to next goalState
		# as your objective.
		# Essentially, if goals are 'ABC', stay in A until you hit A, then make B the goal
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
		h = Hypothesis(self.grid)
		h.sampleHypotheses(samples)
		self.hypotheses = h.hypotheses
		self.primHypotheses = h.primHypotheses

		# Add starting object to map
		self.grid.objects['S'] = tuple(self.sims[0].scalarToCoord(start))

		# Initialize the hypotheis generator
		self.H = Hypothesis(self.grid)

		# Setup the distance matrix, for calculating cost of hypotheses
		self.H.buildDistanceMatrix()

		evalHypotheses = h.evalHypotheses
		# For each hypotheses, evaluate it and return the minCost graphString
		for i in range(len(h.hypotheses)):
			evalHypotheses[i] = self.H.evaluate(h.evalHypotheses[i])

		# Remove the 'S' node from each graph, no longer needed
		# since cost of each graph has been computed
		evalHypotheses = [hyp[1:] for hyp in evalHypotheses]
		self.evalHypotheses = evalHypotheses

		###########
		########### Loop here over all state in list , over all actions in list
		###########

		for i in range(len(actions)):

			# Get state,action vectors to conduct inference over
			self.getStateActionVectors(start,actions[i])

			# Get policySwitch vector to know when to follow which policy
			self.policySwitch = list()
			for j in range(len(h.hypotheses)):
				self.policySwitch.append(self.getPolicySwitch(h.evalHypotheses[j], self.states[i]))

			# Compute the likelihood for all hypotheses
			self.inferLikelihood(self.states[i], self.actions[i], self.policySwitch)
			


		self.inferPrior()

		# Write loop to generates all posteriors
		#####################################
		#####################################
		##

		
		for i in range(len(self.likelihoods)):

			likelihood = 1
			for j in range(i+1):
				likelihood *= np.array(self.likelihoods[j])

			self.inferPosterior(likelihood)

		# self.inferPosterior(state, action)
		# self.expectedPosterior()
		# self.plotDistributions()

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
			self.sims.append(GridWorld(self.objectGrids[i], [10], self.discount, self.tau, self.epsilon))
			


	def inferLikelihood(self, states, actions, policySwitch):
		"""
			Uses inference engine to inferBias predicated on an agents'
			actions and current state.
		"""

		likelihood = list()

		
		for i in range(len(policySwitch)):
			
			p = 1
			for j in range(len(policySwitch[0])-1):

				if states[j] == self.sims[0].coordToScalar(self.grid.objects[self.policySwitch[i][j]]):
					p *= self.sims[self.grid.objects.keys().index(policySwitch[i][j])].policy[self.sims[0].s[len(self.sims[0].s)-1]][actions[j]]
				
				else:
					p *= self.sims[self.grid.objects.keys().index(policySwitch[i][j])].policy[states[j]][actions[j]]

			likelihood.append(p)

		self.likelihoods.append(likelihood)


	def inferPosterior(self, likelihood):
		"""
			Uses inference engine to compute posterior probability from the 
			likelihood and prior (beta distribution).
		"""

		posterior = likelihood * self.prior
		posterior /= posterior.sum()
		self.posteriors.append(posterior)



	




#######################################################################
#################### Testing ##########################################


testGrid = Grid('testGrid')
H = Hypothesis(testGrid)
infer = InferenceMachine(testGrid)

# Define starting state, proceeding actions
start = 8
actions = [[3],[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]]


# Test Hypotheses
infer.inferSummary(1000,start,actions)
# print "\nHypotheses: \n{}".format(infer.hypotheses)
# print "================================="

# for i in range(len(actions)):

# 	print "Event {}\n-------".format(i)

# 	print "Likelihoods: \n{}".format(infer.likelihoods[i])
# 	print "States: \n{}".format(infer.states[i])
# 	print "Actions: \n{}".format(infer.actions[i])

# 	print "\n"


