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

	def __init__(self, depth, grid, start, action, reward=100, 
		hypotheses = None, discount=.9, tau=.01, epsilon=.001, 
		tauChoice=.01, rationalAction=1, rationalChoice=1):

		self.sims = list()
		self.temp = list()

		# Key elements of Bayes' Rule
		self.likelihoods = list()
		self.posteriors = list()
		self.prior = None

		# Modify the GridWorld solver
		self.discount = discount
		self.epsilon = epsilon

		self.reward = reward

		self.grid = grid

		self.states = list()
		self.actions = list()

		# Alternate models
		if rationalChoice == 0:
			self.tauChoice = 100000
		else:
			self.tauChoice = tauChoice

		if rationalAction == 0:
			self.tau = 100000
		else:
			self.tau = tau




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

		# Rewards for objects
		self.rewards = list()
		for i in range(len(self.objectGrids[0])):
			self.rewards.append(10)


		# Simulates the MDP's needed to begin the inference process
		# In this case, one for each object in the map
		# Subpolicies for each objectGrid done here.
		self.buildBiasEngine()
		self.inferSummary(depth,start,action,hypotheses)

		maxH = np.argwhere(self.posteriors[len(self.posteriors)-1] == np.amax(
			self.posteriors[len(self.posteriors)-1]))
		maxH = maxH.flatten().tolist()

		print "\n"
		for i,index in enumerate(maxH):
			print "Max Hypothesis {}: {}".format(i,self.hypotheses[index])

		self.hypPosterior = dict(zip(self.hypotheses,self.posteriors[len(self.posteriors)-1]))
		self.maxHyp = sorted(self.hypPosterior,key=self.hypPosterior.get,reverse=True)

		print "\n"

		limit = 10
		if len(self.hypotheses) < 10:
			limit = len(self.hypotheses)

		for i in range(limit):
			print "Hypothesis {}: {} : {}".format(i+1,self.maxHyp[i],self.hypPosterior[self.maxHyp[i]])

		names = self.maxHyp[0:10]
		data = [self.hypPosterior[i] for i in self.maxHyp[0:10]]

		fig = plt.figure(figsize=(25,25))
		fig.subplots_adjust(bottom=0.39)
		ax = plt.subplot(111)
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(data)+1))
		ax.bar(bins,data,width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(data)+1)))
		ax.set_xticklabels(names,rotation=45, rotation_mode="anchor", ha="right")
		# plt.show(block=False)



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

			if type(actions[i]) is str:
				states.append(self.sims[gridIndex][0].coordToScalar(nextState))
				continue

			# State that an action in current state takes me to
			nextState = self.sims[gridIndex][0].takeAction(self.sims[gridIndex][0].scalarToCoord(states[i]),actions[i])
			states.append(self.sims[gridIndex][0].coordToScalar(nextState))

		self.states.append(states)
		self.actions.append(actions)


	def getPolicySwitch(self, gridIndex, hypothesis, states, actions):
		"""
			Generates a vector detailing, according to a hypothesis, when
			to switch policies when iterating across a vector of states.
		"""
		for i in range(len(self.grid[gridIndex].objects)):
			hypothesis = hypothesis.replace(self.grid[gridIndex].objects.keys()[i],str(i))


		# State location of all goals/object in map
		goalStates = [self.sims[gridIndex][0].coordToScalar(goalCoord) for goalCoord in
					self.grid[gridIndex].objects.values()]

		# Create a dict, mapping goals->state_index
		goalIndex = dict()
		for i in range(len(goalStates)):
			goalIndex[str(i)] = goalStates[i]

		# Initialize policySwitch vector
		switch = np.empty(len(states), dtype=str)

		# Iterate across states, if you hit current goalState, switch to next goalState
		# as your objective.
		# Essentially, if goals are 'ABC', stay in A until you hit A, then make B the goal
		switchCount = 0
		tCounter = 0
		for i, state in enumerate(states):

			if state == goalIndex[hypothesis[switchCount]] and switchCount + 1 < len(hypothesis):

				if i < len(states)-1 and switchCount <= len(hypothesis)-1:
					if actions[i] == 'take':
						switchCount += 1
							
			switch[i] = hypothesis[switchCount]
			
		
		temp = copy.deepcopy(switch[:-1])
		switch[1:] = temp

		if actions[-1] == 'stop' and actions.count('take') < len(hypothesis) or actions.count('take') > len(hypothesis):
			switch[-1] = str(int(switch[-1]) + 1)

		return switch


	def inferSummary(self, depth=None, start=None, actions=None, hypotheses=None):
		"""
			Provide the prior, likelihood, and posterior distributions 
			for a set of hypotheses. 

			Utilizes Bayes' Rule, P(H|D) ~ P(D|H)P(H)

		"""

		if hypotheses is None:
			h = Hypothesis(self.grid[0])
			h.BFSampler(depth)
			self.hypotheses = h.hypotheses
			self.primHypotheses = h.primHypotheses

		else:
			h = hypotheses
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
				evalHypothesesCost[i][j] = self.reward - evalHypothesesCost[i][j]
				evalHypothesesCost[i][j][-1] = 0.0

				evalHypothesesCost[i][j] /= self.tauChoice
				maxCost = max(evalHypothesesCost[i][j])
				arg = maxCost + np.log((np.exp(evalHypothesesCost[i][j] - maxCost).sum()))
				evalHypothesesCost[i][j] -= arg
				evalHypothesesCost[i][j] = np.exp(evalHypothesesCost[i][j])

				# evalHypothesesCost[i][j] = np.exp(evalHypothesesCost[i][j]/self.tauChoice)
				# evalHypothesesCost[i][j] /= np.sum(evalHypothesesCost[i][j])

		self.evalHypotheses = h.evalHypotheses
		self.evalHypothesesSM = evalHypothesesCost

		for i in range(len(actions)):

			# Get state,action vectors to conduct inference over
			self.getStateActionVectors(i,start[i],actions[i])

			# Get policySwitch vector to know when to follow which policy
			self.policySwitch = list()
			for j in range(len(self.evalHypotheses)):

				if type(self.evalHypotheses[j]) is str:
					self.evalHypotheses[j] = np.array([self.evalHypotheses[j]])

				buff = list()
				for k in range(len(self.evalHypotheses[j])):


					##### HERE IS WHERE IT IS MESSED UP
					buff.append(self.getPolicySwitch(i,self.evalHypotheses[j][k], self.states[i],self.actions[i]))
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
		self.prior = [1.0/(i) for i in self.primHypotheses]
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
		temp2 = list()
		
		actions = [4 if type(i) is str else i for i in actions]

		for i in range(len(policySwitch)):
			temp1 = list()
			p_sum = 0
			for k in range(len(policySwitch[i])):
				
				p = 1
				for j in range(len(policySwitch[0][0])-1):

					if states[j] == self.sims[gridIndex][0].coordToScalar(self.grid[gridIndex].objects.values()[int(self.policySwitch[i][k][j])]):
						if actions[j] != 4: 
							p *= self.sims[gridIndex][int(policySwitch[i][k][j])].policy[self.sims[gridIndex][0].s[len(self.sims[gridIndex][0].s)-1]][actions[j]]

					else:
						p *= self.sims[gridIndex][int(policySwitch[i][k][j])].policy[states[j]][actions[j]]

				if policySwitch[i][k][j] != policySwitch[i][k][j+1]:
					p *= 0
 

				p *= self.evalHypothesesSM[gridIndex][i][k]
				p_sum += p
				temp1.append(p)

			likelihood.append(p_sum)
			temp2.append(temp1)

		self.likelihoods.append(likelihood)
		self.temp = temp2

	def inferPosterior(self, likelihood):
		"""
			Uses inference engine to compute posterior probability from the 
			likelihood and prior (beta distribution).
		"""

		posterior = likelihood * self.prior
		posterior /= posterior.sum()
		self.posteriors.append(posterior)




# desires are not states of the world, but given a desire i can infer the states
# of the world #
