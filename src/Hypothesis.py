# Joey Velez-Ginorio
# Hypothesis Implementation
# ---------------------------------

import numpy as np
import copy
from collections import OrderedDict
from GridWorld import GridWorld
from Grid import Grid
from scipy.stats import uniform
from scipy.stats import beta

import itertools

# TODO LIST:
# 1.) Randomly Sample Hypotheses
# 2.) Form prior
# Write a function for multinomial for each derivation, stop symbol
# write a recursive function till it calls a terminal function

class Hypothesis():
	"""

		Provides functions to generate hypotheses from primitives and objects
		in some GridWorld.

		Primitives:
			- And
			- Or
			- Then

	"""
	def __init__(self, grid, occam=.4):		
		self.grid = grid
		self.hypotheses = None
		self.occam = occam

		self.primitives = list()
		self.primitives = [self.And,self.Or,self.Then]
		self.objects = grid.objects.keys()
		self.space = [self.primitives, self.objects]

		# Uniform pDistribution for sampling across objects and primitivres
		self.pPrim = np.zeros(len(self.primitives))
		self.pPrim = uniform.pdf(self.pPrim)
		self.pPrim /= self.pPrim.sum()

		self.pObj = np.zeros(len(self.objects))
		self.pObj = uniform.pdf(self.pObj)
		self.pObj /= self.pObj.sum()

		self.p = [self.pPrim,self.pObj]

		self.primCount = 0
		self.primHypotheses = list()

		self.setBetaDistribution()

	def sampleHypotheses(self, samples):
		"""

		"""

		self.hypotheses = list()

		while len(self.hypotheses) != samples:

			self.resetPrimCount()
			self.setBetaDistribution()

			hypothesis = self.hGenerator()

			if hypothesis not in self.hypotheses: 
				self.hypotheses.append(hypothesis)
				self.primHypotheses.append((self.primCount / self.occam)+1)


		self.evalHypotheses = [eval(i) for i in self.hypotheses]
		self.hypotheses = [i.replace('self.','') for i in self.hypotheses]

	def resetPrimCount(self):
		self.primCount = 0

	def hGenerator(self, arg=None):
		"""
		"""
		choice = np.random.choice([0,1],p=self.choosePrimObj)
		arg = np.random.choice(self.space[choice],p=self.p[choice])


		if choice == 1:
			return "'" + arg + "'"

		self.primCount += self.occam
		if arg == self.And:
			self.primCount += self.occam

		self.setBetaDistribution()


		arg1 = self.hGenerator()
		arg2 = self.hGenerator()

		while arg1 == arg2:
			arg2 = self.hGenerator()
		
		return 'self.' + arg.__name__ + '(' + arg1 + ',' + arg2 + ')'

	def setBetaDistribution(self):
		"""
		"""
		choosePrimObj = [.25,.75]
		choosePrimObj = beta.pdf(choosePrimObj,1.3+self.primCount,1)
		choosePrimObj /= choosePrimObj.sum()
		self.choosePrimObj = choosePrimObj


	def evaluate(self, graphList):
		"""
			Takes in a list of graphStrings, then evaluates by:
			
				1.) Linking graphs to a start node
				2.) Finding minimum cost graphString
				3.) Removing repeated nodes e.g. 'AAB' -> 'AB'
				4.) Returning the optimal path from a series of graphs

			*****
			Cannot call this until starting state 'S' is inserted
			in the grid!
			*****
		"""

		# Attach graphs to start node
		graphList = self.linkGraphStrings(graphList)

		# Find cheapest path
		graphString = self.minCostGraphString(graphList)

		return graphString 


	def buildDistanceMatrix(self):
		"""
			Using GridWorld, creates a distance matrix, detailing the costs 
			to go from any object to another.
		"""

		# Initialize distance matrix
		dist = np.zeros([len(self.grid.objects), len(self.grid.objects)])

		# For each object in gridworld
		for i in range(len(self.grid.objects)):
			for j in range(len(self.grid.objects)):

				# Compute distance from each object to another
				dist[i][j] = self.objectDist(self.grid.objects.keys()[j],
					self.grid.objects.keys()[i])

		# Hold onto distance as instance variable
		self.dist = dist

	def objectDist(self, start, obj):
		"""
			Return cost of going to some object
		"""

		# Generate a grid that only cares about 
		# getting to the input 'obj'
		objectGrid = copy.deepcopy(self.grid)
		objValue = objectGrid.objects[obj] 
		objectGrid.objects.clear()
		objectGrid.objects[obj] = objValue

		# Simulate GridWorld where only goal is obj
		objectWorld = GridWorld(objectGrid, [10])
		startCoord = self.grid.objects[start]

		# Count num. of steps to get to obj from start, that is the distance
		dist = objectWorld.simulate(objectWorld.coordToScalar(startCoord))

		return dist 

	def Or(self, A, B):
		"""
			Primitive function to do the 'Or' operation. 
			Essentially throws the contents of A and B into
			one list (of subgraphs). 

			e.g. A:['A'], B:['A,B'] ; Or(A,B):['A','A','B']
		"""

		# If input is a char, turn into numpy array
		if type(A) is not np.ndarray:
			A = np.array([A])
		if type(B) is not np.ndarray:
			B = np.array([B])

		return np.append(A,B)
		 

	def Then(self, A, B):
		"""
			Primitive function to do the 'Then' operation.
			Adds every possible combination of A->B for all content
			within A and B to a list. 

			e.g. A:['A'], B:['A,B'] ; Then(A,B):['AA','AB']
		"""

		# If input is a char, turn into numpy array
		if type(A) is not np.ndarray:
			A = np.array([A])
		if type(B) is not np.ndarray:
			B = np.array([B])

		# C will hold all combinations of A->B
		C = np.array([])
		for i in range(len(A)):
			for j in range(len(B)):

				if A[i][-1] == B[j][0]:
					C = np.append(C, A[i]+B[j][1:])

				else:
					C = np.append(C, A[i] + B[j])

		return C

	def And(self, A, B):
		"""
			Primitive function to do the 'And' operation.
			Defined as a composition of Or and Then.

			And(A,B) = Or(Then(A,B),Then(B,A))
		"""

		# If input is a char, turn into numpy array
		if type(A) is not np.ndarray:
			A = np.array([A])
		if type(B) is not np.ndarray:
			B = np.array([B])

		return self.Or(self.Then(A,B), self.Then(B,A))


	def minCostGraphString(self, graphList):
		"""
			Considers multiple graphStrings, returns the one
			that costs the least. 

			e.g. graphList = ['AB','AC'], 'AB'=2 'AC'=4
			returns 'AB'
		"""

		# Cost for each graphString in graphList
		costGraphList = np.zeros(len(graphList))

		# Cycle through the list, calculate cost
		for i, graphString in enumerate(graphList):
			costGraphList[i] = self.costGraphString(graphString)

		# Return cheapest graphString
		return costGraphList


	def linkGraphStrings(self, graphList):
		"""
			Join all the graphStrings into one tree, by 
			attaching 'S', the start node, to all graphs.
		"""
		return ['S' + graphString for graphString in graphList]

	def costGraphString(self, graphString):
		"""
			Iterates through a graphString and computes the
			cost.
		"""

		# Check distance between 2 Goals at a time, add to
		# running sum of cost. e.g. 'ABC' = cost('AB') + cost('BC')
		cost = 0
		for i in range(len(graphString)):
			
			# If substring has only one char, stop computing cost
			if len(graphString[i:i+2]) == 1:
				break

			cost += self.costEdge(graphString[i:i+2])

		return cost


	def costEdge(self, edgeString):
		"""
			Computes cost of an edge in the graphString.
			An edge is any two adjacent characters in 
			a graphString e.g. 'AB' in 'ABCD'
 
		"""

		# Find index of object, to use for indexing distance matrix
		objIndex1 = self.grid.objects.keys().index(edgeString[0])
		objIndex2 = self.grid.objects.keys().index(edgeString[1])

		return self.dist[objIndex1][objIndex2]

