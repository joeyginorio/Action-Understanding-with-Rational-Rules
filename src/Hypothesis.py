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
	def __init__(self, grid, occam=2.0):		
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

	def sampleHypotheses(self, samples=None, hypotheses=None):
		"""

		"""

		self.hypotheses = list()
		self.evalHypotheses = list()
		self.primHypotheses = list()

		if hypotheses is None:
			while len(self.hypotheses) != samples:

				self.resetPrimCount()
				self.setBetaDistribution()

				hypothesis = self.hGenerator()
				evalHypothesis = eval(hypothesis)
				if type(evalHypothesis) is not str:
					evalHypothesis = evalHypothesis.tolist()

				if type(evalHypothesis) is not list:
					evalHypothesis = [evalHypothesis]

				if evalHypothesis in self.evalHypotheses:
					currentCount = (self.primCount / self.occam)+1
					index = self.evalHypotheses.index(evalHypothesis)
					if currentCount < self.primHypotheses[index]:
						self.hypotheses[index] = hypothesis
						self.evalHypotheses[index] = evalHypothesis
						self.primHypotheses[index] = currentCount
						continue

				if hypothesis not in self.hypotheses and evalHypothesis not in self.evalHypotheses: 
					self.hypotheses.append(hypothesis)
					self.evalHypotheses.append(evalHypothesis)
					self.primHypotheses.append((self.primCount / self.occam)+1)

		else:
			for i in range(len(hypotheses)):
				primCount = 0
				self.hypotheses.append(hypotheses[i])
				self.evalHypotheses.append(eval(hypotheses[i]))
				primCount += (hypotheses[i].count("And"))
				primCount += hypotheses[i].count("Or")
				primCount += hypotheses[i].count("Then")
				primCount += 1

				self.primHypotheses.append(primCount)

		self.hypotheses = [i.replace('self.','') for i in self.hypotheses]
		self.evalHypotheses = [np.array(i,dtype=object) if type(i) is list else i for i in self.evalHypotheses]

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
		# if arg == self.And:
		# 	self.primCount += (self.occam*2)

		self.setBetaDistribution()

		arg1 = self.hGenerator()
		arg2 = self.hGenerator()

		# while arg1 == arg2:
		# 	arg2 = self.hGenerator()
		
		return 'self.' + arg.__name__ + '(' + arg1 + ',' + self.hGenerator() + ',' + arg2 + ')'

	
	# def BFSampler(self, depth):
	# 	"""
	# 	Breadth-first-search sampler.
	# 	"""

	# 	# Edge Case
	# 	if depth == 1:
	# 		return self.objects

	# 	baseHyp = []
	# 	final = set()
	# 	finalEval = set()

	# 	objects = ["np.array([" + "'" + i + "'"  + "],dtype=object)" for i in self.objects]
	# 	evalObjects = [eval(i) for i in objects]


	# 	self.hypotheses = list()
	# 	self.evalHypotheses = list()
	# 	self.primHypotheses = list()

	# 	for i in ["'" + j + "'" for j in self.objects]: self.hypotheses.append(i)
	# 	for i in evalObjects: 
	# 		self.evalHypotheses.append(i)
	# 		self.primHypotheses.append(1.0)
	# 		finalEval.add(tuple(i))

	# 	for i in range(2, depth+1):

	# 		temp = list(itertools.product(self.objects, repeat=i))
	# 		temp = [list(e) for e in temp]
	# 		temp += baseHyp
			
	# 		for j in temp:

	# 			# Set we will use to check if evaluated hypothesis is same
	# 			if callable(j[0]):
	# 				bufferEval = set()
	# 				parsed = self.parse(j,len(j)-1)
	# 				# print i, j
	# 				# print parsed
	# 				bufferEval.add(tuple(np.sort(eval(parsed))))
	# 				if len(finalEval.intersection(bufferEval)) == 0:
	# 					final.add(tuple(j))
	# 					self.hypotheses.append(parsed.replace('self.',''))
	# 					self.evalHypotheses.append(np.array(np.sort(eval(parsed)),dtype=object))
	# 					self.primHypotheses.append(1.0 + len([k for k in j if callable(k)]))
	# 					finalEval.add(tuple(np.sort(eval(parsed))))

	# 			for k in self.helper(list(j), i):
	# 				bufferEval = set()
	# 				parsed = self.parse(k,len(k)-1)
	# 				# print parsed, i, k
	# 				bufferEval.add(tuple(np.sort(eval(parsed))))
	# 				if len(finalEval.intersection(bufferEval)) == 0:
	# 					final.add(tuple(k))
	# 					self.hypotheses.append(parsed.replace('self.',''))
	# 					self.evalHypotheses.append(np.array(np.sort(eval(parsed)),dtype=object))
	# 					self.primHypotheses.append(1.0 + len([m for m in k if callable(m)]))
	# 					finalEval.add(tuple(np.sort(eval(parsed))))
				
	# 		baseHyp = final

	# 	# 	################
	# 	# self.hypotheses = [i.replace('self.','') for i in finalHypList]
	# 	# self.evalHypotheses = [eval(i) for i in finalHypList]
		
	# 	# self.primHypotheses = list()
	# 	# for i in fullHypList:
	# 	# 	self.primHypotheses.append(1.0 + len([j for j in i if callable(j)]))
	# 	# 	################

	# 	self.finalEval = list(finalEval)
	# 	self.finalHypList = list(final)

	# 	# self.hypParser(list(final))



	def helper(self, hypList, depth):
		"""
			takes in a list of arguments, returns all of them with primitives
			mixed in.
		"""

		hypLimit = len(hypList) - 2
		temp = list()
		finalHypList = list()

		if hypLimit < 2 and callable(hypList[0]):
			finalHypList.append(hypList)
			return finalHypList


		# Loop through all primitives to add

		# primitive at first index
		# FIX THIS DUUUUUUUUUUUUUUUDDDEEEEEEEE
		if(callable(hypList[0])):
			
			for i in self.primitives:
				j = 1

				while(j <= hypLimit):

					temp = hypList[:]
					temp.insert(j, i)
					if temp not in finalHypList:
						finalHypList.append(temp)
					j += 1

		# No primitive at first index
		else:	
			for i in self.primitives:
				temp = hypList[:]
				temp.insert(0,i)
				if temp not in finalHypList:
					finalHypList.append(temp)


		return finalHypList

	def parse(self, hypothesis, coord):
		hyp = ''

		# temp = np.array([])
		# for i in range(len(hypothesis)):
		# 	if callable(hypothesis[i]):
		# 		temp = np.append(temp, 1 + i + depth)
		# 		depth -= 2

		hypothesis = hypothesis[:]

		offset = 0
		for i in coord:
			hypothesis.insert(i+1+offset,')')
			offset += 1
		hypothesis += ')'

		for i in range(len(hypothesis)):

			if(callable(hypothesis[i])):
				hyp += 'self.' + hypothesis[i].__name__ + '('
				# temp -= 1
				# if any([i == 0 for i in temp]):
					# hyp += ')'

			else:
				if i != (len(hypothesis) - 1):

					if i != ')':
						hyp += "'" + hypothesis[i] + "',"
					else:
						hyp += hypothesis[i]
					# temp -= 1
					# if any([i == 0 for i in temp]):
						# hyp = list(hyp)
						# hyp[-1] = ')'
						# hyp += ','
						# hyp = ''.join(hyp)
				else:
					hyp += "'" + hypothesis[i] + "'"
					# temp -= 1
					# if any([i == 0 for i in temp]):
						# hyp += ')'

		# offset = hyp.count('(') - hyp.count(')')
		# for i in range(offset):
			# hyp += ')'

		hyp = hyp.replace("')'",')')
		hyp = hyp.replace(",)",")")

		return hyp

	def fullParse(self, hypothesis):
		coords = self.validCoords(hypothesis)
		parseList = list()
		for coord in coords:
			parseList.append(self.parse(hypothesis,coord))

		return parseList

	def validCoords(self,hypothesis):
		numPrim = [i for i in range(1,len(hypothesis)) if callable(hypothesis[i])]
		coords = list()
		for i in range(len(numPrim)):
			numPrim[i] = range(numPrim[i]+2, len(hypothesis))

		return list(itertools.product(*tuple(numPrim)))


			
	def hypParser(self, fullHypList):
		"""
		Converts the hypotheses generated by BFSampler to eval ready form
		"""

		finalHypList = list()

		for hypothesis in fullHypList:
			hyp = ""
			paren = ""
			for j in range(len(hypothesis)):
				if(callable(hypothesis[j])):
					hyp += 'self.' + hypothesis[j].__name__ + '('
					paren += ')'
				else:
					if j != (len(hypothesis) - 1):
						hyp += "'" + hypothesis[j] + "'," 
					else:
						hyp += "'" + hypothesis[j] + "'"

			hyp += paren
			finalHypList.append(hyp)

		objects = ["np.array([" + "'" + i + "'"  + "],dtype=object)" for i in self.objects]
		evalObjects = [eval(i) for i in objects]


		self.hypotheses = [i.replace('self.','') for i in finalHypList]
		self.evalHypotheses = [eval(i) for i in finalHypList]
		
		self.primHypotheses = list()
		for i in fullHypList:
			self.primHypotheses.append(1.0 + len([j for j in i if callable(j)]))

		for i in ["'" + j + "'" for j in self.objects]: self.hypotheses.append(i)
		for i in evalObjects: self.evalHypotheses.append(i)
		for i in evalObjects: self.primHypotheses.append(1.0)


	def BFSampler(self, depth):
		"""

		"""

		final = set()
		finalEval = set()

		objects = ["np.array([" + "'" + i + "'"  + "],dtype=object)" for i in self.objects]
		evalObjects = [eval(i) for i in objects]

		self.hypotheses = list()
		self.evalHypotheses = list()
		self.primHypotheses = list()

		for i in ["'" + j + "'" for j in self.objects]: self.hypotheses.append(i)
		for i in evalObjects: 
			self.evalHypotheses.append(i)
			self.primHypotheses.append(1.0)
			finalEval.add(np.array_str(i))

		space = self.objects + self.primitives

		# Drop crap expressions from iterator using 
		eSpace = itertools.product(space, repeat=3)
		for i in range(3,depth+1):
			eSpace = itertools.chain(eSpace, itertools.product(space, repeat=i+1))
		check = lambda x: callable(x[0]) and not callable(x[-1]) and not callable(x[-2])
		eSpace = list(itertools.ifilter(check, eSpace))

		hSpace = []
		hypSpace = []

		for e in eSpace:
			try:
				
				# Implement a set to remove duplicates
				hypList = self.fullParse(list(e))
				
				for hyp in hypList:

					hBuffer = set()
					hEvalBuffer = set()

					temp = eval(hyp)
					temp = np.sort(temp)

					hBuffer.add(hyp)
					hEvalBuffer.add(np.array_str(temp))

					if type(temp) is np.ndarray and finalEval.isdisjoint(hEvalBuffer):
						hSpace += [temp]
						hypSpace += [hyp.replace('self.','')]
						finalEval.add(np.array_str(temp))
						final.add(hyp)
						self.primHypotheses.append(1.0 + float(len(e)))
			
			except SyntaxError:
				pass

		self.hypotheses += hypSpace
		self.evalHypotheses += hSpace


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

		if type(graphList) is not np.ndarray:
			graphList = np.array([graphList],dtype='S32')

		# Attach graphs to start node
		graphList = self.linkGraphStrings(graphList)

		# Find cheapest path
		graphString = self.minCostGraphString(graphList)
		graphString = np.append(graphString,0.0)
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
	

	# Testing new stuff

	#  
	def Or(self, *args):

		if len(args) < 2:
			raise SyntaxError('Not enough arguments to function.')


		vector = np.array([],dtype=object)

		for arg in args:
			if type(arg) is tuple:
				vector = np.append(vector, np.array(arg,dtype=object))
			else:
				vector = np.append(vector, arg)

		return np.unique(vector)

	def Then(self,*args):

		if len(args) < 2:
			raise SyntaxError('Not enough arguments to function.')

 		args = np.array(args,dtype=object)
 		for i in range(len(args)):
 			if type(args[i]) is not np.ndarray:
 				args[i] = np.array([args[i]],dtype=object)

		return np.array([''.join(s) for s in list(itertools.product(*args))], dtype=object)


	def And(self, *args):

		if len(args) < 2:
			raise SyntaxError('Not enough arguments to function.')
		
 		args = np.array(args,dtype=object)
 		for i in range(len(args)):
 			if type(args[i]) is not np.ndarray:
 				args[i] = np.array([args[i]],dtype=object)

		final = np.array([])
		temp = list(itertools.permutations(args))
		for arg in temp:
			final = np.append(final, np.array([''.join(s) for s in list(itertools.product(*arg))], dtype=object))

		return final
		# return np.array([''.join(s) for s in list(itertools.permutations(final_args))], dtype=object)


	# def Then(self, A, B):
	# 	"""
	# 		Primitive function to do the 'Then' operation.
	# 		Adds every possible combination of A->B for all content
	# 		within A and B to a list. 

	# 		e.g. A:['A'], B:['A,B'] ; Then(A,B):['AA','AB']
	# 	"""

	# 	# If input is a char, turn into numpy array
	# 	if type(A) is not np.ndarray:
	# 		A = np.array([A],dtype='S32')
	# 	if type(B) is not np.ndarray:
	# 		B = np.array([B],dtype='S32')

	# 	# C will hold all combinations of A->B
	# 	C = np.array([],dtype='S32')
	# 	for i in range(len(A)):
	# 		for j in range(len(B)):
				
	# 			# if A[i][-1] == B[j][0]:
	# 			# 	C = np.append(C, A[i]+B[j][1:])

	# 			# else:
	# 			C = np.append(C, A[i] + B[j])

	# 	return C


	# def Or(self, A, B):
	# 	"""
	# 		Primitive function to do the 'Or' operation. 
	# 		Essentially throws the contents of A and B into
	# 		one list (of subgraphs). 

	# 		e.g. A:['A'], B:['A,B'] ; Or(A,B):['A','A','B']
	# 	"""

	# 	# If input is a char, turn into numpy array
	# 	if type(A) is not np.ndarray:
	# 		A = np.array([A],dtype='S32')
	# 	if type(B) is not np.ndarray:
	# 		B = np.array([B],dtype='S32')

	# 	return np.unique(np.append(A,B))
		


	# def And(self, A, B):
	# 	"""
	# 		Primitive function to do the 'And' operation.
	# 		Defined as a composition of Or and Then.

	# 		And(A,B) = Or(Then(A,B),Then(B,A))
	# 	"""

	# 	# If input is a char, turn into numpy array
	# 	if type(A) is not np.ndarray:
	# 		A = np.array([A],dtype='S32')
	# 	if type(B) is not np.ndarray:
	# 		B = np.array([B],dtype='S32')

	# 	return self.Or(self.Then(A,B), self.Then(B,A))


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

		for i in range(len(self.objects)):
			graphString = graphString.replace(self.objects[i],str(i))

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
		return self.dist[int(edgeString[0])][int(edgeString[1])]




"""

Depth 1: A, B, C
	- arg = 0

Depth 2: A, B, C, Or(A,B), And(A,B), Then(A,B)
	- arg = 2

	Or, A , B -> 3 length


Depth 3: A, B, C, Or(A,B), And(A,B,C), Or(A,B,C))
	- arg = 3


And, A, B,  C -> 4 length

Depth 4: Or

	Or(And(A,B),C)

	Or, And, A, B, C -> 5 length
	Or, A, B, C, D   -> 5 length


"""


