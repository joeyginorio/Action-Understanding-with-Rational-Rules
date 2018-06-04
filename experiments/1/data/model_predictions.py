import sys
sys.path.append('../../..')
from model_src.grid_world import GridWorld
from model_src.grid import Grid
from model_src.hypothesis import Hypothesis
from model_src.inference_machine import InferenceMachine
import numpy as np
import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import collections
import sys

#################### Testing ############################

use = True
trials = [use for i in range(19)]

# trials[12] = True
trialOn = 0

if len(sys.argv) > 1:
	trialOn = int(sys.argv[1])

	trials[trialOn-1] = True
	ra = [0,1]
	rc = [0,1]
else:
	sys.argv.append('-1')
	sys.argv.append('-1')
	sys.argv.append('-1')
	ra = [0,1]
	rc = [0,1]

actions = [
[[0,0,0,'take','stop']],
[[6,6,6,'take','stop']],
[[0,0,0,'take',3,3,3,'take','stop']],
[[6,6,6,'take',2,2,2,'take','stop']],
[[0,0,0,'take',8,8,8,'take','stop']],
[[0,0,0,'take',3,3,3,'take',1,1,1,'take','stop']],
[[0,0,0,'take',8,8,8,'take',0,0,0,'take','stop']],
[[6,6,6,'take',2,2,2,'take',8,8,8,'take','stop']],
[[3,3,3,'take','stop'],[0,0,0,'take','stop']],
[[0,0,0,'take',3,3,3,'take','stop'],[3,3,3,'take',0,0,0,'take','stop']],
[[6,6,6,'take',2,2,2,'take','stop'],[3,3,3,'take',5,5,5,'take','stop']],
[[0,0,0,'take',8,8,8,'take','stop'],[3,3,3,'take',5,5,5,'take','stop']],
[[0,0,0,'take','stop'],[0,0,0,'take','stop']],
[[0,0,0,'take',3,3,3,'take','stop'],[0,0,0,'take',3,3,3,'take','stop']],
[[6,6,6,'take',2,2,2,'take','stop'],[6,6,6,'take',2,2,2,'take','stop']],
[[0,0,0,'take',8,8,8,'take','stop'],[0,0,0,'take',8,8,8,'take','stop']],
[[6,6,6,'take',2,2,2,'take','stop'],[0,0,0,'take',8,8,8,'take','stop']],
[[6,6,6,'take',2,2,2,'take','stop'],[6,6,6,'take',1,1,1,'take','stop']],
[[0,0,0,'take',8,8,8,'take','stop'],[6,6,6,'take',1,1,1,'take','stop']]
]
	
chek = [['A', 'A or B', 'A or C'], 
['B', 'B or (A and B)', 'B or (B and C)'], 
['A then B', 'A and B', 'A then (B or C)'], 
['B then A', 'A and B', '(B or C) then A'], 
['A then C', 'A and C', '(A or B) then C'], 
['A then B then C', 'A and B and C', '(A and B) then C'], 
['A then C then B', 'B and (A then C)', '(A and C) then B'], 
['B then A then C', 'B then (A and C)', 'C and (B then A)'], 
['A or C', 'A or B or C', 'A or C or (C then B)'], 
['(A or C) then B', 'B and (A or C)', '(A and B) or (C then B)'], 
['(B or C) then A', 'A and (B or C)', '(A and C) or (B then A)'], 
['A and C', 'A and (B or C)', 'C and (A or B)'], 
['A', 'A or B', 'A or (B and C)'], 
['A then B', 'A and B', 'A then (B or C)'], 
['B then A', 'A and B', '(B or C) then A'], 
['A then C', 'A and C', '(A or B) then C'], 
['(A then C) or (B then A)', '(A and C) or (B then A)', 'A and (B or C)'], 
['B then (A or C)', 'B and (A or C)', '(A and B) or (B then C)'], 
['(A or B) then C', 'C and (A or B)', '(A and C) or (B then C)']]

testGrid = Grid('grid_worlds/testGrid.txt',True)
grid1 = [[testGrid] for i in range(8)]
grid2 = [[testGrid,testGrid] for i in range(8,19)]
grid = grid1 + grid2
start1 = [[12] for i in range(8)]
start2 = [[12,12] for i in range(8,19)]
start = start1 + start2

#############################################################


def getDepthPosterior(posterior, primHypotheses, depth):


	currentPosterior = posterior[0]
	for i in range(1,len(posterior)):
		currentPosterior *= posterior[1]
	posterior = currentPosterior



	total = list()
	
	for i in range(2,depth+1):
		temp = 0
		for j in range(len(posterior)):
			if primHypotheses[j] < i+2:
				temp += posterior[j]
			else:
				break

		total.append(temp)

	return total


H = Hypothesis(Grid('grid_worlds/testGrid.txt', True))
depth = 6
# H.sampleHypotheses(5000)

H.BFSampler(depth)
H.parseToEnglish()

# H.flattenAll()

oFile = open('model_results'+sys.argv[1]+'_'+'.csv','w')
CSV = csv.writer(oFile)
# CSV.writerow(['Trial','Rational Action', 'Rational Choice', 'Hypothesis Rank',# 'Hypothesis','Posterior'])
# oFile2 = open('parsed_english.csv','w')
# CSV2 = csv.writer(oFile2)

# oFile3 = open('marginalLikelihood'+'Trial'+sys.argv[1]+'.csv','w')
# CSV3 = csv.writer(oFile3)
# CSV3.writerow(['Trial','Model','Depth2','Depth3','Depth4','Depth5','Depth6'])

stimCounter = 1

lesion = ['RA/RC Lesion', 'RA Lesion','RC Lesion','Full Model']

cfull = list()
infers = list()

for trial in range(len(trials)):

	if trials[trial]:
		print '\n_______________'
		print 'TRIAL ',trial
		print '_______________'
	# plt.rcParams.update({'font.size': 100})
	# plt.rcParams['xtick.major.pad']='15'
	# plt.rcParams['ytick.major.pad']='15'
	# fig = plt.figure(figsize=(100,100))
	# fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
	allMaxHyp = set()
	allData = list()
	allEval = list()
	ins = list()
	for i in ra:

		for j in rc:

			if i!=j:
				continue

			if trials[trial]:

				print '\n--------------------------------------------------------'
				# print 'Test single_1:'
				print 'Rational Action: ',i
				print 'Rational Choice: ',j
				print '\n'
				
				infer = InferenceMachine(3, grid[trial], start[trial], actions[trial], tauChoice=.01,
					rationalAction=i, rationalChoice=j, hypotheses=H,MCMCOn=False,trials=trialOn,reward=30)

				c = list()
				# infers.append(infer)

				for k in range(3):

					ind = H.englishHypotheses.index(chek[trial][k])
					# ind = H.hypotheses.index(infer.maxHyp[k])

					# if i == 1 and j == 1:
					# 	a = H.hypotheses[ind].count('Then')
					# 	b = H.hypotheses[ind].count('And')
					# 	c = H.hypotheses[ind].count('Or')
					# 	args = 0
					# 	args += H.hypotheses[ind].count("'C'")
					# 	args += H.hypotheses[ind].count("'B'")
					# 	args += H.hypotheses[ind].count("'A'")

					# 	CSV.writerow([trial+1,'Full Model',k+1,H.englishHypotheses[ind],infer.posteriors[-1][ind]/infer.prior[ind],a,b,c,args,infer.primHypotheses.count(infer.primHypotheses[ind])])
					# 	# c.append(H.englishHypotheses[ind])

					# if i == 1 and j == 1:
					# 	CSV.writerow([trial+1,'Prior Model',k+1,H.englishHypotheses[ind],H.hypotheses[ind],infer.prior[ind]])
					# 	c.append(H.englishHypotheses[ind])

					if i == 1 and j == 1:
						CSV.writerow([trial+1,'Full Model',k+1,H.englishHypotheses[ind],H.hypotheses[ind],infer.posteriors[-1][ind]])
						c.append(H.englishHypotheses[ind])

					elif i == 0 and j == 0:
						CSV.writerow([trial+1,'Alternate Model',k+1,H.englishHypotheses[ind],H.hypotheses[ind],infer.posteriors[-1][ind]])

				ins.append(infer)

				if i ==1 and j==1:
					cfull.append(c)

	infers.append(ins)    

				# allMaxHyp = set(infer.maxHypMCMC[0:3])`
				# allData.append(infer.hypPosteriorMCMC)
				# allEval.append(infer.evalHypMCMC)

				# CSV


				# OLD 
				# unblock below this line
				# print '\n--------------------------------------------------------'
				# # print 'Test single_1:'
				# print 'Rational Action: ',i
				# print 'Rational Choice: ',j
				# print '\n'
				
				# infer = InferenceMachine(3, grid[trial], start[trial], actions[trial], tauChoice=.01,
				# 	rationalAction=i, rationalChoice=j, hypotheses=H,MCMCOn=False,trials=trialOn)

				# for k in range(3):

				# 	ind = infer.hypotheses.index(chek[trial][k])

				# 	if i == 1 and j == 1:
				# 		CSV.writerow([trial+1,'Full Model',k+1,H.englishHypotheses[ind],infer.posteriors[-1][ind]])
				# 	elif i == 1 and j == 0:
				# 		CSV.writerow([trial+1,'Alternate Model',k+1,H.englishHypotheses[ind],infer.posteriors[-1][ind]])

				# # allMaxHyp = set(infer.maxHypMCMC[0:3])`
				# # allData.append(infer.hypPosteriorMCMC)
				# # allEval.append(infer.evalHypMCMC)

				# # CSV3.writerow([stimCounter,lesion[i*2+1*j]] + getDepthPosterior(infer.posteriors, infer.primHypotheses, depth))





	# allMaxHyp = chek[trial]
	# allResults = list()
	# if trials[trial]:

	# 	for i in range(len(allData)):
	# 		# allMaxHyp = list(allMaxHyp)
	# 		results = dict()
	# 		for h in allMaxHyp:
	# 			if h in allData[i].keys():
	# 				results.update({h:allData[i][h]})
	# 			else:
	# 				ht = h
	# 				ht = ht.replace('And','H.And')
	# 				ht = ht.replace('Or','H.Or')
	# 				ht = ht.replace('Then','H.Then')
	# 				evalH = eval(ht)
	# 				check = [np.array_equal(evalH,c) for c in allEval[i]]
	# 				if any(check):
	# 					ind = check.index(True)
	# 					# print ind
	# 					results.update( {h: allData[i][allData[i].keys()[ind]] } )
	# 				else:
	# 					results.update({h:0.0})



	# 		allResults.append(results)
					# print 'here'
					# checkH = h

					# if 'H.' not in checkH:
					# 	checkH = checkH.replace('Or','H.Or')
					# 	checkH = checkH.replace('And','H.And')
					# 	checkH = checkH.replace('Then','H.Then')

					# check = [np.array_equal(eval(checkH), m) for m in infer.evalHypMCMC]

					# if not any(check):
					# 	results.update({h:0.0})
					# else:
					# 	hIndex = check.index(True)
					# 	results.update({h:infer.hypPosteriorMCMC[infer.hypMCMC[hIndex]]})

		# raise Exception

		# hypList = sorted(results,key=results.get,reverse=True)

		# results = list()
		# for data in allData:

		# 	temp = list()
		# 	for h in hypList:
		# 		try:
		# 			temp.append(data[h])
		# 		except:
		# 			temp.append(0.0)

		# 	results.append(temp)

		# for i in range(len(results)):
		# 	fig.subplots_adjust(bottom=.2)
		# 	ax = fig.add_subplot(2,2,i+1)
		# 	ax.spines['bottom'].set_linewidth(10)
		# 	ax.spines['left'].set_linewidth(10)
		# 	ax.spines['right'].set_linewidth(0)
		# 	ax.spines['top'].set_linewidth(0)


		# 	ax.set_title(lesion[i],fontweight='bold')
		# 	width=0.8
		# 	bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		# 	ax.bar(bins,results[i],width=width)
		# 	ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		# 	ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


		# fig.subplots_adjust(hspace=.8)
		# plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
		# plt.close('all')

		# if len(results) > len(infer.hypotheses):
		# 	temp = len(infer.hypotheses)
		# else:
		# 	temp = len(results[0])

		# print 'allMAXHYP'
		# print allMaxHyp
		# englishMaxHyp = list()
		# for h in allMaxHyp:
		# 	h = h.replace('Then','H.T')
		# 	h = h.replace('And','H.A')
		# 	h = h.replace('Or','H.O')

		# 	# print h
		# 	hEval = eval(h)
		# 	if hEval[0] == '(':
		# 		hEval = hEval[1:len(hEval)-1]
		# 	englishMaxHyp.append(hEval)

		# # print englishMaxHyp
		# for i in ra:
		# 	for j in rc:
		# 		for k in range(len(allMaxHyp)):
		# 			if k < 3:	
		# 				# print englishMaxHyp
		# 				CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, 
		# 					englishMaxHyp[k],allResults[0][allMaxHyp[k]]])


		# allResultsCopy = allResults
		# allDataCopy = allData
		# allMaxHypCopy = allMaxHyp
						# except:
						# 	aMaxHyp = allMaxHyp[k]
						# 	if 'H.' not in aMaxHyp:
						# 		aMaxHyp = aMaxHyp.replace('Or','H.Or')
						# 		aMaxHyp = aMaxHyp.replace('And','H.And')
						# 		aMaxHyp = aMaxHyp.replace('Then','H.Then')

						# 	check = [np.array_equal(eval(aMaxHyp), m) for m in infer.evalHypMCMC]
						# 	if not any(check):
						# 		h = aMaxHyp

						# 		evalH = eval(h)
						# 		if evalH[0] == '(':
						# 			evalH = evalH[1:len(evalH)-1]

						# 		print 'Here'
						# 		print evalH
						# 		CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						# 			evalH,0.0])
						# 	else:
						# 		index = check.index(True)
						# 		h = aMaxHyp
						# 		h = h.replace('Or','O')
						# 		h = h.replace('And','A')
						# 		h = h.replace('Then','T')
								
						# 		evalH = eval(h)
						# 		if evalH[0] == '(':
						# 			evalH = evalH[1:len(evalH)-1]

						# 		# print infer.hypotheses[index]
						# 		# print allData[i*2+1*j]
						# 		try:
						# 			CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						# 			evalH, allData[i*2+1*j][infer.hypMCMC[index]]])
						# 		except KeyError:
						# 			print 'HEREE'
						# 			CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						# 			evalH, 0.0])									

					# elif k==temp-1:
					# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
					# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



		# englishReference = [infer.hypotheses.index(i) for i in hypList]
		# englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
		# # temp = englishHypotheses[-1]
		# # temp = '"' + temp + '"' 
		# englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
		# # englishHypotheses.append(temp)
		# englishHypotheses = [','.join(englishHypotheses)]
		# CSV2.writerow([stimCounter]+englishHypotheses)

	stimCounter += 1


oFile.close()
# oFile2.close()
# oFile3.close()



"""
Parents turned around, any effecct?
Was it mentioned why parents peeking is a problem?





To represent a new concept as a new, unseen kind, must be able to compare 


"""