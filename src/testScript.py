from GridWorld import GridWorld
from Grid import Grid
from Hypothesis import Hypothesis
from InferenceMachine import InferenceMachine
import csv
import matplotlib.pyplot as plt
import collections


#################### Testing ############################

use = False
single_1 = use
single_2 = use
single_3 = use
single_4 = use
single_5 = use
single_6 = use
single_7 = use
single_8 = use
double_1 = use
double_2 = use
double_3 = use
double_4 = use
double_5 = use
double_6 = use
double_7 = use
double_8 = use
double_9 = use
double_10 = use
double_11 = use


H = Hypothesis(Grid('testGrid'))
H.BFSampler(5)
H.parseToEnglish()

oFile = open('model_results.csv','w')
CSV = csv.writer(oFile)
CSV.writerow(['Trial','Rational Action', 'Rational Choice', 'Hypothesis Rank',
				'Hypothesis','Posterior'])
oFile2 = open('parsed_english.csv','w')
CSV2 = csv.writer(oFile2)

stimCounter = 1

lesion = ['RA/RC Lesion', 'RA Lesion','RC Lesion','Full Model']

plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):

	for j in range(2):

		if single_1:

			print '\n--------------------------------------------------------'
			print 'Test single_1:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j
			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12]
			actions = [[0,0,0,'take','stop']]
			
			infer = InferenceMachine(3, [testGrid], start, actions,
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


			

if single_1:

	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)


stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):

		if single_2:

			print '\n--------------------------------------------------------'
			print 'Test single_2:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j
			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12]
			actions = [[0,3,0,3,0,3,'take','stop']]
			
			infer = InferenceMachine(3, [testGrid], start, actions,
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if single_2:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)


stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):

		if single_3:

			print '\n--------------------------------------------------------'
			print 'Test single_3:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j
			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12]
			actions = [[0,0,0,'take',3,3,3,'take','stop']]

			infer = InferenceMachine(3, [testGrid], start, actions,
				rationalAction=i,rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if single_3:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)




stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):

		if single_4:

			print '\n--------------------------------------------------------'
			print 'Test single_4:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j
			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12]
			actions = [[0,3,0,3,0,3,'take',2,2,2,'take','stop']]

			infer = InferenceMachine(3, [testGrid], start, actions,
				rationalAction=i,rationalChoice=j, hypotheses=H)
			
			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if single_4:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)




stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):

		if single_5:

			print '\n--------------------------------------------------------'
			print 'Test single_5:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j
			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12]
			actions = [[0,0,0,'take',3,1,3,1,3,1,'take','stop']]

			infer = InferenceMachine(3, [testGrid], start, actions,
				rationalAction=i,rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if single_5:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)



stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):

		if single_6:

			print '\n--------------------------------------------------------'
			print 'Test single_6:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j
			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12]
			# actions = [[0,0,'take','take','take',3]]
			actions = [[0,0,0,'take',3,3,3,'take',1,1,1,'take','stop']]

			infer = InferenceMachine(4, [testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if single_6:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)


stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):

		if single_7:

			print '\n--------------------------------------------------------'
			print 'Test single_7:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j
			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12]
			# actions = [[0,0,'take','take','take',3]]
			actions = [[0,0,0,'take',3,1,3,1,3,1,'take',0,0,0,'take','stop']]

			infer = InferenceMachine(4, [testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if single_7:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)



stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):

		if single_8:

			print '\n--------------------------------------------------------'
			print 'Test single_8:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j
			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12]
			# actions = [[0,0,'take','take','take',3]]
			actions = [[0,3,0,3,0,3,'take',2,2,2,'take',1,3,1,3,1,3,'take','stop']]

			infer = InferenceMachine(4, [testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if single_8:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)



stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):
		
		if double_1:

			print '\n--------------------------------------------------------'
			print 'Test double_1:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j
			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			actions = [[3,3,3,'take','stop'],[0,0,0,'take','stop']]

			infer = InferenceMachine(3, [testGrid,testGrid], start, actions,
				rationalAction=i,rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if double_1:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)




stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):
		if double_2:

			print '\n--------------------------------------------------------'
			print 'Test double_2:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j

			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			# actions = [[0,0,'take','take','take',3]]
			actions = [[0,0,0,'take',3,3,3,'take','stop'],[3,3,3,'take',0,0,0,'take','stop']]

			infer = InferenceMachine(5, [testGrid,testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if double_2:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)




stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):
		if double_3:

			print '\n--------------------------------------------------------'
			print 'Test double_3:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j

			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			# actions = [[0,0,'take','take','take',3]]
			actions = [[0,3,0,3,0,3,'take',2,2,2,'take','stop'],[3,3,3,'take',0,2,0,2,0,2,'take','stop']]

			infer = InferenceMachine(5, [testGrid,testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if double_3:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)



stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):

		if double_4:

			print '\n--------------------------------------------------------'
			print 'Test double_4:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j
			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			actions = [[0,0,0,'take',3,1,3,1,3,1,'take','stop'],[3,3,3,'take',0,2,0,2,0,2,'take','stop']]

			infer = InferenceMachine(4, [testGrid,testGrid], start, actions,
				rationalAction=i,rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if double_4:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)



stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):
		if double_5:

			print '\n--------------------------------------------------------'
			print 'Test double_5:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j

			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			actions = [[0,0,0,'take','stop'],[0,0,0,'take','stop']]

			infer = InferenceMachine(5, [testGrid,testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if double_5:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)



stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):
		if double_6:

			print '\n--------------------------------------------------------'
			print 'Test double_6:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j

			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			actions = [[0,0,0,'take',3,3,3,'take','stop'],[0,0,0,'take',3,3,3,'take','stop']]

			infer = InferenceMachine(5, [testGrid,testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if double_6:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)




stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):
		if double_7:

			print '\n--------------------------------------------------------'
			print 'Test double_7:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j

			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			actions = [[0,3,0,3,0,3,'take',2,2,2,'take','stop'],[0,3,0,3,0,3,'take',2,2,2,'take','stop']]

			infer = InferenceMachine(5, [testGrid,testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if double_7:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)



stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):
		if double_8:

			print '\n--------------------------------------------------------'
			print 'Test double_8:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j

			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			actions = [[0,0,0,'take',3,1,3,1,3,1,'take','stop'],[0,0,0,'take',3,1,3,1,3,1,'take','stop']]

			infer = InferenceMachine(5, [testGrid,testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if double_8:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)


stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):
		if double_9:

			print '\n--------------------------------------------------------'
			print 'Test double_9:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j

			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			actions = [[0,3,0,3,0,3,'take',2,2,2,'take','stop'],[0,0,0,'take',3,1,3,1,3,1,'take','stop']]

			infer = InferenceMachine(5, [testGrid,testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if double_9:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)



stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
for i in range(2):
	for j in range(2):
		if double_10:

			print '\n--------------------------------------------------------'
			print 'Test double_10:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j

			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			actions = [[0,3,0,3,0,3,'take',2,2,2,'take','stop'],[0,3,0,3,0,3,'take',1,1,1,'take','stop']]

			infer = InferenceMachine(5, [testGrid,testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)


if double_10:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)



stimCounter += 1
plt.rcParams.update({'font.size': 100})
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'
fig = plt.figure(figsize=(100,100))
fig.suptitle('Experiment '+str(stimCounter),fontsize=300)
allMaxHyp = set()
allData = list()
infers = list()
for i in range(2):
	for j in range(2):
		if double_11:

			print '\n--------------------------------------------------------'
			print 'Test double_11:'
			print 'Rational Action: ',i
			print 'Rational Choice: ',j

			testGrid = Grid('testGrid')
			testGrid2 = Grid('testGrid2')
			start = [12,12]
			actions = [[0,0,0,'take',3,1,3,1,3,1,'take','stop'],[0,3,0,3,0,3,'take',1,1,1,'take','stop']]

			infer = InferenceMachine(5, [testGrid,testGrid], start, actions, 
				rationalAction=i, rationalChoice=j, hypotheses=H)

			allMaxHyp = allMaxHyp.union(set(infer.maxHyp[0:6]))
			allData.append(infer.hypPosterior)

			infers.append(infer)


if double_11:
	allMaxHyp = list(allMaxHyp)
	results = dict()
	for h in allMaxHyp:
		results.update({h:infer.hypPosterior[h]})

	hypList = sorted(results,key=results.get,reverse=True)

	results = list()
	for data in allData:

		temp = list()
		for h in hypList:
			temp.append(data[h])

		results.append(temp)

	for i in range(len(results)):
		fig.subplots_adjust(bottom=.2)
		ax = fig.add_subplot(2,2,i+1)
		ax.spines['bottom'].set_linewidth(10)
		ax.spines['left'].set_linewidth(10)
		ax.spines['right'].set_linewidth(0)
		ax.spines['top'].set_linewidth(0)


		ax.set_title(lesion[i],fontweight='bold')
		width=0.8
		bins = map(lambda x: x-width/2,range(1,len(results[i])+1))

		ax.bar(bins,results[i],width=width)
		ax.set_xticks(map(lambda x: x, range(1,len(results[i])+1)))
		ax.set_xticklabels(hypList,rotation=45, rotation_mode="anchor", ha="right")


	fig.subplots_adjust(hspace=.8)
	plt.savefig('charts/experiment'+str(stimCounter), dpi=fig.dpi)
	plt.close('all')

	if len(results) > len(infer.hypotheses):
		temp = len(infer.hypotheses)
	else:
		temp = len(results[0])
	for i in range(2):
		for j in range(2):
			for k in range(temp):
				if k < 3:
					CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
						H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])
				# elif k==temp-1:
				# 	CSV.writerow([stimCounter, True if i is 1 else False, True if j is 1 else False, k+1, 
				# 		H.englishHypotheses[infer.hypotheses.index(hypList[k])],results[i*2+j*1][k]])



	englishReference = [infer.hypotheses.index(i) for i in hypList]
	englishHypotheses = [H.englishHypotheses[i] for i in englishReference]
	# temp = englishHypotheses[-1]
	# temp = '"' + temp + '"' 
	englishHypotheses = ['"' + i + '"' for i in englishHypotheses[0:3]]
	# englishHypotheses.append(temp)
	englishHypotheses = [','.join(englishHypotheses)]
	CSV2.writerow([stimCounter]+englishHypotheses)



oFile.close()
oFile2.close()

