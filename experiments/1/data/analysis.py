import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('ggplot')
from pandas import Series, DataFrame
from scipy.stats import pearsonr
from matplotlib import rcParams
from matplotlib.legend_handler import HandlerLine2D
import scikits.bootstrap as bootstrap
import numpy as np
import pandas as pd
import seaborn as sns
import math
from copy import deepcopy

""" Model Results Analysis """

# Read in the data
modelResults = pd.read_csv('model_results.csv')
fullModel = modelResults.loc[modelResults.Model=='Full Model',:]
# lesionModel = modelResults.loc[modelResults.Model=='Alternate Model',:]
lesionModel = pd.read_csv('priormodel_results.csv')

# Do stuff per Trial 
for i in range(1,20):
	# Normalize
	temp = fullModel.loc[fullModel.Trial==i,'Result']
	fullModel.loc[fullModel.Trial==i,'Result'] = temp / temp.sum()
	# # Z-Score per trial
	# temp = fullModel.loc[fullModel.Trial==i,'Result']
	# fullModel.loc[fullModel.Trial==i,'Result'] = (temp - temp.mean()) / temp.std()

for i in range(1,20):
	# Normalize
	temp = lesionModel.loc[lesionModel.Trial==i,'Result']
	lesionModel.loc[lesionModel.Trial==i,'Result'] = temp / temp.sum()
	# Z-Score per trial
	# temp = lesionModel.loc[lesionModel.Trial==i,'Result']
	# lesionModel.loc[lesionModel.Trial==i,'Result'] = (temp - temp.mean()) / temp.std()


""" Turk Results Analysis """

# Load in turk data
turkResults = pd.read_csv('mturk_results.csv')
turkResults.drop('Image',1,inplace=True)
turkResults.drop('Key',1,inplace=True)

# Separate per participant
allIDs = turkResults.Id.unique()
participants = list()
for ID in allIDs:
	# Get participants data
	participants.append(turkResults.loc[turkResults.Id==ID,:])

	# Sort by trial, rename to combine columns
	participants[-1] = participants[-1].rename(columns={'H1':'H', 'H1Data':'Result'})

	# Rename to combine columns
	temp2 = participants[-1].loc[:,['Id','Trial','H2','H2Data']]
	temp2 = temp2.rename(columns={'H2':'H', 'H2Data':'Result'})
	temp3 = participants[-1].loc[:,['Id','Trial','H3','H3Data']]
	temp3 = temp3.rename(columns={'H3':'H', 'H3Data':'Result'})

	# Edit original columns, combine columns
	participants[-1].drop(['H2','H2Data','H3','H3Data'],1,inplace=True)
	participants[-1] = participants[-1].append([temp2,temp3])
	participants[-1] = participants[-1].sort_values(['Trial','H'])

	# ZScore per Trial
	for i in range(1,20):
		temp = participants[-1].loc[participants[-1].Trial==i,'Result']
		if math.isnan((temp/temp.sum()).values[0]):
			participants[-1].loc[participants[-1].Trial==i, 'Result'] = 0.0

		else:
			participants[-1].loc[participants[-1].Trial==i, 'Result'] = temp / temp.sum()


		# # Some people answered same for all, so no std dev (zscore breaks)
		# if temp.std() != 0.0:
		# 	participants[-1].loc[participants[-1].Trial==i, 'Result'] = (temp - temp.mean())/temp.std()
		# else:
		# 	temp[:] = 0.0
		# 	participants[-1].loc[participants[-1].Trial==i, 'Result'] = temp
	
	# Z-Score per participant
	temp = participants[-1].loc[participants[-1].Trial>0,'Result']
	participants[-1].loc[participants[-1].Trial>0,'Result'] = (temp - temp.mean())/temp.std()


# Average over all the participants
temp = participants[0].copy()
for i in range(1,len(participants)):
	temp.loc[temp.Trial > 0, 'Result'] += participants[i].loc[participants[i].Trial>0,'Result'].values

# Take average
temp.Result /= len(participants)

# Sort turk data so it matches order of model
categories = list()
for i in range(1,20):
	categories.append(fullModel.loc[fullModel.Trial==i,'Hypothesis'].values)

for i in range(1,20):
	repl = temp.loc[temp.Trial==i,:]
	repl.H = pd.Categorical(repl.H,categories[i-1])
	repl.sort_values('H',inplace=True)
	temp.loc[temp.Trial==i,:] = repl

turkResults = temp

# for i in range(1,20):
# 	temp = turkResults.loc[turkResults.Trial==i,'Result']
# 	turkResults.loc[turkResults.Trial==i,'Result'] = (temp - temp.min())/temp.ptp()
# for i in range(1,20):
# 	temp = fullModel.loc[fullModel.Trial==i,'Result']
# 	fullModel.loc[fullModel.Trial==i,'Result'] = (temp - temp.min())/temp.ptp()

temp = fullModel.loc[fullModel.Trial>0,'Result']
fullModel.loc[fullModel.Trial>0,'Result'] = (temp - temp.mean())/temp.std()

temp = lesionModel.loc[lesionModel.Trial>0,'Result']
lesionModel.loc[lesionModel.Trial>0,'Result'] = (temp - temp.mean())/temp.std()

# Add all the data to one dataframe to check correlation!
turkResults.reset_index(drop=True,inplace=True)
fullModel.reset_index(drop=True,inplace=True)
lesionModel.reset_index(drop=True,inplace=True)
# priorModel.reset_index(drop=True,inplace=True)

###########
def bootstrap_sample(values):
	idx = np.random.choice(len(values), size=len(values), replace=True)
	sample = [values[i] for i in idx]
	return sample

def bootstrap_95CI(values):
	corrs = list()
	for i in xrange(10000):
		sample = bootstrap_sample(values)
		temp = zip(*sample)
		corrs.append(pearsonr(temp[0],temp[1])[0])

	corrs = sorted(corrs)
	return (corrs[499],corrs[9499])

def bootstrap_95CI_Diff(values1,values2):
	corrs = list()
	for i in xrange(10000):
		sample1 = bootstrap_sample(values1)
		sample2= bootstrap_sample(values2)

		temp1 = zip(*sample1)
		temp2 = zip(*sample2)
		corrs.append(pearsonr(temp1[0],temp1[1])[0]-pearsonr(temp2[0],temp2[1])[0])

	corrs = sorted(corrs)
	return (corrs[499],corrs[9499])

def bootstrap_mean(values):
	corrs = list()
	for i in xrange(10000):
		sample = bootstrap_sample(values)
		corrs.append(np.mean(sample))

	corrs = sorted(corrs)
	return (corrs[499],corrs[9499])

############




""" Plot of correlations / CIs """
# turkResults = turkResults[turkResults.Trial != 11]
# fullModel = fullModel[fullModel.Trial != 11]
# lesionModel = lesionModel[lesionModel.Trial != 11]


a = list()
b = list()
full = fullModel.Result.values
lesion = lesionModel.Result.values
turk = turkResults.Result.values


for i in range(1,20):
	# if i == 11:
	# 	continue

	a.append(turkResults.loc[turkResults.Trial==i,'Result'].corr(fullModel.loc[fullModel.Trial==i].Result))
	b.append(turkResults.loc[turkResults.Trial==i,'Result'].corr(lesionModel.loc[lesionModel.Trial==i].Result))

a = [i for i in a if not np.isnan(i)]
b = [i for i in b if not np.isnan(i)]

fig2 = plt.figure(1,figsize=(12,6))
gs2 = gridspec.GridSpec(1,2)
gs2.update(wspace=.08,hspace=.6)

# plt.suptitle("Model: Normalized + Z-Score across Trials , Turk: Normalized + Z-Score across Trials")
fig2.add_subplot(gs2[0])
ax_1 = fig2.add_subplot(gs2[0])
ax_1.set_axis_bgcolor('white')
for spine in ['left','right','top','bottom']:
	ax_1.spines[spine].set_color('k')

ax_1.get_yaxis().set_tick_params(width=1)
ax_1.get_xaxis().set_tick_params(width=1)

ax_1.set_axisbelow(True)
ax_1.grid(linestyle='-',color='black',linewidth=.1)

ax_1.set_title('Our model',fontsize=14)

# plt.subplot(311)
x = turkResults.loc[turkResults.Trial>0,'Result'].values
y = fullModel.loc[fullModel.Trial>0,'Result'].values
corr1 = round(turkResults.loc[turkResults.Trial>0,'Result'].corr(fullModel.loc[fullModel.Trial>0].Result,method='pearson'),2)
CI1 = bootstrap_95CI(zip(full,turk))
CI_diff = bootstrap_95CI_Diff(zip(full,turk),zip(lesion,turk))

plt.grid(linewidth=.15, color='gray')
plt.ylim([-2.0,2.0])
plt.xlim([-1.5,2.5])
g = sns.regplot(y,x,color='blue',scatter_kws=dict(color='black'),ax=ax_1)
# plt.scatter(y,x,color='black')
# fit = np.polyfit(y,x,deg=1)
# ax_1.plot(y,fit[0]*y+fit[1],color='blue')

# plt.title('Full vs Turk: r='+str(corr)+', CI='+str(CI[0])+'-'+str(CI[1]))

# plt.subplot(312)
# y = lesionModel.loc[lesionModel.Trial>0,'Result'].values
# corr = round(turkResults.loc[turkResults.Trial>0,'Result'].corr(lesionModel.loc[fullModel.Trial>0].Result,method='spearman'),2)
# CI = [round(i,2) for i in bootstrap.ci(b)]
# plt.scatter(x,y)
# plt.title('Lesion vs Turk: r='+str(corr)+', CI='+str(CI[0])+'-'+str(CI[1]))

# plt.subplot(313)
ax_2 = fig2.add_subplot(gs2[1])
ax_2.set_axis_bgcolor('white')
for spine in ['left','right','top','bottom']:
	ax_2.spines[spine].set_color('k')


# ax_1.get_yaxis().set_tick_params(width=1)
ax_2.set_yticklabels([])
ax_2.get_xaxis().set_tick_params(width=1)

ax_2.set_axisbelow(True)
# ax_2.grid(linestyle='-',color='black',linewidth=.1)

ax_2.set_title('Alternative model',fontsize=14)

y = lesionModel.loc[lesionModel.Trial>0,'Result'].values
corr2 = round(turkResults.loc[turkResults.Trial>0,'Result'].corr(lesionModel.loc[lesionModel.Trial>0].Result,method='pearson'),2)
# plt.title('Prior vs Turk: r='+str(corr)+', CI='+str(CI[0])+'-'+str(CI[1]))

# plt.scatter(y,x,color='black')
# fit = np.polyfit(y,x,deg=1)
# ax_2.plot(y,fit[0]*y+fit[1],color='blue')
CI2 = bootstrap_95CI(zip(lesion,turk))
plt.grid(linewidth=.15,color='gray')
sns.regplot(y,x,color='blue',scatter_kws=dict(color='black'),ax=ax_2)
plt.ylim([-2.0,2.0])


fig2.subplots_adjust(bottom=.2)
fig2.text(0.04,0.7,'Participant judgments',rotation='90',fontsize=16)
fig2.text(0.45,.06,'Model predictions',fontsize=16)
fig2.text(.35,.25,'r='+str(corr1),fontsize=14)
fig2.text(.35,.22,'95% CI: ('+str(round(CI1[0],2))+'-'+str(round(CI1[1],2))+')',fontsize=14)
fig2.text(.75,.22,'95% CI: ('+str(round(CI2[0],2))+'-'+str(round(CI2[1],2))+')',fontsize=14)
fig2.text(.75,.25,'r='+str(corr2),fontsize=14)


# plt.show()d
plt.savefig('models.pdf',dpi=300)
plt.close()


""" Plotting the 19 Trials Full vs Turk """

# for i in range(len(participants)):
# 	participants[i].reset_index(drop=True,inplace=True)
# allCI = list()
# for i in range(len(participants[0].Result.values)):
# 	allCI.append([])
# CI = list()
# for i in range(len(participants[0].Result.values)):
# 	for j in range(len(participants)):
# 		allCI[i].append(participants[j].loc[i,'Result'])
# finalCIs = list()
# for i in range(len(allCI)):
# 	finalCIs.append(bootstrap.ci(allCI[i]))
# buff = list()
# lastCIs = list()
# for i in range(len(finalCIs)):
# 	buff.append(finalCIs[i])
# 	if (i+1) % 3 == 0:
# 		lastCIs.append([list(buff[0]),list(buff[1]),list(buff[2])])
# 		del buff[:]
# lowCIs = list()
# highCIs = list()
# for i in range(19):
# 	buffLo = list()
# 	buffHi = list()
# 	for j in range(3):
# 		buffLo.append(lastCIs[i][j][0])
# 		buffHi.append(lastCIs[i][j][1])
# 	lowCIs.append([buffLo[0],buffLo[1],buffLo[2]])
# 	highCIs.append([buffHi[0],buffHi[1],buffHi[2]])
# 	del buffLo[:]
# 	del buffHi[:]

full = list()
fullNames = list()
turk = list()
turkNames = list()
corrs = list()
corrs_prior = list()
corrs2 = list()
lesion = list()
prior=  list()
for i in range(1,20):
	full.append(fullModel[fullModel.Trial==i].Result.values)
	fullNames.append(fullModel[fullModel.Trial==i].Hypothesis.values)
	turk.append(turkResults[turkResults.Trial==i].Result.values)
	turkNames.append(turkResults[turkResults.Trial==i].H.values)
	corrs.append(pearsonr(full[i-1],turk[i-1])[0])
	lesion.append(lesionModel[lesionModel.Trial==i].Result.values)
	corrs2.append(pearsonr(lesion[i-1],turk[i-1])[0])

fig = plt.figure(2,figsize=(20,16),facecolor='white')
fig.text(0.08,0.5,'Judgment',rotation='90',fontsize=15)
fig.text(0.5,.03,'Hypothesis',fontsize=15)
gs1 = gridspec.GridSpec(4,5)
gs1.update(wspace=.08,hspace=.4)
# plt.suptitle('Model vs Turk, per Trial',fontsize=15)
for i in range(19):
	print 'here'
	# plt.subplot(5,4,i+1)
	# plt.scatter(turk[i],full[i])
	# plt.title('Trial '+str(i+1)+': r=' + str(corrs[i]))
	# # plt.xlabel('model results')
	# # plt.ylabel('turk results')g
	# plt.subplots_adjust(hspace=.5)
	ax = fig.add_subplot(gs1[i])
	ax.set_axis_bgcolor('white')
	for spine in ['left','right','top','bottom']:
		ax.spines[spine].set_color('k')

	if i % 5 == 0:
		# ax.xaxis.set_tick_params(width=5)
		ax.get_yaxis().set_tick_params(width=1)
	else:
		ax.set_yticklabels([])

	ax.get_xaxis().set_tick_params(width=1)

	ax.set_axisbelow(True)
	ax.grid(linestyle='-',color='gray',linewidth=.2)
	x = range(3)

	plt.plot(x, full[i],'ro')
	plt.plot(x, turk[i],'co')
	plt.plot(x, lesion[i],'bo')
	plt.title('Trial '+str(i+1)+':\nFull/Turk r=' + str(corrs[i])+'\nLesion/Turk r='+str(corrs2[i]), fontsize=9)
	f = plt.plot(x, full[i],'r',label='Model Prediction',linewidth=1.5)
	# plt.plot(x, turk[i],'co')
	# plt.errorbar(x, turk[i],yerr=[abs(turk[i]-lowCIs[i]),abs(turk[i]-highCIs[i])],fmt='co',linewidth=1.5,barsabove=True,capsize=4)
	t = plt.plot(x, turk[i],'c',label='Participant judgments',linewidth=1.5)
	l = plt.plot(x, lesion[i],'b',label='Lesion judgments',linewidth=1.5)
	plt.xticks(x,fullNames[i],rotation=-20,horizontalalignment='left')
	# plt.ylim([-2.5,2.5])
	plt.ylim([-2.5,2.5])
	plt.margins(.2)
	plt.subplots_adjust(hspace=0.0)

legend = fig.legend((f[0],t[0],l[0]),('Model Prediction','Participant judgments','Lesion Prediction'),fontsize=15,loc=(.78,.20),shadow=True,fancybox=True)
legend.get_frame().set_facecolor('white')
# plt.tight_layout()
plt.savefig('trials.pdf',dpi=300)

"""
Fig5
"""
fig5 = plt.figure(5,figsize=(18,6))
gs5 = gridspec.GridSpec(1,3)
gs5.update(wspace=.08,hspace=.6)

#1
ax = fig5.add_subplot(gs5[0])
ax.set_axis_bgcolor('white')
for spine in ['left','right','top','bottom']:
	ax.spines[spine].set_color('k')
ax.get_yaxis().set_tick_params(width=1)
ax.yaxis.label.set_size(12)
ax.get_xaxis().set_tick_params(width=1)
ax.set_axisbelow(True)
# ax.grid(linestyle='-',color='#D3D3D3',linewidth=.1)
# plt.plot(range(3),turk[4],'ro')
# plt.plot(range(3),turk[4],'r')
# plt.margins(.2)
# plt.ylim([-1.5,1.5])

temp0 = [i[i.Trial==16].Result.values[0] for i in participants]
temp1 = [i[i.Trial==16].Result.values[1] for i in participants]
temp2 = [i[i.Trial==16].Result.values[2] for i in participants]
ci0 = bootstrap_mean(temp0)
ci1 = bootstrap_mean(temp1)
ci2 = bootstrap_mean(temp2)

lo = [turk[15][0]-ci0[0], turk[15][1]-ci1[0], turk[15][2]-ci2[0]]
hi = [ci0[1]-turk[15][0], ci1[1]-turk[15][1], ci2[1]-turk[15][2]]
ax.grid(linestyle='-',linewidth=.20,color='gray')

sns.barplot(range(3),turk[15],ax=ax, yerr=(lo,hi),ecolor='black',error_kw=dict(elinewidth=2,capsize=10,capthick=2),edgecolor='black',linewidth=1)
ax.set_title('Participants',fontsize=16)
plt.ylim([-1.5,2.0])
plt.xticks(x,fullNames[15],rotation=-20,horizontalalignment='center',fontsize=15)
# plt.errorbar(range(3),yerr=[[-1,-1,-1],[1,1,1]])

ax = fig5.add_subplot(gs5[1])
ax.set_axis_bgcolor('white')
for spine in ['left','right','top','bottom']:
	ax.spines[spine].set_color('k')
ax.get_yaxis().set_tick_params(width=1)
ax.set_yticklabels([])
ax.get_xaxis().set_tick_params(width=1)
ax.set_axisbelow(True)
# ax.grid(linestyle='-',color='#D3D3D3',linewidth=.1)
# plt.plot(range(3),full[4],'co')
# plt.plot(range(3),full[4],'c')
# plt.xticks(x,fullNames[4],rotation=-20,horizontalalignment='center',fontsize=12)
# plt.margins(.2)
# plt.ylim([-1.5,1.5])
# ax.set_title('Full Model')
ax.grid(linestyle='-',linewidth=.20,color='gray')

sns.barplot(range(3),full[15],ax=ax,edgecolor='black',linewidth=1)
ax.set_title('Our model',fontsize=16)
plt.ylim([-1.5,2.0])
plt.xticks(x,fullNames[15],rotation=-20,horizontalalignment='center',fontsize=15)

ax = fig5.add_subplot(gs5[2])
ax.set_axis_bgcolor('white')
for spine in ['left','right','top','bottom']:
	ax.spines[spine].set_color('k')
ax.get_yaxis().set_tick_params(width=1)
ax.set_yticklabels([])
ax.get_xaxis().set_tick_params(width=1)
ax.set_axisbelow(True)
# ax.grid(linestyle='-',color='#D3D3D3',linewidth=.1)
# plt.plot(range(3),prior[4],'go')
# plt.plot(range(3),prior[4],'g')
# plt.xticks(x,fullNames[4],rotation=-20,horizontalalignment='center',fontsize=12)
# plt.margins(.2)
# plt.ylim([-1.5,1.5])
# ax.set_title('Prior only')

ax.grid(linestyle='-',linewidth=.20,color='gray')

sns.barplot(range(3),lesion[15],ax=ax,edgecolor='black',linewidth=1)
ax.set_title('Alternative model',fontsize=16)
plt.ylim([-1.5,2.0])
plt.xticks(x,fullNames[15],rotation=-20,horizontalalignment='center',fontsize=15)


fig5.subplots_adjust(bottom=.30)
fig5.text(0.06,0.7,'Judgment',rotation='90',fontsize=20)
fig5.text(0.5,.06,'Hypothesis',fontsize=20)

plt.savefig('fig5.pdf',dpi=300)


"""
Fig6
"""
fig6 = plt.figure(6,figsize=(18,6))
gs6 = gridspec.GridSpec(1,3)
gs6.update(wspace=.08,hspace=.6)

#1
ax = fig6.add_subplot(gs6[0])
ax.set_axis_bgcolor('white')
for spine in ['left','right','top','bottom']:
	ax.spines[spine].set_color('k')
ax.get_yaxis().set_tick_params(width=1)
ax.yaxis.label.set_size(12)
ax.get_xaxis().set_tick_params(width=1)
ax.set_axisbelow(True)
# ax.grid(linestyle='-',color='#D3D3D3',linewidth=.1)
ax.grid(linestyle='-',linewidth=.20,color='black')

# plt.plot(range(3),turk[4],'ro')
# plt.plot(range(3),turk[4],'r')
# plt.margins(.2)
# plt.ylim([-1.5,1.5])

temp0 = [i[i.Trial==16].Result.values[0] for i in participants]
temp1 = [i[i.Trial==16].Result.values[1] for i in participants]
temp2 = [i[i.Trial==16].Result.values[2] for i in participants]
ci0 = bootstrap_mean(temp0)
ci1 = bootstrap_mean(temp1)
ci2 = bootstrap_mean(temp2)

lo = [turk[15][0]-ci2[0], turk[15][1]-ci1[0], turk[15][2]-ci0[0]]
hi = [ci2[1]-turk[15][0], ci1[1]-turk[15][1], ci0[1]-turk[15][2]]

# plt.grid(linewidth=.20,color='gray')
ax.grid(linestyle='-',linewidth=.20,color='black')


sns.barplot(range(3),turk[15],ax=ax, yerr=(lo,hi),ecolor='black',error_kw=dict(elinewidth=2,capsize=10,capthick=2),edgecolor='black',linewidth=1)
ax.set_title('Participants',fontsize=16)
plt.ylim([-1.5,1.5])
plt.xticks(x,fullNames[15],rotation=-20,horizontalalignment='center',fontsize=15)
# plt.errorbar(range(3),yerr=[[-1,-1,-1],[1,1,1]])


ax = fig6.add_subplot(gs6[1])
ax.set_axis_bgcolor('white')
for spine in ['left','right','top','bottom']:
	ax.spines[spine].set_color('k')
ax.get_yaxis().set_tick_params(width=1)
ax.set_yticklabels([])
ax.get_xaxis().set_tick_params(width=1)
ax.set_axisbelow(True)
# ax.grid(linestyle='-',color='#D3D3D3',linewidth=.1)
# plt.plot(range(3),full[4],'co')
# plt.plot(range(3),full[4],'c')
# plt.xticks(x,fullNames[4],rotation=-20,horizontalalignment='center',fontsize=12)
# plt.margins(.2)
# ax.set_title('Full Model')

# plt.grid(linewidth=.20,color='gray')
ax.grid(linestyle='-',linewidth=.20,color='black')



sns.barplot(range(3),full[15],ax=ax,edgecolor='black',linewidth=1)
plt.ylim([-1.5,2.0])
ax.set_title('Our model',fontsize=16)
plt.xticks(x,fullNames[15],rotation=-20,horizontalalignment='center',fontsize=15)



ax = fig6.add_subplot(gs6[2])
ax.set_axis_bgcolor('white')
for spine in ['left','right','top','bottom']:
	ax.spines[spine].set_color('k')
ax.get_yaxis().set_tick_params(width=1)
ax.set_yticklabels([])
ax.get_xaxis().set_tick_params(width=1)
ax.set_axisbelow(True)
# ax.grid(linestyle='-',color='#D3D3D3',linewidth=.1)
# plt.plot(range(3),prior[4],'go')
# plt.plot(range(3),prior[4],'g')
# plt.xticks(x,fullNames[4],rotation=-20,horizontalalignment='center',fontsize=12)
# plt.margins(.2)
# plt.ylim([-1.5,1.5])
# ax.set_title('Prior only')

# plt.grid(linewidth=.20,color='gray')
ax.grid(linestyle='-',linewidth=.20,color='black')


sns.barplot(range(3),lesion[15],ax=ax,edgecolor='black',linewidth=1)
ax.set_title('Alternative model',fontsize=16)
plt.ylim([-1.5,1.5])
plt.xticks(x,fullNames[15],rotation=-20,horizontalalignment='center',fontsize=15)

fig6.subplots_adjust(bottom=.30)
fig6.text(0.06,0.7,'Judgment',rotation='90',fontsize=20)
fig6.text(0.5,.06,'Hypothesis',fontsize=20)

plt.savefig('fig6.pdf',dpi=300)



# """
# Fig7
# """
# fig7 = plt.figure(7,figsize=(18,6))
# gs7 = gridspec.GridSpec(1,3)
# gs7.update(wspace=.08,hspace=.6)

# #1
# ax = fig7.add_subplot(gs7[0])
# ax.set_axis_bgcolor('white')
# for spine in ['left','right','top','bottom']:
# 	ax.spines[spine].set_color('k')
# ax.get_yaxis().set_tick_params(width=1)
# ax.yaxis.label.set_size(12)
# ax.get_xaxis().set_tick_params(width=1)
# ax.set_axisbelow(True)
# ax.grid(linestyle='-',color='#D3D3D3',linewidth=.1)
# # plt.plot(range(3),turk[4],'ro')
# # plt.plot(range(3),turk[4],'r')
# # plt.margins(.2)
# # plt.ylim([-1.5,1.5])

# temp0 = [i[i.Trial==18].Result.values[0] for i in participants]
# temp1 = [i[i.Trial==18].Result.values[1] for i in participants]
# temp2 = [i[i.Trial==18].Result.values[2] for i in participants]
# ci0 = bootstrap_mean(temp0)
# ci1 = bootstrap_mean(temp1)
# ci2 = bootstrap_mean(temp2)

# lo = [turk[20][0]-ci2[0], turk[20][1]-ci1[0], turk[20][2]-ci0[0]]
# hi = [ci2[1]-turk[20][0], ci1[1]-turk[20][1], ci0[1]-turk[20][2]]
# sns.barplot(range(3),turk[20],ax=ax, yerr=(lo,hi),ecolor='black',error_kw=dict(elinewidth=2,capsize=10,capthick=2),edgecolor='black',linewidth=1)
# ax.set_title('Participants')
# plt.ylim([-1.5,1.5])
# plt.xticks(x,fullNames[20],rotation=-20,horizontalalignment='center',fontsize=12)
# # plt.errorbar(range(3),yerr=[[-1,-1,-1],[1,1,1]])


# ax = fig7.add_subplot(gs7[1])
# ax.set_axis_bgcolor('white')
# for spine in ['left','right','top','bottom']:
# 	ax.spines[spine].set_color('k')
# ax.get_yaxis().set_tick_params(width=1)
# ax.set_yticklabels([])
# ax.get_xaxis().set_tick_params(width=1)
# ax.set_axisbelow(True)
# ax.grid(linestyle='-',color='#D3D3D3',linewidth=.1)
# # plt.plot(range(3),full[4],'co')
# # plt.plot(range(3),full[4],'c')
# # plt.xticks(x,fullNames[4],rotation=-20,horizontalalignment='center',fontsize=12)
# # plt.margins(.2)
# # ax.set_title('Full Model')
# sns.barplot(range(3),full[20],ax=ax,edgecolor='black',linewidth=1)
# plt.ylim([-1.5,1.5])
# ax.set_title('Our model')
# plt.xticks(x,fullNames[20],rotation=-20,horizontalalignment='center',fontsize=12)



# ax = fig7.add_subplot(gs7[2])
# ax.set_axis_bgcolor('white')
# for spine in ['left','right','top','bottom']:
# 	ax.spines[spine].set_color('k')
# ax.get_yaxis().set_tick_params(width=1)
# ax.set_yticklabels([])
# ax.get_xaxis().set_tick_params(width=1)
# ax.set_axisbelow(True)
# ax.grid(linestyle='-',color='#D3D3D3',linewidth=.1)
# # plt.plot(range(3),prior[4],'go')
# # plt.plot(range(3),prior[4],'g')
# # plt.xticks(x,fullNames[4],rotation=-20,horizontalalignment='center',fontsize=12)
# # plt.margins(.2)
# # plt.ylim([-1.5,1.5])
# # ax.set_title('Prior only')
# sns.barplot(range(3),lesion[20],ax=ax,edgecolor='black',linewidth=1)
# ax.set_title('Alternative model')
# plt.ylim([-1.5,1.5])
# plt.xticks(x,fullNames[20],rotation=-20,horizontalalignment='center',fontsize=12)


# fig7.subplots_adjust(bottom=.30)
# fig7.text(0.05,0.7,'Judgment',rotation='90',fontsize=15)
# fig7.text(0.5,.06,'Hypothesis',fontsize=15)

# plt.savefig('fig7.pdf')


"""

Mean corr_max for cogsci: .61, std: .21
Mean corr_max for new: .59, std: 


Stimuli Design

Standardized 

Check union of the top 3


3,6,13,14

"""

"""
Checklist:
- Pairwise correlations
- Trial analysis

Look for:
- corr avg for each
- std dev for each

Check corr avg and std dev for exp 2 and exp 3

"""

"""
Trial notes: 
r

"""
