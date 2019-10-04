import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 12, 6

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from math import log
from math import exp

import pandas as pd
from qpython import qconnection
q = qconnection.QConnection(host="qqq.aaa", port=8026)
q.open()
pos = q.sync("`date`time_open xasc delete opening_venues, closing_venues from viewPositions[(2014.01.01 + til(.z.d - 2014.01.01) + 1);``MAXB`MAXC`MAXD`MAXM`MAXP;3]", pandas=True)

from datetime import datetime
from datetime import date
from datetime import time
import math
pos = pos[pos['date'] != date(2015, 8, 24)] # removing Aug 24
pos = pos[pos['date'] >= date(2014, 9, 1)] # remove everything before sept 2014
hedgepos = pos[pos['symbol'].isin(['SPY', 'USO', 'IWM'])] # hedge
posnh = pos[np.logical_not(pos['symbol'].isin(['SPY', 'USO', 'IWM']))].copy().reset_index() # hedge

hedgepos.groupby('symbol').sum().sort_values('shares_traded', ascending=False)

pos['money'] = pos['first_open_price'] * pos['shares_traded']
posnh['money'] = posnh['first_open_price'] * posnh['shares_traded']

bydate = pos[['date', 'shares_traded', 'gross_pnl', 'net_pnl', 'money']].groupby(['date']).sum().reset_index()
bydatenh = posnh[['date', 'shares_traded', 'gross_pnl', 'net_pnl', 'money']].groupby(['date']).sum().reset_index()

bydate['netcum'] = bydate[['net_pnl']].cumsum()
bydate['normnet'] = bydate['net_pnl'] / bydate['money']
median_coeff = bydate['normnet'].median() / bydate['net_pnl'].median()
bydate['normnet'] /= median_coeff

def rs(param1) :
    return (((bydate['normnet'] * param1) - bydate['net_pnl']) ** 2).sum()
vrs = np.vectorize(rs)

from scipy.optimize import brute
coeff = 1.2246413092864188; #brute(rs, ranges=[(0.5, 1.5)])[0]

bydate['normnet'] *= coeff
bydate['index1'] = bydate.index # switching from dates to day indexes
bydate['normnetcum'] = bydate[['normnet']].cumsum() 
fig = plt.figure()
plt.plot(bydate['index1'], bydate['netcum'])
plt.plot(bydate['index1'], bydate['normnetcum'])
plt.title('Net cumulative pnl')

fig = plt.figure()
d1 = bydate[(bydate['date'] > date(2014,5,1)) & (bydate['date'] < date(2015,1,1))]['normnet']
d2 = bydate[(bydate['date'] > date(2015,1,1)) & (bydate['date'] < date(2015,9,14))]['normnet']
d3 = bydate[(bydate['date'] > date(2015,9,14)) & (bydate['date'] < date(2016,2,20))]['normnet']
d4 = bydate[(bydate['date'] > date(2016,2,20)) & (bydate['date'] < date(2017,1,1))]['normnet']

p1 = fig.add_subplot(4,1,1)
p1.hist(d1, alpha=0.5, normed=1)
p2 = fig.add_subplot(4,1,2, sharex=p1)
p2.hist(d2, alpha=0.5, normed=1)
p3 = fig.add_subplot(4,1,3, sharex=p1)
p3.hist(d3, alpha=0.5, normed=1)
p4 = fig.add_subplot(4,1,4, sharex=p1)
p4.hist(d4, alpha=0.5, normed=1)

##########################################################################

x = np.linspace(1.5,7,20)
y = np.vectorize(lambda x: 1/exp(x))(x)
fig = plt.figure()
p1 = fig.add_subplot(1,1,1)
p1.plot(1-y, x)

##########################################################################

from scipy import stats
from datetime import timedelta 

base = bydate[(bydate['date'] > date(2015,1,1)) & (bydate['date'] < date(2015,9,14))]['normnet']

result = []
dates = []
periods = [60, 30, 20, 15, 10, 7]
cutoff = date(2015, 1, 1)
days = (date.today() - cutoff).days
for c in range(days):
    cutoff = cutoff + timedelta(days=1)
    lbase = bydate[(bydate['date'] > (cutoff - timedelta(days=120))) & (bydate['date'] < (cutoff - timedelta(days=30)))]['normnet']
    sample = lambda x: bydate[(bydate['date'] > (cutoff - timedelta(days=x))) & (bydate['date'] < cutoff)]['normnet']
    samples = map(sample, periods)
    if (cutoff > date(2015, 10, 1)):
        lbase = base
    result.append(map(lambda x: (log(1/stats.anderson_ksamp([lbase, samples[x]]).significance_level)) if len(samples[x]) > 0 else 0, range(len(periods))))
    dates.append(cutoff)
	
fig = plt.figure()
#p1 = fig.add_subplot(len(periods) + 1,1,1)
p1 = fig.add_subplot(3,1,1)
alldates = []
allresults = []
tresult = np.asarray(result).T.tolist() # transpose result
for x1 in range(0, len(periods)):
    alldates += dates
    allresults += tresult[x1]

convdate = lambda d: (d - date(2014,1,1)).days
alldates = map(convdate, alldates)
p1.hexbin(alldates, allresults, bins='log', cmap=plt.cm.YlOrRd_r)
p1.plot(alldates, [3] * len(alldates))
p1.set_title("by period")
p2 = fig.add_subplot(3,1,2, sharex=p1)
p2.plot(map(convdate, dates), np.apply_along_axis(lambda x: len(x[x>3]), 1, np.asarray(result)) / float(len(periods)))
p2.set_title("% of periods abnormal")
p3 = fig.add_subplot(3,1,3, sharex=p1)
todt = lambda d: d.to_datetime().date()
p3.plot(map(convdate, map(todt, bydate.date)), bydate.normnetcum)

##########################################################################

fig = plt.figure(dpi=80, figsize=(8,6))
plots = len(periods) + 2
p1 = fig.add_subplot(plots,1,1)
p1.plot(dates, np.asarray(result).T.tolist()[0])
p1.plot(dates, [3] * len(dates))
for x in range(1, len(periods)):
    pz = fig.add_subplot(plots,1,x + 1, sharex=p1)
    pz.plot(dates, np.asarray(result).T.tolist()[x])
    pz.plot(dates, [3] * len(dates))
p2 = fig.add_subplot(plots,1,plots - 1, sharex=p1)
p2.plot(bydate.date, bydate.normnetcum)
p3 = fig.add_subplot(plots,1,plots, sharex=p1)
p3.bar(map(lambda d: d.to_datetime(), bydate.date), bydate.normnet, width=0.35)
p3.xaxis_date()

##########################################################################

sigs = pd.DataFrame({'date': map(lambda x: pd.Timestamp(x), dates)})
for c in range(len(periods)):
    sigs[str(periods[c])] = np.asarray(result).T.tolist()[c]
sigs = pd.merge(sigs, bydate[['date', 'net_pnl', 'normnet']], on='date', left_index=False, right_index=False, suffixes=('_x', '_y'), how='inner')
fig = plt.figure()
days = 20
pix = periods.index(15)
p1 = fig.add_subplot(3,1,1)
p1.plot(dates[-days:], np.asarray(result).T.tolist()[pix][-days:])
p1.plot(dates[-days:], [3] * len(dates[-days:]))
p2= fig.add_subplot(3,1,2, sharex=p1)
p2.bar(map(lambda d: d.to_datetime(), sigs.date[-days:]), sigs['normnet'][-days:], width=0.35)
p3= fig.add_subplot(3,1,3, sharex=p1)
p3.plot(sigs.date[-days:], sigs['normnet'][-days:].cumsum())

#########################################################################

groups = pd.read_csv("groups.csv")
groups.set_index('sym', inplace=True)
grp = lambda sym: groups.group[sym] if sym in groups.index else -1

grpv = np.vectorize(grp)
pos['group'] = grpv(pos.symbol)
posnh['group'] = grpv(posnh.symbol)
bygrd = posnh[['date', 'shares_traded', 'gross_pnl', 'net_pnl', 'money', 'group']].groupby(['date', 'group']).sum().reset_index()
grplist = bygrd.group.unique()
bygrds = map(lambda g: bygrd[bygrd.group == g].copy().reset_index(), grplist)
for g in range(len(bygrds)):
    bygrds[g]['normnet'] = bygrds[g]['net_pnl'] / bydate['money'] / median_coeff * coeff
    bygrds[g]['normnetcum'] = bygrds[g]['normnet'].cumsum()

gresult = []
gdates = []
for g in range(len(bygrds)):
    test = bygrds[g]
    base = test[(test['date'] > date(2014,5,1)) & (test['date'] < date(2015,9,14))]['normnet']
    result = []
    dates = []
    cutoff = date(2015, 1, 1)
    days = (date.today() - cutoff).days
    for c in range(days):
        cutoff = cutoff + timedelta(days=1)
        lbase = test[(test['date'] > (cutoff - timedelta(days=90))) & (test['date'] < (cutoff - timedelta(days=15)))]['normnet']
        sample = test[(test['date'] > (cutoff - timedelta(days=15))) & (test['date'] < cutoff)]['normnet']
        if (len(sample) == 0):
            continue
        if (cutoff > date(2015, 10, 10)):
            lbase = base
        result.append(log(1/stats.anderson_ksamp([base, sample]).significance_level))
        dates.append(cutoff)
    gresult.append(result)
    gdates.append(dates)

import itertools

fig = plt.figure()
p1 = fig.add_subplot(1,1,1)
cdates = list(itertools.chain(*gdates))
cresults = list(itertools.chain(*gresult))
p1.hexbin(map(convdate, cdates), cresults)
	
#########################################################################
	
data = pd.concat(map(lambda x: x[['date', 'normnetcum', 'group', 'normnet']].copy(), bygrds))
pivoted = data.pivot(index='date', columns='group', values='normnetcum')
pivoted = pivoted.fillna(method='ffill')
sum = pivoted.sum(axis=1)
normed = pivoted.div(sum, axis=0)
sorted = normed.T.sort_values('2016-03-10 00:00:00')

sortedp = pivoted.T.sort_values('2016-03-10 00:00:00')	
fig = plt.figure()
p1 = fig.add_subplot(1,1,1)
p1.pcolor(sorted, cmap='RdBu', vmin=-0.07, vmax=0.07)

#########################################################################

sortedneg = sorted.copy()
sortedpos = sorted.copy()

sortedneg[sortedneg > 0] = 0
sortedpos[sortedneg < 0] = 0
sortedneg = sortedneg.iloc[::-1]

sortedneg = np.cumsum(sortedneg).T
sortedpos = np.cumsum(sortedpos).T

sortedpos = sortedpos.reset_index(drop=True).T.reset_index(drop=True).T
sortedneg = sortedneg.reset_index(drop=True).T.reset_index(drop=True).T

def prisma(rng=[-2,2]):
    fig = plt.figure();
    p1 = fig.add_subplot(1,1,1)

    p1.set_ylim(rng)

    for g in range(80,0,-1):
        p1.fill_between(range(len(sortedpos[g])), 0, sortedpos[g], color=plt.cm.prism(g/80.0))
        p1.fill_between(range(len(sortedneg[g])), 0, sortedneg[g], color=plt.cm.prism(1-g/80.0))

prisma()		

#########################################################################

def redblue(rng=[-2,2]):
    fig = plt.figure();
    p1 = fig.add_subplot(1,1,1)

    p1.set_ylim(rng)

    for g in range(80,0,-1):
        p1.fill_between(range(len(sortedpos[g])), 0, sortedpos[g], color=plt.cm.RdBu(g/80.0))
        p1.fill_between(range(len(sortedneg[g])), 0, sortedneg[g], color=plt.cm.RdBu(1-g/80.0))
		
redblue()		

#########################################################################

sortedneg = sortedp.copy()
sortedpos = sortedp.copy()

sortedneg[sortedneg > 0] = 0
sortedpos[sortedneg < 0] = 0
sortedneg = sortedneg.iloc[::-1]

sortedneg = np.cumsum(sortedneg).T
sortedpos = np.cumsum(sortedpos).T

sortedpos = sortedpos.reset_index(drop=True).T.reset_index(drop=True).T
sortedneg = sortedneg.reset_index(drop=True).T.reset_index(drop=True).T

prisma([-2000000,4000000])
redblue([-2000000,4000000])
	
#########################################################################	

pnlts = bydate[['date', 'normnet']][(bydate.date > date(2015,1,1)) & (bydate.date < date(2015,9,1))].copy()
pnlts.set_index(['date'], inplace=True)
		
from scipy.stats import gaussian_kde
import matplotlib.mlab as mlab
fig, ax = plt.subplots(figsize=(12,6))  
n, bins, patches = ax.hist(pnlts.normnet, 25, normed=1, facecolor='green', alpha=0.75)

bincenters = 0.5*(bins[1:]+bins[:-1])

nmean = np.mean(pnlts.normnet)
nstd = np.std(pnlts.normnet)
y = mlab.normpdf( bincenters, nmean, nstd )
l = ax.plot(bincenters, y, 'r--', linewidth=1)		

#########################################################################

pnlw = bydate[['date', 'normnet', 'net_pnl']][bydate.date > date.today() - timedelta(14)].copy()
pnlw['coeff'] = pnlw.net_pnl / pnlw.normnet
print ""
print pnlw


last_norm_coeff = pnlw.coeff[-1:].iloc[0]
from scipy.stats import norm
losses = np.arange(-np.sum(pnlw.net_pnl) / 2, 0, np.sum(pnlw.net_pnl) / 20)
norm.cdf(losses, nmean, nstd)
print ""
print pd.DataFrame({"loss": losses, "2wk %": losses / np.sum(pnlw.net_pnl), "prob": norm.cdf(losses, nmean * last_norm_coeff * 5, nstd * last_norm_coeff * 5)})

prc10loss = norm.ppf(0.1, nmean * last_norm_coeff * 5, nstd * last_norm_coeff * 5)
print ""
print "amount we can lose in a week with 10% probability"
print prc10loss

print ""
print "expected gain over the next week"
print nmean * last_norm_coeff * 5

print ""
print "fraction of 2 week pnl we can lose using the current scaling with 10% probability"
print abs(prc10loss) / np.sum(pnlw.net_pnl)

print ""	
print "fraction of 2 week pnl we can lose with 10% probability with coefficient raised 20%"
print abs(norm.ppf(0.1, nmean * last_norm_coeff * 5 * 1.2, nstd * last_norm_coeff * 5 * 1.2)) / np.sum(pnlw.net_pnl)

print ""	
print "fraction of 2 week pnl we can lose with 10% probability with coefficient lowered 20%	"
print abs(norm.ppf(0.1, nmean * last_norm_coeff * 5 * 0.8, nstd * last_norm_coeff * 5 * 0.8)) / (np.sum(pnlw.net_pnl))
	
plt.show()

