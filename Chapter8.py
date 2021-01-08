
#####################
## CHAPTER 8 : Ploting and Visualization
#####################



# 
import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.random.seed(12345)
import matplotlib.pyplot as plt  # package for plot 
import matplotlib # package for plot
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

#############
data = np.arange(10)
data
plt.plot(data)
# create  a new figure with plt.figure function: 

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

from numpy.random import randn

plt.plot(np.random.randn(50).cumsum(), 'k--') # to have black dash plot use " 'k--' "


_ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))

plt.close('all')

fig, axes = plt.subplots(2, 3) # 2*3 plots at the same time. 
axes

#############
## Adjusting the spacing around subplots

# wspace and hspace are controls width and height of plot

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True) # define 2*2 plots
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)  # histogram plot
plt.subplots_adjust(wspace=0, hspace=0) # control both y and x axes. 

############33

plt.figure()

from numpy.random import randn
plt.plot(randn(30).cumsum(), 'ko--')

data = np.random.randn(30).cumsum()
plt.plot(data, 'k--', label='Default')
plt.plot(data, 'k-', drawstyle='steps-post', label='steps-post')
plt.legend(loc='best')

###################
# Setting the title, axis labels, ticks, and ticklabels
####

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())


ticks = ax.set_xticks([0, 250, 500, 750, 1000]) # change x axis ticks
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                            rotation=30, fontsize='small') # x axis 
ax.set_title('My first matplotlib plot') # title
ax.set_xlabel('Stages') # x axis title

####
# Adding legends: 

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum(), 'k', label='one')
ax.plot(randn(1000).cumsum(), 'k--', label='two')
ax.plot(randn(1000).cumsum(), 'k.', label='three')
ax.legend(loc='best') 3 give a location of legend, "loc=.."
############3

# Annotations and Drawing on a Subplot

from datetime import datetime


data = pd.read_csv('D:/Python/Python for Data Analysis/examples/spx.csv', index_col=0, parse_dates=True)
spx = data['SPX']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

spx.plot(ax=ax, style='k-')

crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]

for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 75),
                xytext=(date, spx.asof(date) + 225),
                arrowprops=dict(facecolor='black', headwidth=4, width=2,
                                headlength=4),
                horizontalalignment='left', verticalalignment='top')
# Zoom in on 2007-2010
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600, 1800])

ax.set_title('Important dates in the 2008-2009 financial crisis')

#
fig = plt.figure(figsize=(12, 6)); ax = fig.add_subplot(1, 1, 1)
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3) # rectangle
circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3) # circle
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                   color='g', alpha=0.5) # ploygon
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)

#################3
## Saving Plots to File

plt.savefig('figpath.svg')

plt.savefig('figpath.png', dpi=400, bbox_inches='tight')

from io import BytesIO buffer = BytesIO() plt.savefig(buffer) plot_data = buffer.getvalue()


###############
# Line Plots

plt.close('all')

s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
s.plot()

# couple data                    (row,column)
df = pd.DataFrame(np.random.randn(10, 4).cumsum(0),
                  columns=['A', 'B', 'C', 'D'],
                  index=np.arange(0, 100, 10))
df.plot() 

# Bar Plots
fig, axes = plt.subplots(2,1)
data = pd.Series(np.random.rand(16), index = list('abcdefghijklmnop'))
data.plot.bar(ax=axes[0], color = 'k', alpha =0.7) # while axes[0] shows "ab..." on x axes
data.plot.barh(ax=axes[1],color='k',alpha=0.7)

#
np.random.seed(12348)
df = pd.DataFrame(np.random.rand(6, 4),
                  index=['one', 'two', 'three', 'four', 'five', 'six'],
                  columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df
df.plot.bar() # bar plot

#
plt.figure()
df.plot.barh(stacked=True, alpha=0.5)
plt.close('all')
tips = pd.read_csv('D:/Python/Python for Data Analysis/examples/tips.csv')
party_counts = pd.crosstab(tips['day'], tips['size'])
party_counts
# Not many 1- and 6-person parties
party_counts = party_counts.loc[:, 2:5]
# Normalize to sum to 1
party_pcts = party_counts.div(party_counts.sum(1), axis=0)
party_pcts
party_pcts.plot.bar()
plt.close('all')
##
import seaborn as sns
tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
tips.head()
sns.barplot(x='tip_pct', y='day', data=tips, orient='h')

plt.close('all')

sns.barplot(x='tip_pct', y='day', hue='time', data=tips, orient='h')

plt.close('all')

sns.set(style="whitegrid")

# Histograms and Density Plots

plt.figure()
tips['tip_pct'].plot.hist(bins=50)

plt.figure()
tips['tip_pct'].plot.density()

plt.figure()
comp1 = np.random.normal(0, 1, size=200)
comp2 = np.random.normal(10, 2, size=200)
values = pd.Series(np.concatenate([comp1, comp2]))
sns.distplot(values, bins=100, color='k')

#Scatter or Point Plots

macro = pd.read_csv('D:/Python/Python for Data Analysis/examples/macrodata.csv')
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
trans_data = np.log(data).diff().dropna()
trans_data[-5:]

plt.figure()
sns.regplot('m1', 'unemp', data=trans_data)
plt.title('Changes in log %s versus log %s' % ('m1', 'unemp'))

sns.pairplot(trans_data, diag_kind='kde', plot_kws={'alpha': 0.2}) # have a pair plot

#############3
# Facet Grids and Categorical Data

sns.factorplot(x='day', y='tip_pct', hue='time', col='smoker',
               kind='bar', data=tips[tips.tip_pct < 1])

sns.factorplot(x='day', y='tip_pct', row='time',
               col='smoker',
               kind='bar', data=tips[tips.tip_pct < 1])

sns.factorplot(x='tip_pct', y='day', kind='box',
               data=tips[tips.tip_pct < 0.5])


pd.options.display.max_rows = PREVIOUS_MAX_ROWS
