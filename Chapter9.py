
#####################
## CHAPTER 9 : Data Aggregation and Group Operations
#####################

# import numpy and pandas packages
import numpy as np 
import pandas as pd 

PREVIOUS_MAX_ROWS = pd.options.display.max_rows #
np.random.seed(12345) 
import matplotlib.pyplot as plt # matplotlib package
plt.rc('figure',figsize=(10,6)) #  figure size
np.set_printoptions(precision=4, suppress =True) ##


############
# GroupBy Mechanics
# first define data: 

df= pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'], 
                  'key2' : ['one','two','one', 'two', 'one'],
                  'data1' : np.random.random(5),
                  'data2' : np.random.randn(5)})



grouped = df['data1'].groupby(df['key1']) # .groupby() is useing for group

grouped

grouped.mean()

# 

means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means
means.unstack()
#
# now add two more column to df

states= np.array(['Ohio','California', 'California', 'Ohio','Ohio'])
years=np.array([2005,2005,2006,2005,2006])

df['data1'].groupby([states,years]).mean()
#
df.groupby('key1').mean()
df.groupby(['key1', 'key2']).mean() # group by key1 and key2 then take mean of both data1 and data2

# size() function give a series  group size. 

df.groupby(['key1', 'key2']).size()
###########################
# Iterating Over Groups

for name, group in df.groupby('key1'):
    print(name)
    print(group)


for (k1, k2), group in df.groupby(['key1', 'key2']):
    print((k1, k2))
    print(group)

##
pieces = dict(list(df.groupby('key1')))
pieces['b']

df.dtypes
grouped = df.groupby(df.dtypes, axis=1)

for dtype, group in grouped:
    print(dtype)
    print(group)
####################
# Selecting a Column or Subset of Columns
    
df.groupby(['key1', 'key2'])[['data2']].mean() # groub by key1 and then key2 and then take mean of data2


s_grouped = df.groupby(['key1', 'key2'])['data2']
s_grouped
s_grouped.mean()

###############
# Grouping with Dicts and Series

# if we have data in form rather than an array: 
people = pd.DataFrame(np.random.randn(5, 5),
                      columns=['a', 'b', 'c', 'd', 'e'],
                      index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])

people.iloc[2:3, [1, 2]] = np.nan # Add a few NA values
people

# now mapping: 
mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd': 'blue', 'e': 'red', 'f' : 'orange'}

by_column = people.groupby(mapping, axis=1)
by_column.sum()

# now inspect each Series to ensure that its index is aligned with the axis it's grouping: 
map_series = pd.Series(mapping)
map_series
people.groupby(map_series, axis=1).count()

# Grouping with Functions

people.groupby(len).sum() # grouop by length of the names and then get sum

key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()

# Group by Index Levels: 

columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]],
                                    names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
hier_df

##############
# Data Aggregation: 


df
grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)

# define function : 
def peak_to_peak(arr):
    return arr.max() - arr.min()
# now use agg function to pass the above function: 
grouped.agg(peak_to_peak)


grouped.describe() # group data and give summary of data


############################
# Column-Wise and Multiple Function Application: 
#
# use data set: 
tips=pd.read_csv()

tips = pd.read_csv('D:/Python/Python for Data Analysis/examples/tips.csv')
# Add tip percentage of total bill
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips[:6] # tips[columns,rows]

grouped = tips.groupby(['day', 'smoker'])

grouped_pct = grouped['tip_pct']
grouped_pct.agg('mean')

grouped_pct.agg(['mean', 'std', peak_to_peak])


grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])

# or use function: 

functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
result
result['tip_pct']

ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
grouped['tip_pct', 'total_bill'].agg(ftuples)

grouped.agg({'tip' : np.max, 'size' : 'sum'})
grouped.agg({'tip_pct' : ['min', 'max', 'mean', 'std'], 'size' : 'sum'})


#Returning Aggregated Data Without Row Indexes

tips.groupby(['day', 'smoker'], as_index=False).mean()

######################

# Apply: General split-apply-combine

# define "top" function to give top 5 with sorting value of 'tip_" column
def top(df, n=5, column='tip_pct'):
    return df.sort_values(by=column)[-n:] # this is way to write a function
top(tips, n=6)

tips.groupby('smoker').apply(top) # use apply function to adop our logic from top function


tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')

result = tips.groupby('smoker')['tip_pct'].describe()
result
result.unstack('smoker') # re write our output

# Suppressing the Group Keys

tips.groupby('smoker', group_keys=False).apply(top) # disable Index


# Quantile and Bucket Analysis

frame = pd.DataFrame({'data1': np.random.randn(1000),
                      'data2': np.random.randn(1000)})
quartiles = pd.cut(frame.data1, 4)
quartiles[:10]


def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
grouped = frame.data2.groupby(quartiles)
grouped.apply(get_stats).unstack()


# Return quantile numbers
grouping = pd.qcut(frame.data1, 10, labels=False)
grouped = frame.data2.groupby(grouping)
grouped.apply(get_stats).unstack()

###########################################################################################
# Example: Filling Missing Values with Group-Specific Values
##########################################################################################

s = pd.Series(np.random.randn(6))
s[::2] = np.nan # put nan for even numbers
s
s.fillna(s.mean()) # replace nan with mean


states = ['Ohio', 'New York', 'Vermont', 'Florida',
          'Oregon', 'Nevada', 'California', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4
data = pd.Series(np.random.randn(8), index=states)
data

data[['Vermont', 'Nevada', 'Idaho']] = np.nan
data
data.groupby(group_key).mean()



fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)


fill_values = {'East': 0.5, 'West': -1}
fill_func = lambda g: g.fillna(fill_values[g.name])
data.groupby(group_key).apply(fill_func)


###########################################################################################
# Example: Random Sampling and Permutation
##########################################################################################

# Hearts, Spades, Clubs, Diamonds
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)

deck = pd.Series(card_val, index=cards)

deck[:13]
#
def draw(deck, n=5):
    return deck.sample(n)
draw(deck)

get_suit = lambda card: card[-1] # last letter is suit
deck.groupby(get_suit).apply(draw, n=2)

deck.groupby(get_suit, group_keys=False).apply(draw, n=2)


###########################################################################################
# Example: Group Weighted Average and Correlation
##########################################################################################

df = pd.DataFrame({'category': ['a', 'a', 'a', 'a',
                                'b', 'b', 'b', 'b'],
                   'data': np.random.randn(8),
                   'weights': np.random.rand(8)})
df


grouped = df.groupby('category')
get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
grouped.apply(get_wavg)


close_px = pd.read_csv('D:/Python/Python for Data Analysis/examples/stock_px_2.csv', parse_dates=True,
                       index_col=0)
close_px.info()
close_px[-4:]

spx_corr = lambda x: x.corrwith(x['SPX']) # correlation

rets = close_px.pct_change().dropna() # drop na

get_year = lambda x: x.year
by_year = rets.groupby(get_year)
by_year.apply(spx_corr)

by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))

###########################################################################################
#  Example: Group-Wise Linear Regression
###########################################################################################

import statsmodels.api as sm
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X).fit() # fit ols for (y,x)
    return result.params


by_year.apply(regress, 'AAPL', ['SPX'])


###########################################################################################
# Example: Pivot Tables and Cross-Tabulation
###########################################################################################

tips.pivot_table(index=['day', 'smoker'])


tips.pivot_table(['tip_pct', 'size'], index=['time', 'day'],
                 columns='smoker')

tips.pivot_table(['tip_pct', 'size'], index=['time', 'day'],
                 columns='smoker', margins=True)

tips.pivot_table('tip_pct', index=['time', 'smoker'], columns='day',
                 aggfunc=len, margins=True)

tips.pivot_table('tip_pct', index=['time', 'size', 'smoker'],
                 columns='day', aggfunc='mean', fill_value=0)


###########################################################################################
# Example: Cross-Tabulations: Crosstab
###########################################################################################

from io import StringIO
data = """\
Sample  Nationality  Handedness
1   USA  Right-handed
2   Japan    Left-handed
3   USA  Right-handed
4   Japan    Right-handed
5   Japan    Left-handed
6   Japan    Right-handed
7   USA  Right-handed
8   USA  Left-handed
9   Japan    Right-handed
10  USA  Right-handed"""
data = pd.read_table(StringIO(data), sep='\s+')

data

pd.crosstab(data.Nationality, data.Handedness, margins=True)

pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)

pd.options.display.max_rows = PREVIOUS_MAX_ROWS


##########################################################################################


