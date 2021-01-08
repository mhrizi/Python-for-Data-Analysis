
#####################
## CHAPTER 11: Advanced NumPy
#####################


import numpy as np
import pandas as pd
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)

############################################
#Categorical Data
############################################

###############
# Background and Motivation
###############

values = pd.Series(['apple', 'orange', 'apple',
                    'apple'] * 2)
values
pd.unique(values) # only unique
pd.value_counts(values) # count value 

values = pd.Series([0, 1, 0, 0] * 2)
dim = pd.Series(['apple', 'orange'])
values
dim

dim.take(values) # merge both data

############
# Categorical Type in pandas
############

fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits) # use "len()" function to get length of data
df = pd.DataFrame({'fruit': fruits,
                   'basket_id': np.arange(N),
                   'count': np.random.randint(3, 15, size=N), # randint()
                   'weight': np.random.uniform(0, 4, size=N)}, # uniform()
                  columns=['basket_id', 'fruit', 'count', 'weight'])
df


fruit_cat = df['fruit'].astype('category')
fruit_cat

c = fruit_cat.values
type(c) # type() function for type of our data

c.categories
c.codes

df['fruit'] = df['fruit'].astype('category')
df.fruit

my_categories = pd.Categorical(['foo', 'bar', 'baz', 'foo', 'bar'])
my_categories

categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
my_cats_2 = pd.Categorical.from_codes(codes, categories)
my_cats_2

ordered_cat = pd.Categorical.from_codes(codes, categories,
                                        ordered=True)
ordered_cat

my_cats_2.as_ordered()


#################
# Computations with Categoricals
#################

np.random.seed(12345)
draws = np.random.randn(1000)
draws[:5]

bins = pd.qcut(draws, 4)
bins

bins = pd.qcut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
bins
bins.codes[:10]

bins = pd.Series(bins, name='quartile')
results = (pd.Series(draws)
           .groupby(bins)
           .agg(['count', 'min', 'max'])
           .reset_index())
results

results['quartile']

#############
# Better performance with categoricals
############

N = 10000000
draws = pd.Series(np.random.randn(N))
labels = pd.Series(['foo', 'bar', 'baz', 'qux'] * (N // 4))

categories = labels.astype('category')

labels.memory_usage()
categories.memory_usage()

%time _ = labels.astype('category')

#############
# Categorical Methods
#############

s = pd.Series(['a', 'b', 'c', 'd'] * 2)
cat_s = s.astype('category')
cat_s

cat_s.cat.codes
cat_s.cat.categories

actual_categories = ['a', 'b', 'c', 'd', 'e']
cat_s2 = cat_s.cat.set_categories(actual_categories)
cat_s2

cat_s.value_counts()
cat_s2.value_counts()

cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
cat_s3
cat_s3.cat.remove_unused_categories()
## Creating dummy variables for modeling

cat_s = pd.Series(['a', 'b', 'c', 'd'] * 2, dtype='category')

pd.get_dummies(cat_s)

###########################
#Advanced GroupBy Use
###########################


### Group Transforms and "Unwrapped" GroupBys
df = pd.DataFrame({'key': ['a', 'b', 'c'] * 4,
                   'value': np.arange(12.)})
df

g = df.groupby('key').value
g.mean()

g.transform(lambda x: x.mean())

g.transform('mean')

g.transform(lambda x: x * 2)

g.transform(lambda x: x.rank(ascending=False))

def normalize(x):
    return (x - x.mean()) / x.std()

g.transform(normalize)
g.apply(normalize)

g.transform('mean')
normalized = (df['value'] - g.transform('mean')) / g.transform('std')
normalized

## Grouped Time Resampling

N = 15
times = pd.date_range('2017-05-20 00:00', freq='1min', periods=N)
df = pd.DataFrame({'time': times,
                   'value': np.arange(N)})
df

df.set_index('time').resample('5min').count()

df2 = pd.DataFrame({'time': times.repeat(3),
                    'key': np.tile(['a', 'b', 'c'], N),
                    'value': np.arange(N * 3.)})
df2[:7]

#time_key = pd.TimeGrouper('5min')

resampled = (df2.set_index('time')
             .groupby(['key', time_key])
             .sum())
resampled
resampled.reset_index()

#########