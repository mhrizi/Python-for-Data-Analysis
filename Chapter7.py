
#####################
## CHAPTER 7: Data Wrangling: Clean, Transform, Merge, Reshape
#####################



import pandas as pd

import numpy as np


# pandas.merge
# pandas.concat
#combin_first

df1= pd.DataFrame({'Key' : ['b','b','a','c','a','a','b'], 'data1' : range(7)})


df2=pd.DataFrame({'Key' : ['a','b','d'], 'data2': range(3)})


pd.merge(df1,df2)

pd.merge(df1,df2, on='Key')

# by default 'merge' is inner join, 

pd.merge(df1,df2, how='outer')

pd.merge(df1,df2, on='Key', how='left')

pd.merge(df1,df2, on='Key', how='inner')


# Merging On Index: us 'left_index' or 'right_index'

lefth=pd.DataFrame({'key1' : ['Ohio', 'Ohio', 'Nevada'], 
                    'key2': [2000,2001, 2001], 
                    'data': np.arange(3)})

righth=pd.DataFrame(np.arange(12).reshape((6,2)),
                    index=[['Nevada','Nevada','Ohio',
                            'Ohio','Ohio','Ohio'],
                          [2001, 2000, 2000, 2000,
                                                2001, 2002]],
                            columns=['event1','event2'])
 
                             
                                          
pd.merge(lefth,righth, left_on=['key1','key2'],right_index=True)

pd.merge(lefth,righth, left_on=['key1','key2'],right_index=True, how='outer')

# if there is overlap in index then : 

lefth.join(righth,how='outer')


# Concatenating Along an Axis from NumPy package: 

arr=np.arange(12).reshape((3,4))

arr

# now Merge: 

np.concatenate([arr,arr],axis=1)

# 

s1=pd.Series([0,1],index=['a','b'])

s2=pd.Series([2,3,4],index=['c','d','e'])

s3=pd.Series([5,6],index=['f','g'])


pd.concat([s1,s2,s3])

pd.concat([s1,s2,s3],axis=1)

pd.concat([s1,s2,s3],keys=['one','two','three'])

pd.concat([s1,s2,s3],axis=1,keys=['one','two','three'])


# Reshaping and Pivoting: 

data=pd.DataFrame(np.arange(6).reshape((2,3)),
                  index=pd.Index(['Ohio','Colorado'], name='state'),
                  columns=pd.Index(['one','two','three'], name='number'))

# stack: to pivotes the columns into the rows: 

result=data.stack()
result1=result.unstack()

# pivot funstion: 
##ldata[:10]
##data1=ldata.pivot('date','item', 'value')

# Removing Duplicates: 

data=pd.DataFrame({'k1' : ['one'] * 3 + ['two'] * 4 ,
                'k2' : [1,1,2,3,3,4,4]})

# dublicated() function give if there is dublicate in our data
data.duplicated()


data.drop_duplicates()

 # filte rduplicates only based on the "k1" column:
 
data['v1']=range(7)


data.drop_duplicates(['k1'])

# if we want to remove the first and return the last ones: 

data.drop_duplicates(['k1','k2'], keep='last')

#Transforming Data Using a Function or Mapping

data= pd.DataFrame({'food': ['bacon','pulled pork', 'bacon', 'Pastrami','corned beef', 'Bacon', 'pastrami', 'honey ham', 'nova lox'],
                          'ounces' : [4,3,12,6,7.5, 8, 3, 5, 6]})
data

# write down a mapping of each meat to animal:

meat_to_animal = { 'bacon': 'pig',
                  'pulled pork': 'pig',
                  'pastrami' : 'cow',
                  'corned beef': 'cow',
                  'honey ham': 'pig',
                  'nova lox': 'salmon'
                  }
    

# have all value in lower case and mapp: 

# use 'map' function: 
data['animal']=data['food'].map(str.lower).map(meat_to_animal)

# another way: 

data['food'].map(lambda x: meat_to_animal[x.lower()])

### Replaceing Values: 

data = pd.Series([1., -999., 2., -999., -1000., 3.])

# replace -999 with np.nan as missing: 

data.replace(-999, np.nan)

# now suppose we want to replace  more values:

data.replace([-999,-1000], [np.nan, 0])

# or: 
data.replace({-999: np.nan, -1000: 0})


# create a data frame with 3 row and 4 column: 

data = pd.DataFrame(np.arange(12).reshape((3,4)),
                    index= ['Ohio', 'Colorado','New york'],
                    columns=['one', 'two', 'three','four'])
# first upper: 

data.index.map(str.upper)
data.index=data.index.map(str.upper)
# Rename function: 

data.rename(index=str.title, columns=str.upper)

# rename : 

data.rename(index={'OHIO': 'INDIANA'}, columns={'three': 'peekaboo'})

# alwayes a reference to a DataFrame: 

data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
data

#### Discretization and Binning: 

ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]

# creaet bin : creaet 18-25, 26-35, 36-60: use "cut" function:

bins= [18, 25, 60, 100]

cats = pd.cut(ages, bins)

# rightside of interval is open: 

pd.cut(ages, [18, 26, 36, 61, 100], right=False)

## give name for each bin: 

group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']

pd.cut(ages, bins, labels=group_names)


### Normaly distribute d: 
data=np.random.rand(1000)

# cut into quantile: 
cuts=pd.qcut(data,4)

pd.value_counts(cats)

##Detecting and Filtering Outliers

data=pd.DataFrame(np.random.rand(1000,4))

data.describe() # give data information

# suppose we want to have number greater than 3 in column 2 of data: 

col = data[3]
col[np.abs(col) > 0.23]

# 'any()' will give rows 
data[(np.abs(data) > 0.23).any(1)]

# what is "sign()" function? 

data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()
np.sign(data).head()


### Permutation and Random Sampling: 

df1=pd.DataFrame(np.arange(5*4).reshape(5,4))

# use premutation() function for samling:

sampler=np.random.permutation(5)

df1
df1.take(sampler)

### sample with resampling :: np.random.randint() function: 


bag=np.array([5,7,-1,6,4])

sampler=np.random.randint(0,len(bag), size=10)

# use "take()" function to get sample: 
draws=bag.take(sampler)

##################
#Computing Indicator/Dummy Variables
######
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                   'data1': range(6)})

# "get_dummies()" will create dummy variable :( here based on column "key")
pd.get_dummies(df['key'])

# add dummies column with prefix: 

dummies=pd.get_dummies(df['key'],prefix='key')

df_with_dummy=df[['data1']].join(dummies)

#### Example with dataset of movie: 

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('D:/Python/Python for Data Analysis/datasets/movielens/movies.dat', sep='::', header=None, names=mnames)


movies[:10]


all_genres = []
     for x in movies.genres:
        all_genres.extend(x.split('|'))
genres = pd.unique(all_genres)

# set zero all now:
zero_matrix = np.zeros((len(movies), len(genres)))
dummies = pd.DataFrame(zero_matrix, columns=genres)


gen = movies.genres[0]
gen.split('|')
dummies.columns.get_indexer(gen.split('|'))


for i, gen in enumerate(movies.genres):
    indices = dummies.columns.get_indexer(gen.split('|'))
    dummies.iloc[i, indices] = 1

movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.iloc[0]

# another example for dummy: 

## use "get_dummies" function
np.random.seed(12345)
values = np.random.rand(10)
values
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))

######################
# Srring Manipulation: 
#####################

#String Object Methods

# 1 separated string: "split()" function : 

val='a,b, guido'
val.split(',')
# to trim whitespace 
pieces = [x.strip() for x in val.split(',')]
pieces

# use two-colon delimiter: 

first, second, third = pieces
first + '::' + second + '::' + third
# use faster 
'::'.join(pieces)

##########
#Regular Expressions
##

## "re" module should be import: 

import re

text= " foo     bar\t baz   \tqux"

re.split('\s+', text)

## This method is highly recommend:
regex = re.compile('\s+')
regex.split(text)
# lsit of patterns: 

regex.findall(text)

# 
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

# re.IGNORECASE makes the regex case-insensitive
regex = re.compile(pattern, flags=re.IGNORECASE)

regex.findall(text)


###############
#Vectorized String Functions in pandas

