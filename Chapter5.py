#####################
## CHAPTER 5: Getting Started with pandas
#####################

import pandas as pd
import numpy as np

from pandas import Series, DataFrame

obj = Series([4,7,-5,3])

obj
obj.values
obj.index

obj2 = Series([1,2,3,4], index=['d','s','a','g'])

obj2['d']

obj2[obj2>3]

'd' in obj2
'j' in obj2

" pass value" 

sdata = { 'Ohio': 23 , 'text': 22}
sdata 
Series(sdata)

' missing and null '

pd.isnull(sdata)
pd.notnull(sdata)

" Data Frame" 

obj=Series(range(3), index=['a','b','c'])

obj1=Series([2,3,2,1], index=['a','d','s','e'])

obj2=obj1.reindex(['w','e','r','t'])

" drop fucntion" 

obj=Series(np.arange(5),index=['a','c','d','e','r'])


new_obj=obj.drop('c')
"                 " 

" Selection, filtering "

obj=Series(np.arange(4),index=['a','b','c','e'])

obj
obj['b']

data= pd.DataFrame(np.arange(16).reshape((4, 4)),
                index=['ohio','Colorado','Utah','New York'],
                 columns=['one', 'two', 'three', 'four'])
data[:2]

data.ix['Colorado',['two','three']]
data.ix['Colorado',[3,1,2]]
data.ix[2]

"    " 

s1= Series([7,-2,3,1], index=['a','c','d','e'])

s2=Series([-2,3,1,4,5], index=['a','c','e','f','g'])

s1+s2

data1=pd.DataFrame(np.arange(9).reshape((3,3)),columns=list('abc'),
                index=['Ohio','Texas','Colorado'])

data2=pd.DataFrame(np.arange(12).reshape((4,3)),columns=list('abe'),
                index=['Ohio','Texas','Colorado','WI'])

data1+data2

data1.add(data2,fill_value=0)

"apply function:" 

frame= pd.DataFrame(np.random.rand(4,3), columns=list('bde'),
                     index=['Utah','OH','WI','IL'])

f=lambda x : x.max() - x.min()


frame.apply(f)
frame.apply(f,axis=1)

" now return max and min" 

def f(x): 
    return Series([x.min(),x.min()], index=['min', 'max'])

frame.apply(f)

frame.sort_index(axis=0)

frame.sort_index(by='e')


' Unique Value' 


obj11=Series(['a','a','c','d','d','c'])

uniq=obj11.unique()

obj11.value_counts()

' select subset of value: with isin' 

obj11.isin(['d','c'])
obj11[obj11.isin(['d','c'])]


string=Series(['majid','all',np.nan,'avocado'])
string

string.isnull()

string[0]=None

string

' handing missing' 

from numpy import nan as NA

' filtering out missing' 


data=Series([1, NA, 3, NA, 7])


data.dropna()

data[data.notnull()]

' missing with DataFrame'


data=DataFrame([[1,4,3],[1,NA,NA],[NA,NA,NA],[NA,3,2]])

data

data.dropna()

data.dropna(how='all')






