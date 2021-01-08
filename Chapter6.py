
#####################
## CHAPTER 6: Data Loading, Storgae, and File Formats
#####################


import numpy as np

import pandas as pd

np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

# there is two ways to read data: 
# 1- read directly from director : 

df = pd.read_csv("D:/Python/Python for Data Analysis/examples/ex1.csv")

# 2. read when chane directory and .... 
# List contents in the current working directory
ls

# Navigate into the `data` sub-directory
cd D:\Python\Python for Data Analysis\examples

# List contents of `data`
 ls

# Print new working directory
pwd

df = pd.read_csv("ex1.csv")

df

# read table: 
table1=pd.read_table("ex1.csv", sep=',')


pd.read_csv("ex2.csv", header=None)

pd.read_csv("ex2.csv",names=['a','b','c','d', 'message'])

# use message as index column

names=['a','b','c','d', 'message']

pd.read_csv("ex2.csv",names=names, index_col='message')

# hierarchical index: 

parsed= pd.read_csv("csv_mindex.csv",index_col=['key1','key2'])


result=pd.read_table("ex3.txt",sep='\s+')

# skip some rows

pd.read_csv("ex4.csv", skiprows=[0,2,3])


pd.read_csv("ex5.csv")

pd.isnull(pd.read_csv("ex5.csv"))


#missing

pd.read_csv("ex5.csv",na_values=['NULL'])



# replace value with 'sentinels' function


sentinels={'message': ['foo','NA'], 'something' : ['two']}

pd.read_csv('ex5.csv', na_values=sentinels)


# read couple rows:

pd.read_csv("ex6.csv", nrows=5)


# chunksize: 

pd.read_csv('ex6.csv',chunksize=1000)



# writing data out to text format: 



# writ out data with to_csv  function 

data=pd.read_csv("ex5.csv")


data.to_csv("data.csv")

#  read excel file: 

xl=pd.ExcelFile("ex112.xlsx")
# use "parse" function to read sheet whihc has data. 
xl.parse('Sheet1')







