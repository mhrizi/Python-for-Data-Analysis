
#####################
## CHAPTER 4: NumPy Basics: Arrays and Vectorized Computation
#####################
import numpy as np

"Genrate random data"
data=np.random.randn(2,3)
"Show Data"
data
data*10
data + data

" Dimention of Data "
data.shape
data.dtype

""""""""""""""""""""""""""

" Creating ndarrays" 

"array function: to generate array"
" data1 is list"
data1= [1,2,3,4]
"np.array function:arr1 is array"
arr1= np.array(data1)
arr1

data2=[[1,2,3],[4,5,6]]

arr2=np.array(data2)

arr2
arr2.ndim

arr2.shape
" np.zero"

np.zeros(10) " return zero"
np.zeros((3,6)) " return zero"
np.empty((2,3,5)) " return unintializing values"
np.ones(4)
np.arange(6) "arrange data"

" astype function: change data types" 
arr3=np.array([2,3,1])
arr3.dtype
float_arr=arr3.astype(np.float64)

float_arr.dtype
"""""""""""""""""""""""""
arr4=np.array([[2,4,1],[3,6,1]])

arr4

arr4*arr4
arr4-arr4
arr4**2
""""""""""""""''

arr5=np.arange(10)
arr5

arr5[5]
arr5[5:8]=11
arr5

arr6=np.array([[1,2,4],[4,5,6],[5,4,3]])
arr6[0,2]

arr7=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

arr7

arr7[0]

old_value=arr7[0]

old_value

arr7[0,1]
" " 
arr8=np.array([1,2,3,4,5,6,7,8])

arr8[2:4]

" Boolean Index" 

names = np.array(['bob', 'Joe', 'Will', 'bob', 'Will', 'Joe', 'Joe'])

data=np.random.randn(7,4)

names == 'bob'
data[names == 'bob' ]

data[names == 'bob' , 2:]
" Not included"
data[names !='bob']
" | for OR , & for and"
mask = (names =='bob') | (names== 'Joe')
mask

data[data<0] = 0

data

'Fancy Indexing'
arr= np.empty((8,4))

arr

for i in range(8):
    arr[i]=i
    
arr

' now to get subset'

arr[[4,3,0,1]]

' if you want to have rows from the end:"

arr[[-2,-3,-4]]
 """"""""""""""""""" 
 ' if you want to have multipe code in one line: 
     ' for example: create array of 32 obser. and shaped in 8*4 array"
 arr=np.arange(32).reshape((8,4))

arr[[1,5,7,2],[0,3,2,1]]

' np.ix_ function: if we want to have row and change column: ' 

arr[np.ix_([1,5,7,2],[0,3,1,2])]

' Transposing'

arr=np.arange(15).reshape((3,5))

arr

' .T is used for transposing'
arr.T

" X'X "

x=np.random.randn(6,3)

"Use np.dot for transpos of two matrix"
np.dot(arr.T,arr)

" swapaxes function to transposing a pair of axis" 

arr=np.arange(16).reshape((2,2,4))
arr
arr.swapaxes(1,2)
'   Universal Function   ' 

arr=np.arange(10)
np.sqrt(arr)
np.exp(arr)
" max" 

x=np.random.randn(8)
y=np.random.randn(8)

" return, max for each element'
np.maximum(x,y)

" Data Processing" 

points=np.arange(-5,5,0.01)

points
'np.meshgrid will produce two 2D matrices'
xs, ys = np.meshgrid(points,points)

" for plot"
import matplotlib.pyplot as plt 

z= np.sqrt(xs**2+ys**2)
z

" plt.imshow fucntion for Plot" 
plt.imshow(z,cmap=plt.cm.gray); plt.colorbar()
plt.title("Image Plot")

" condition" 

xarr= np.array([1,2,3,4,5])
yarr=np.array([2,3,4,5,6])
cond=np.array([True, False, True, False, True])

result = [( x if c else y)
 for x, y, c in zip(xarr,yarr,cond)]
result

" fasetr: np.where" 

result= np.where(cond, xarr, yarr)

result

arr= np.random.randn(4,4,)

np.where( arr > 0, 2, -4)

np.where(arr>0, 2, arr)

" sum, mean, std" 

np.sum(np.random.rand(4,5))
np.mean(np.random.rand(4,5), axis=1)

arr=np.random.rand(100)

(arr>0).sum()

" sorting" 

arr=np.random.rand(6)
arr
arr.sort()
arr

" np.unique: which return the sorted unique values in an array"

names= np.array(['bob', 'joe', 'bob','joe','will'])
np.unique(names)


"    " 

arr=np.arange(10)
"np.save function, will saved in our computer"

np.save('some_array', arr)

" np.load function: will load array from drive"
np.load('some_array.npy')

" load data: np.loadtext" 

arr=np.loadtxt('historical_data1_Q11999.txt', delimiter='|')



" saved data: np.savetext" 