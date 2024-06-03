#!/usr/bin/env python
# coding: utf-8

# # Numpy in Python
# 

# In[1]:


import numpy as np
arr = np.array([1,2,3,4,5,6,7,8,9,10])
print(arr)


# In[2]:


print(type(arr))


# In[3]:


print(np.__version__)


# In[4]:


arr = np.array((34))
print(arr)


# In[5]:


arr = np.array((1,2,3,4,5))
print(arr)


# In[6]:


arr = np.array([[1,2,3],[5,6,7]])
print(arr)


# In[7]:


array = np.array([[[1,2,3],[4,5,6]],[[5,6,7],[6,7,8]]])
print(array)


# In[8]:


a = np.array([1,2,3,4,5])
b = np.array([[1,2,3],[4,5,6]])
c = np.array([[[2,3,4],[2,2,2]],[[1,2,3],[6,7,8]]])
d = np.array(100)

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)


# In[9]:


import numpy as np
arr = np.array([1,2,3,4,5], ndmin = 6)
print(arr)
print("number of dimensions :" , arr.ndim)


# In[10]:


arr = np.array([1,2,3,4,5,6,7,8,9,10])
print(arr)
print("The element at first index is :",arr[0])
print("The element at second index is :",arr[1])
print("The element at third index is :",arr[2])
print("The element at fourth index is :",arr[3])
print("The element at fifth index is :",arr[4])
print("The element at sixth index is :",arr[5])


# In[11]:


#Accesing elements in 2-D array
arr = np.array([[1,2,3,4],[3,4,5,6]])
print(arr)
print("The element at [0,1]", arr[0,1])
print("The element at [0,1]", arr[1,1])
print("The element at [0,1]", arr[1,2])
print("The element at [0,1]", arr[1,3])


# In[12]:


arr = np.array([[1,2,3,4],[3,4,5,6]])
print(arr)
print(arr[1,-1])
print(arr[1:2])
print(arr[1:])
print(arr[:2])


# In[13]:


arr = np.array([1,2,3,4,5,6,7,8,9,10])

print(arr[1:5])
print(arr[1:])
print(arr[:2])


# In[14]:


arr = np.array([1,2,3,4])
print(arr.dtype)


# In[15]:


arr = np.array([0 , 1])
print(arr.dtype)


# In[16]:


arr = np.array(["apple","mango","cherry"])
print(arr.dtype)


# In[17]:


import numpy as np
arr = np.array([1.1,2.3,3.2,4.1,5.5]) # convert float dtype into integer dtype
newarr = arr.astype(int)
print(newarr)


# In[18]:


arr = np.array([1,0,1,0])
newarr = arr.astype(bool) # convert integer dtype to boolean
print(newarr)


# # Copy And View

# In[19]:


#make a copy, change the original array and display both the array using copy method
arr = np.array([1,2,3,4,5,6,7,8,9,10])
x = arr.copy()
arr[0] = 11
print(arr)
print(x) #Copy should not be affected with the changes in the original array


# In[20]:


#make a copy , change the original array and display both the arrays using view method
arr = np.array([1,2,3,4,5,6,7,8,9,10])
x = arr.view()
arr[0] = 32
print(arr)
print(x) # here original array is affected by the changes


# In[21]:


arr = np.array([1,2,3,4,5,6,7,8,9,10])

x = arr.copy() #returns None
y = arr.view() #returns the original array

print(x.base)
print(y.base)


# # Shape of an Array

# In[22]:


arr = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr.shape) # Here 2 represents the rows and 5 represents the column


# In[23]:


a = np.array([1,2,3,4,5,6] , ndmin = 7)
print(a)
print(a.shape)


# In[24]:


arr = np.array([1,2,3,4,5,6,7,8])
for x in arr :
    print(x)


# In[25]:


import numpy as np
arr = np.array([[1,2,3,4],[5,6,7,8]])
for x in arr :
    print(x)


# In[26]:


a = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
for x in a:
    print(x)


# In[27]:


arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  for y in x:
    for z in y:
      print(z) 
    


# In[28]:


arr = np.array([[1,2,3,4],[5,6,7,8]])
for x in np.nditer(arr):
    print(x)


# In[29]:


a1 = np.array([1,2,3])
b1 = np.array([4,5,6])
c = np.concatenate((a1 , b1))
print(c)


# In[30]:


a1 = np.array([[1,2,3],[1,2,3]])
b1 = np.array([[4,5,6],[7,8,9]])
c = np.concatenate((a1 , b1))
print(c)


# In[31]:


a1 = np.array([[1,2,3],[1,2,3]])
b1 = np.array([[4,5,6],[7,8,9]])
c = np.concatenate((a1 , b1) , axis=1)
print(c)


# In[32]:


a1 = np.array([[1,2,3],[1,2,3]])
b1 = np.array([[4,5,6],[7,8,9]])
c = np.concatenate((a1 , b1) , axis=0)
print(c)


# In[33]:


a1 = np.array([[1,2,3],[1,2,3]])
b1 = np.array([[4,5,6],[7,8,9]])
c = np.stack((a1 , b1) , axis=1)
print(c)


# In[34]:


#Stacking along the rows
a1 = np.array([[1,2,3],[1,2,3]])
b1 = np.array([[4,5,6],[7,8,9]])
c = np.hstack((a1 , b1))
print(c)


# In[35]:


#Stacking along the columns
a1 = np.array([[1,2,3],[1,2,3]])
b1 = np.array([[4,5,6],[7,8,9]])
c = np.vstack((a1 , b1))
print(c)


# In[36]:


#Stacking along the depth
a1 = np.array([[1,2,3],[1,2,3]])
b1 = np.array([[4,5,6],[7,8,9]])
c = np.dstack((a1 , b1))
print(c)


# # Splitting Array

# In[37]:


arr = np.array([1,2,3,4,5,6])
newarr = np.array_split(arr,3)
print(newarr)


# In[38]:


arr = np.array([1,2,3,4,5,6])
newarr = np.array_split(arr,4)
print(newarr)


# In[39]:


arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
newarr = np.array_split(arr, 3 )
print(newarr)


# In[40]:


arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
newarr = np.array_split(arr, 4 )
print(newarr)


# In[41]:


arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
newarr = np.array_split(arr, 3 , axis = 1 )
print(newarr)


# In[42]:


arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
newarr = np.hsplit(arr, 3 )
print(newarr)


# # Searching Arrays

# In[43]:


arr = np.array([1,2,3,4,5,67,8,9])
newarr = np.where(arr == 3)
print(newarr)


# In[44]:


arr = np.array([1,2,3,4,5,67,8,9])
newarr = np.searchsorted(arr, 3)
print(newarr)


# # Sorting Array

# In[45]:


arr = np.array([1,5,2,6,4])
newarr = np.sort(arr)
print(newarr)


# In[46]:


arr = np.array(["mango","cherry","blueberry","watermelon"])
B = np.sort(arr)
print(B)


# In[47]:


arr = np.array([True , False , True])
print(np.sort(arr))


# In[48]:


arr = np.array([[1,5,2],[6,4,7]])
newarr = np.sort(arr)
print(newarr)


# # Introduction To Random Numbers

# In[49]:


from numpy import random
x = random.rand()
print(x)


# In[50]:


x = random.randint(100 , size = (5))
print(x)


# In[51]:


x = random.randint(100 , size = (3 , 5))
print(x)


# In[52]:


x = random.randint(3,6 )
print(x)


# In[53]:


x = random.choice([1,2,3,4,5,6,7,8,9])
print(x)


# In[54]:


from numpy import random
x = random.choice([3,5,7,9] , p = [0.1, 0.2, 0.7, 0.0], size=(100))
print(x)


# In[55]:


from numpy import random
x = random.choice([1,2,3,4] , p = [0.1 , 0.4 , 0.5 , 0.0] , size = (3,5))
print(x)


# In[56]:


import numpy as np
from numpy import random

arr = np.array([1,2,3,4,5])
random.shuffle(arr)

print(arr)


# # Numpy ufunc

# In[57]:


#ufunc = universal functions
x = [1,2,3,4,5]
y = [6,7,8,9,10]
z = []

for i,j in zip(x,y):
    z.append(i + j)
print(z)


# In[58]:


import numpy as np
x = [1,2,3,4,5]
y = [6,7,8,9,10]
z = np.add(x,y)
print(z)

