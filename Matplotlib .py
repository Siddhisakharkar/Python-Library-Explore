#!/usr/bin/env python
# coding: utf-8

# # Practice Of Matplotlib in Python

# In[ ]:


import matplotlib


# In[2]:


print(matplotlib.__version__)


# In[5]:


import matplotlib.pyplot as plt
import numpy as np


# In[7]:


xpoints = np.array([0,6])
ypoints = np.array([0,506])


# In[8]:


plt.plot(xpoints , ypoints)
plt.show()


# In[10]:


xpoints = np.array([0,10])
ypoints = np.array([0,410])

plt.plot(xpoints , ypoints)
plt.show()


# In[11]:


x_p = np.array([1 , 5])
y_p = np.array([1,5])

plt.plot(x_p , y_p , 'o')
plt.show()


# In[13]:


xp = np.array([1,2,3,6,8])
yp = np.array([3,4,3,4,9])

plt.plot(xp , yp)
plt.show()


# In[18]:


yp = np.array([1,2,3,7,8])

plt.plot(yp , marker = 'o')
plt.show()


# In[22]:


xp = np.array([1,3,6,7,9])
plt.plot(xp ,marker = 'o')
plt.show()


# In[33]:


xp = np.array([1,2,3,4,5])
yp = np.array([3,4,5,6,3])

#plt.plot(xp , yp , 'o:g')
#plt.plot(xp , yp , 'o-b')
plt.plot(xp , yp , 'o--r')
plt.show()


# In[40]:


xp = np.array([1,3,5,7])

plt.plot(xp , marker='o' , ms = 18 , mec = 'b' , mfc = 'r')
plt.show()


# In[49]:


xp = np.array([1,5,3,7,6])
yp = np.array([2,4,7,5,8])
plt.plot(xp , linestyle='--' , color = 'r' , linewidth = 5)
plt.plot(yp , linestyle='--' , color = 'b' , linewidth = 5, marker = 'o' , mfc = '#4CAF50')
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
xp = np.array([23,45,67,89,34,56,78,83])
yp = np.array([12,23,34,45,56,78,98,43])

plt.plot(xp , yp)
plt.xlabel("Average pulse")
plt.ylabel("calorie burnage")
plt.show()


# In[7]:


xp = np.array([11,22,33,44,55,66,77,88])
yp = np.array([23,45,13,78,83,41,81,33])

plt.plot(xp , yp)
plt.title("Sports watch data")
plt.xlabel("Average pulse")
plt.ylabel("Calorie burnage")
plt.show()


# In[11]:


xp = np.array([11,22,33,44,55,66,77,88])
yp = np.array([23,45,13,78,83,41,81,33])

plt.plot(xp,yp)

font1 = {'family':'serif','color':'Red'}

plt.title("Sports watch data" , fontdict= font1 , color='r', loc = 'left')
plt.xlabel("Average pulse" , fontdict=font1 , color='r')
plt.ylabel("Calorie burnage", fontdict=font1 , color='r')
plt.show()


# In[15]:


xp = np.array([10,20,30,40,50,60,70,80,90,100])
yp = np.array([23,45,13,72,83,41,81,33,10,80])

plt.plot(xp , yp)
plt.title("Sports watch data")
plt.xlabel("Average pulse")
plt.ylabel("Calorie burnage")

plt.grid(axis = 'x')

plt.show()


# In[17]:


xp = np.array([10,20,30,40,50,60,70,80,90,100])
yp = np.array([23,45,13,72,83,41,81,33,10,80])

plt.plot(xp , yp)
plt.grid(axis = 'y')


# In[20]:


xp = np.array([11,22,33,44,55,66,77,88])
yp = np.array([23,45,13,78,83,41,81,33])

plt.plot(xp , yp , color = 'Red', linestyle = '-' , linewidth='2')
plt.title("Sports watch data")
plt.xlabel("Average pulse")
plt.ylabel("Calorie burnage")

plt.grid(color = 'green', linestyle = '-' , linewidth='1')
plt.show()


# # Matplotlib Subplot

# In[24]:


#plot1
x = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])
plt.subplot(1,2,1)
plt.plot(x,y)

#plot2
x = np.array([1,2,3,4,5])
y = np.array([3,6,9,12,15])
plt.subplot(1,2,2)
plt.plot(x,y)

plt.show()


# In[36]:


x = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])
plt.subplot(2,3,1)
plt.plot(x , y)

x = np.array([1,2,3,4,5])
y = np.array([1,3,5,2,6])
plt.subplot(2,3,2)
plt.plot(x , y)

x = np.array([1,2,3,4,5])
y = np.array([3,4,5,6,7])
plt.subplot(2,3,3)
plt.plot(x , y)

x = np.array([1,2,3,4,5])
y = np.array([1,3,5,7,9])
plt.subplot(2,3,4)
plt.plot(x , y)

x = np.array([1,2,3,4,5])
y = np.array([9,8,7,6,5])
plt.subplot(2,3,5)
plt.plot(x , y)

x = np.array([1,2,3,4,5])
y = np.array([6,5,4,3,2])
plt.subplot(2,3,6)
plt.plot(x , y)

plt.suptitle("My Shop")


# In[40]:


xp = np.array([11,22,33,44,55,66,77,88])
yp = np.array([23,40,18,70,89,45,80,36])


plt.scatter(xp , yp)


# In[46]:


xp = np.array([1,2,3,4,5,6,7,8,9,10])
yp = np.array([32,40,36,35,12,10,9,22,45,50])
plt.scatter(xp , yp)

xp = np.array([11,12,13,14,15,16,17,18,19,20])
yp = np.array([22,45,12,11,35,40,42,49,50,33])
plt.scatter(xp , yp , color='red')

plt.show()


# # Colour Map

# In[55]:


xp = np.array([1,2,3,4,5,6,7,8,9,10])
yp = np.array([32,40,36,35,12,10,9,22,45,50])

colors = np.array([10,20,30,40,50,60,70,80,90,100])
sizes = np.array([200,250,300,350,400,450,500,550,600,650])
plt.scatter(xp , yp , c = colors , s = sizes , cmap = 'Accent_r')
plt.colorbar()
plt.show()


# In[66]:


xp = np.random.randint(100, size=(100))
yp = np.random.randint(100, size=(100))

colors = np.random.randint(100, size=(100))
sizes = 10 * np.random.randint(100, size=(100))

plt.scatter(xp , yp, alpha = 0.5 , s = sizes, cmap = 'nipy_spectral' , c = colors)
plt.colorbar()


# In[70]:


xp = np.random.randint(100, size=(20))
yp = np.random.randint(100, size=(20))

plt.bar(x , y , color='red')
plt.show()


# In[71]:


plt.barh(x , y , color='Green')


# In[77]:


xp = np.array([1,2,3,4,5,6,7,8,9,10])
yp = np.array([32,40,36,35,12,10,9,22,45,50])

plt.bar(xp , yp , width = 0.5 )


# In[82]:


xp = np.random.normal(170,10,250)


plt.hist(xp )


# In[85]:


xp = np.array([170,10,250 , 25 , 15])
plt.pie(y , startangle = 90)


# In[91]:


x= np.array([170,10,250 , 25 ])
myexplode = [0.2 , 0, 0 , 0]
plt.pie(x , startangle = 90 , explode = myexplode )


# In[ ]:




