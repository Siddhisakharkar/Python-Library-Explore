#!/usr/bin/env python
# coding: utf-8

# # Seaborn in Python
# 

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot([0,1,2,3,4,5,6])
plt.show()


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot([0,1,2,3,4,5,6], [10,20,30,40,50,60])
plt.show()


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot([1,2,3,4,5] , hist = False)
plt.show()


# # Normal Distribution (Guassian Distribution)

# In[4]:


#Normal Distribution consist of a three parameters as loc , size and scale
from numpy import random
arr = random.normal(size = (2,3))
print(arr)


# In[5]:


from numpy import random
arr = random.normal(loc = 1, scale = 2 ,size = (2,3))
print(arr)


# In[6]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(size = 100), hist=False)
plt.show()


# # Binomial Distribution
# 

# In[7]:


#Binomial distribution consist of a parameter n , p ,size
from numpy import random
import seaborn as sns

arr = random.binomial(n = 10 , p = 0.6 , size = 10)
print(arr)


# In[8]:


from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(random.binomial(n = 10 , p = 0.6 , size = 1000), hist = True , kde = False) #kde = kernel density estimate
plt.show()


# # Difference between Normal and Binomial Distribution

# In[9]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(loc = 1, scale = 2 , size = 100) , hist = False , label = 'normal')
sns.distplot(random.binomial(n = 10, p = 0.6 , size = 100) , hist = False , label = 'binomial')

plt.show()


# # Poisson Distribution

# In[10]:


from numpy import random
import seaborn as sns

arr = random.poisson(lam = 6 , size = 10)
print(arr)


# In[11]:


from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(random.poisson(lam = 6 , size = 1000), hist = True , kde = False) #kde = kernel density estimate
plt.show()


# # Difference between Normal and Poisson Distribution

# In[12]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(loc = 1, scale = 2 , size = 100) , hist = False , label = 'normal')
sns.distplot(random.poisson(lam = 6 , size = 100) , hist = False , label = 'poisson')

plt.show()


# # Uniform Distribution

# In[13]:


from numpy import random 
import seaborn as sns

arr = random.uniform(size=(2,3))
print(arr)


# In[14]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.uniform(size = 100), hist = False  , kde = True)
plt.show()


# # Logistic Distribution

# In[15]:


from numpy import random
import seaborn as sns

arr = random.logistic(loc = 10 , scale = 20 , size = (4,5))
print(arr)


# In[16]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.logistic(size = 10000), hist = False , kde = True)
plt.show()


# # Difference Between Logistic and Normal Distribution

# In[17]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(loc = 1, scale = 2 , size = 100) , hist = False , label = 'normal')
sns.distplot(random.logistic(size = 100) , hist = False , label = 'logistic')

plt.show()


# # Multinomial Distribution

# In[18]:


from numpy import random
import seaborn as sns

arr = random.multinomial(n = 7, pvals = [1/6,1/6,1/6,1/6,1/6,1/6,1/6])
print(arr)


# # Exponential Distribution

# In[19]:


from numpy import random
import seaborn as sns

arr = random.exponential(scale = 2 , size = [2,3])
print(arr)


# In[20]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.exponential(size = 10000), hist = False , kde = True)
plt.show()


# # chiSquare Distribution

# In[21]:


from numpy import random
import seaborn as sns

arr = random.chisquare(df = 2 , size = [2,3])
print(arr)


# In[22]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.chisquare(df = 1 , size = 10000), hist = False , kde = True)
plt.show()


# # Rayleigh Distribution

# In[23]:


from numpy import random
import seaborn as sns

arr = random.rayleigh(scale = 2 , size = [2,3])
print(arr)


# In[24]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.rayleigh(scale = 1 , size = 10000), hist = False , kde = True)
plt.show()


# # Pareto Distribution

# In[25]:


from numpy import random
import seaborn as sns

arr = random.pareto(a = 2 , size = [2,3])
print(arr)


# In[26]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.pareto(a = 50 , size = 10000), hist = False , kde = True)
plt.show()


# # ZipF Distribution

# In[27]:


from numpy import random
import seaborn as sns

arr = random.zipf(a = 2 , size = [2,3])
print(arr)


# In[29]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
x = random.zipf(a = 2 , size = [3,4])
sns.distplot(x[x< 12] , kde = False)

