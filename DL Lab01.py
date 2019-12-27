#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import statistics as s
   
data = pd.read_csv("Training_Set.csv") 
colname = []
l = []
for i in data:
    colname.append(i)
#print(colname)
colname = colname[:-1]
for i in colname:
    l.append(data[i])
#print(l)

mn = []
#print(data.x1)
total = len(data.x1)

print("Mean Values :")
print("*************")
for i in l:
    #print(i)
    print(sum(i)/total)
    mn.append(sum(i)/total)

std = []
print()
print("Std Values :")
print("************")

for i in l:
    #print(i)
    print(s.stdev(i))
    std.append(s.stdev(i))

count = 0
for i in l:
    for j in range(total):
        #print(i.iloc[j])
        i.iloc[j] = (i.iloc[j] - mn[count])/(std[count])
        #print(i.iloc[j])
    count+=1

#data

alpha = 0.01

theta = [0,0,0,0,0,0]

for it in range(0,1000):
    cm = [0,0,0,0,0,0]
    for i in range(total):

        xi = [1]

        for j in range(0,5):
            xi.append(data.iloc[i,j])

        #print("XI :",xi)
        #print("Theta :",theta)
        c = 0
        diff = 0
        
        for j in range(len(xi)):
            c = c + xi[j]*theta[j]
        
        #print("c:",c)

        diff = (c - data.iloc[i,5]) 

        for j in range(len(xi)):
            cm[j] += diff*xi[j] 

        #print("CM:",cm)

    for i in range(0,6):

        theta[i] = theta[i] - ((alpha*cm[i])/total)

#new_theta = theta - ((alpha*cm)/total)
print()
print("Trained Model : ")
print("*****************")
print("Theta Values :")
print(theta)


# In[17]:


#Trained Model : 
#*****************
#Theta Values :
#[23.724782178952555, -0.6189386025654944, -2.488613838813864, -0.13462728894325218, -1.7504571477345958, -3.3334374718515862]
print(list(data.iloc[1,0:6]))
print(total)
print(len(data.x1))
print(data.iloc[1,5])


# In[3]:


a = [2,2,3]
b = [1,1,1]
c = 0
for i in range(len(a)):
    c  +=a[i]*b[i]
    
print(c)
print(theta)


# In[9]:


tdata = pd.read_csv("Testing_Set.csv") 

print("Predicted Values :")
print("******************")

n = len(tdata.x1)

for i in range(n):
    c = theta[0]
    for j in range(5):
        tdata.iloc[i,j] = (tdata.iloc[i,j] - mn[j])/std[j]
        c += tdata.iloc[i,j]*theta[j+1]
    print(c)


# In[12]:


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
tdata = pd.read_csv("Testing_Set.csv")
tdata


# In[14]:


import pandas as pd
import numpy as np
import statistics as s
   
data = pd.read_csv("Training_Set.csv") 
colname = []
l = []
for i in data:
    colname.append(i)
#print(colname)
colname = colname[:-1]
for i in colname:
    l.append(data[i])
#print(l)

mn = []
#print(data.x1)
total = len(data.x1)

print("Mean Values :")
print("*************")
for i in l:
    #print(i)
    print(sum(i)/total)
    mn.append(sum(i)/total)

std = []
print()
print("Std Values :")
print("************")

for i in l:
    #print(i)
    print(s.stdev(i))
    std.append(s.stdev(i))

count = 0
for i in l:
    for j in range(total):
        #print(i.iloc[j])
        i.iloc[j] = (i.iloc[j] - mn[count])/(std[count])
        #print(i.iloc[j])
    count+=1

#data

alpha = 0.01

theta = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for it in range(0,2000):
    
    cm = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(total):

        xi = [1]

        for j in range(0,5):
            xi.append(data.iloc[i,j])
        for j in range(0,5):
            for k in range(j,5):
                xi.append(data.iloc[i,k]*data.iloc[i,j])
            
        #print("XI :",xi)
        #print("Theta :",theta)
        c = 0
        diff = 0
        
        for j in range(len(xi)):
            c = c + xi[j]*theta[j]
        
        #print("c:",c)

        diff = (c - data.iloc[i,5]) 

        for j in range(len(xi)):
            cm[j] += diff*xi[j] 

        #print("CM:",cm)

    for i in range(0,21):

        theta[i] = theta[i] - ((alpha*cm[i])/total)

#new_theta = theta - ((alpha*cm)/total)
print()
print("Trained Model : ")
print("*****************")
print("Theta Values :")
print(theta)


# In[3]:


count = 1
for j in range(0,5):
    for k in range(j,5):
        print(j,k)


# In[8]:


print(len(theta))


# In[15]:


print(theta)


# In[19]:


tdata = pd.read_csv("Testing_Set.csv") 

print("Predicted Values :")
print("******************")

n = len(tdata.x1)

for i in range(n):
    c = 0
    x = [1]
    for j in range(5):
        tdata.iloc[i,j] = (tdata.iloc[i,j] - mn[j])/std[j]
        x.append(tdata.iloc[i,j])
    for j in range(0,5):
            for k in range(j,5):
                x.append(data.iloc[i,k]*data.iloc[i,j])
    for j in range(0,21):
        c += x[j]*theta[j]
    print(c)


# In[18]:


print(x)
print(len(x))


# In[ ]:


tdata = [1,12,8,307,130,3504]
c = 0
    x = [1]
    for j in range(5):
        tdata.[j] = (tdata.iloc[i,j] - mn[j])/std[j]
        x.append(tdata.iloc[i,j])
    for j in range(0,5):
            for k in range(j,5):
                x.append(data.iloc[i,k]*data.iloc[i,j])
    for j in range(0,21):
        c += x[j]*theta[j]
    print(c)

