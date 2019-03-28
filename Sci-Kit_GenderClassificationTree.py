#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn import tree

#Training samples
#[height, weight, shoe size]
X = [
     [181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37], [166, 65, 40], [190, 90, 47], 
     [175, 64, 39], [177, 70, 40], [159, 55, 37], 
     [171, 75, 42], [181, 85, 43]]

#Class labels for the training samples
Y = [
    'male', 'male', 'female',
    'female', 'male', 'male',
    'female', 'female', 'female',
    'male', 'male']

clf = tree.DecisionTreeClassifier()

#.fit builds the decisionTreeClassifier from the training data X and Y
clf = clf.fit(X,Y)


# In[4]:


prediction = clf.predict([[190, 70, 43]])


# In[5]:


print(prediction)


# In[1]:


#build classifiers using:
#support vector machines, adaboost, MLP, 
#Quadratic Discriminant Analysis, K nearest neighbor, 
#Gaussian Process Classifier, and Random Forrest


# In[4]:


from sklearn.neighbors import KNeighborsClassifier


#Training samples
#[height, weight, shoe size]
X = [
     [181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37], [166, 65, 40], [190, 90, 47], 
     [175, 64, 39], [177, 70, 40], [159, 55, 37], 
     [171, 75, 42], [181, 85, 43]]

#Class labels for the training samples
Y = [
    'male', 'male', 'female',
    'female', 'male', 'male',
    'female', 'female', 'female',
    'male', 'male']

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, Y)

prediction = neigh.predict([[190, 70, 43]])

print(prediction)


# In[6]:


from sklearn.svm import SVC

#Training samples
#[height, weight, shoe size]
X = [
     [181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37], [166, 65, 40], [190, 90, 47], 
     [175, 64, 39], [177, 70, 40], [159, 55, 37], 
     [171, 75, 42], [181, 85, 43]]

#Class labels for the training samples
Y = [
    'male', 'male', 'female',
    'female', 'male', 'male',
    'female', 'female', 'female',
    'male', 'male']

clf = SVC(gamma='auto')
clf.fit(X, Y) 

prediction = clf.predict([[190, 70, 43]])

print(prediction)


# In[9]:


from sklearn.neural_network import MLPClassifier

#Training samples
#[height, weight, shoe size]
X = [
     [181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37], [166, 65, 40], [190, 90, 47], 
     [175, 64, 39], [177, 70, 40], [159, 55, 37], 
     [171, 75, 42], [181, 85, 43]]

#Class labels for the training samples
Y = [
    'male', 'male', 'female',
    'female', 'male', 'male',
    'female', 'female', 'female',
    'male', 'male']

clf = MLPClassifier(alpha=5)
clf.fit(X, Y) 

#interesting: for alpha=1 or alpha=3 answer came back female...

prediction = clf.predict([[190, 70, 43]])

print(prediction)


# In[10]:


from sklearn.ensemble import AdaBoostClassifier

#Training samples
#[height, weight, shoe size]
X = [
     [181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37], [166, 65, 40], [190, 90, 47], 
     [175, 64, 39], [177, 70, 40], [159, 55, 37], 
     [171, 75, 42], [181, 85, 43]]

#Class labels for the training samples
Y = [
    'male', 'male', 'female',
    'female', 'male', 'male',
    'female', 'female', 'female',
    'male', 'male']

clf = AdaBoostClassifier()
clf.fit(X, Y) 

prediction = clf.predict([[190, 70, 43]])

print(prediction)


# In[11]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#Training samples
#[height, weight, shoe size]
X = [
     [181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37], [166, 65, 40], [190, 90, 47], 
     [175, 64, 39], [177, 70, 40], [159, 55, 37], 
     [171, 75, 42], [181, 85, 43]]

#Class labels for the training samples
Y = [
    'male', 'male', 'female',
    'female', 'male', 'male',
    'female', 'female', 'female',
    'male', 'male']

clf = QuadraticDiscriminantAnalysis()
clf.fit(X, Y) 

prediction = clf.predict([[190, 70, 43]])

print(prediction)


# In[ ]:




