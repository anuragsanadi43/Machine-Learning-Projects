# Credits: Sarvesh Kumar Sharma
# Github repo: https://github.com/shsarv/Machine-Learning-Projects/


#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/shsarv/ML-and-its-Application/blob/main/Loan_Repayment_Prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# <h1 align="center">Loan Repayment Prediction Using Ensemble Learning Methods</h1>
# <hr>
# 
# ### Objective:
# 
# * Predicts whether the bank should approves the loan of an applicant based on his profit using Ensemble Learning Methods.
# 
# **Submission by:-**
# 
# **Name:-** Sarvesh Kumar Sharma
# 
# **Section:-** A
# 
# **Roll no:-** ***50***
# 
# **Uni. Roll no:-** ***181500625***
# <hr>

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
kFold = StratifiedKFold(n_splits=5)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from  sklearn.metrics  import  accuracy_score , precision_score , recall_score,confusion_matrix,classification_report


# #### Reading file

# In[2]:


df = pd.read_csv("content\\loan_data.csv")
df.head()


# In[3]:


# Consise Summery

df.info()


# Here we can see that attribute purpose has object datatype. We need to deal with it.

# In[4]:


# Summery
df.describe()


# #### Checking For Null Values
# 

# In[5]:


df.isnull().sum().sum()


# Our DataFrame contain **Zero** Null values.
# <br><br>
# Now lets solve the problem with **Purpose** Attribute.

# In[6]:


# unique values in purpose attribute

df.purpose.value_counts()


# It has **6 unique values**. lets convert these labels into numeric form.

# ### Encoding
# 
# * We will be using **Label Encoder** to convert labels available in purpose attribute.
# 
# * It will Encode purpose labels with value between 0 and n_classes-1(5).

# In[7]:


df['purpose']=LabelEncoder().fit_transform(df['purpose'])
df.head()


# ## Data Visualization

# In[8]:


sns.set_style('darkgrid')
plt.hist(df['fico'].loc[df['credit.policy']==1], bins=30, label='Credit.Policy=1')
plt.hist(df['fico'].loc[df['credit.policy']==0], bins=30, label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# In[9]:


plt.figure(figsize=(10,6))
df[df['not.fully.paid']==1]['fico'].hist(bins=30, alpha=0.5, color='blue', label='not.fully.paid=1')
df[df['not.fully.paid']==0]['fico'].hist(bins=30, alpha=0.5, color='green', label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[10]:


#creating a countplot to see the counts of purpose of loans by not.fully.paid
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='purpose', hue='not.fully.paid')


# In[11]:


#checking the trend between FICO and the interest rate
plt.figure(figsize=(10,6))
sns.jointplot(x='fico', y='int.rate', data=df)


# In[12]:


#understanding the relationship between credit.policy and not.fully.paid
sns.lmplot(data=df, x='fico', y='int.rate', hue='credit.policy', col='not.fully.paid', palette='Set2')


# In[13]:


plt.figure(figsize = (20, 15)) 
sns.heatmap(df.corr(), cmap='BuPu', annot=True)
plt.show()


# We can see that **init rate, credit policy, fico and inq.last.6mths** has corresponding grater impact on target class(not.gully.paid)
# 
# 

# ### Train-Test Split
# 
# Splitting the dataset for training and testing purpose.

# In[14]:


# Dropping target class

X = df.drop('not.fully.paid',axis=1)
y = df['not.fully.paid']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)


# # Modellng

# ## Decision Tree

# In[16]:


from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
param_grid = {'max_depth': [2,3, 4,5,6,7,8,9,10,11,13,15,20]}

grid_search = GridSearchCV(dt_clf, param_grid, scoring = 'recall_weighted',cv=kFold, return_train_score=True)
grid_search.fit(X_train,y_train)


# In[17]:


grid_search.best_params_


# In[18]:


dt_clf = DecisionTreeClassifier(max_depth=2)
dt_clf.fit(X_train, y_train)
y_pred_train = dt_clf.predict(X_train)
y_pred_test = dt_clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)


# In[19]:


print("Confusion Matrix \n",confusion_matrix(y_test,y_pred_test))
print("\n")
print("<-------------------Classification Report---------------------->\n")
print(classification_report(y_test,y_pred_test))
print("\n")
print("<---------------Accuracy Scores------------------->\n")
print('Train Accuracy score: ',train_accuracy)
print('Test Accuracy score:',test_accuracy)


# We got **Accuracy of 84.58%** using Decision Tree Classifier.

# ### **Bagging** with Decision Tree

# In[20]:


from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
scaler=StandardScaler()
X_scaled = scaler.fit_transform(X)
bag_dt = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=2),n_estimators=100,bootstrap=True)
score = cross_val_score(estimator=bag_dt, X=X_scaled, y=y, scoring='recall_weighted', cv=kFold, n_jobs=-1)
print('Mean score:', score.mean())


# Bagging is not improving the score of model and giving only **73.10% of mean Score.**

# ### **AdaBoosting** with Decision Tree

# In[21]:


from sklearn.ensemble import AdaBoostClassifier

adaboost_clf = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth=2), learning_rate = 0.5)
adaboost_clf.fit(X_train, y_train)
print('Train score: {0:0.2f}'.format(adaboost_clf.score(X_train, y_train)))
print('Test score: {0:0.2f}'.format(adaboost_clf.score(X_test, y_test)))


# It giving the same result of **84%** and not improving our Model.

# ## Random Forest Classifier

# In[22]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=600)
rf_clf.fit(X_train, y_train)
y_pred_train = rf_clf.predict(X_train)
y_pred_test = rf_clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)


# In[23]:


print("Confusion Matrix \n",confusion_matrix(y_test,y_pred_test))
print("\n")
print("<-------------------Classification Report---------------------->\n")
print(classification_report(y_test,y_pred_test))
print("\n")
print("<---------------Accuracy Scores------------------->\n")
#print('Train Accuracy score: ',train_accuracy)
print('Test Accuracy score:',test_accuracy)


# We got the **Accuracy of 84.7%** with random Forest Classifier

# ### AdaBoosting with RandomForest

# In[24]:


from sklearn.ensemble import AdaBoostClassifier
adaboost_clf = AdaBoostClassifier(estimator = rf_clf, learning_rate = 0.5)
adaboost_clf.fit(X_train, y_train)
#print('Train score: {0:0.2f}'.format(adaboost_clf.score(X_train, y_train)))
#print('Test score: {0:0.2f}'.format(adaboost_clf.score(X_test, y_test)))
y_pred_train = adaboost_clf.predict(X_train)
y_pred_test = adaboost_clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)


# In[25]:


print("Confusion Matrix \n",confusion_matrix(y_test,y_pred_test))
print("\n")
print("<-------------------Classification Report---------------------->\n")
print(classification_report(y_test,y_pred_test))
print("\n")
print("<---------------Accuracy Scores------------------->\n")
#print('Train Accuracy score: ',train_accuracy)
print('Test Accuracy score:',test_accuracy)


# ### Gradient Boosting

# In[26]:


from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(learning_rate = 0.05)
gb_clf.fit(X_train, y_train)
#print('Train score: {0:0.2f}'.format(gb_clf.score(X_train, y_train)))
#print('Test score: {0:0.2f}'.format(gb_clf.score(X_test, y_test)))
y_pred_train = gb_clf.predict(X_train)
y_pred_test = gb_clf.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)


# In[27]:


print("Confusion Matrix \n",confusion_matrix(y_test,y_pred_test))
print("\n")
print("<-------------------Classification Report---------------------->\n")
print(classification_report(y_test,y_pred_test))
print("\n")
print("<---------------Accuracy Scores------------------->\n")
#print('Train Accuracy score: ',train_accuracy)
print('Test Accuracy score:',test_accuracy)


# In[28]:


# In[29]:


# While Computing different **Ensemble Learning Technologies**, We Found that Most of the **Bagging and Boosting** algo are giving similar result with minimum difference in accuracy. Even though in all these Ensembles-
# 
# We Found that the **Best Model for this DataSet is Random Forest with Accuracy of 85%.**

# <h3 align="center">Thank You</h3>

# In[ ]:




