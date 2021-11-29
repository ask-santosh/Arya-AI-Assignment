#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix,log_loss


# In[5]:


# loading the train data
df_train = pd.read_csv('./Arya_DataScientist_Assignment/training_set.csv',index_col=0)
df_train.shape

#57 features and Y label is binary in nature


# In[21]:


correleation = df_train.corr()
plt.figure(figsize=(57,57))
sns.heatmap(correleation, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title("Correlation between different fitures")


# In[7]:


# X and Y distribution in the dataset(60/40)
df_train['Y'].value_counts(normalize=True)


# In[9]:


df_train.describe()


# In[13]:


X = df_train.drop(['Y'],axis=1)
y = df_train['Y']
X


# In[20]:


# Train and Test Split
# We want our train and validation set in the ratio of 4:1.
# Which means we would have 20% of the data as validation set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# In[18]:


# Using Random Forest Classifier for Feature Selection
clf = RandomForestClassifier(100, max_depth=None, n_jobs=-1)
clf.fit(X_train,y_train)
feature_importance = clf.feature_importances_


# In[19]:


x = [x for x in range(X_train.shape[1])]
plt.plot(x,feature_importance)
plt.xlabel("Features")
plt.ylabel("FI")
plt.title("Feature Importances")
plt.show()


# In[23]:


# Finding total features having greater feature importance than average
mean_fi = np.mean(feature_importance)
top_fi = len(np.where(feature_importance > mean_fi)[0])
print(f'Total features having feature importance greater than average are {top_fi}')


# In[26]:


# CDF Plot of Feature Importance
plt.plot(np.cumsum(sorted(feature_importance,reverse=True)))
plt.title('CDF of Feature Importance')
plt.xlabel('Features')
plt.ylabel('Feature Importance')


# In[27]:


''' From the above graph we conclude that there are approx. 30 important features which is very 
 near to one'''


# In[30]:


# Ranking the features with their respect to feature importances
fi = sorted(zip(X.columns,feature_importance),key=lambda x: x[1], reverse=True)
# Extracting Top 30 features
top_features = [x[0] for x in fi[:30]]
top_features


# In[33]:


# Selecting the top features from data
X_train_dash = X_train[top_features]
X_test_dash = X_test[top_features]
# Checking the shape of X_train_dash
X_train_dash.shape
X_test_dash.shape


# In[34]:


'''Normalizing our data using standard scaler'''
scaler = StandardScaler()
scaler.fit(X_train_dash)

# Transform the dataset
X_train_dash = pd.DataFrame(scaler.transform(X_train_dash),columns=X_train_dash.columns)
X_test_dash = pd.DataFrame(scaler.transform(X_test_dash),columns=X_test_dash.columns)


# In[35]:


'''Since we have binary values in our class labels, so we can choose Binary Cross Entropy here.
Also, we can use AUC score here.
We will be using different models for evaluation '''


# In[37]:


# 1. Random Model 

y_train_prob = np.random.rand(len(X_train_dash))
y_test_prob = np.random.rand(len(X_test_dash))
y_test_prob


# In[39]:


# Calculating logloss score for our model
print(f'Train Logloss for the model -> {log_loss(y_train,y_train_prob)}')
print(f'Test Logloss for the model -> {log_loss(y_test,y_test_prob)}')

print('='*55)

# Calculating the AUC score for our model
print(f'Train AUC Score for the model -> {roc_auc_score(y_train, y_train_prob)}')
print(f'Test AUC Score for the model -> {roc_auc_score(y_test, y_test_prob)}')


# In[41]:


# 2. K-Nearest Neighbour Classifier

estimator = KNeighborsClassifier()
parameters = {'n_neighbors':[3,5,11,15,25,51,75]}

# Performing GridSearchCV
clf = GridSearchCV(estimator, parameters, cv=10, n_jobs=-1, scoring='roc_auc', return_train_score=True)
clf.fit(X_train_dash, y_train)


# In[43]:


# Storing all the results of GridSearchCV in a DataFrame
results = pd.DataFrame.from_dict(clf.cv_results_)

x = list(results['param_n_neighbors'].values)
y1 = results['mean_train_score'].values
y2 = results['mean_test_score'].values

plt.plot(x, y1, label='Train')
plt.plot(x, y2, label='CV')
plt.legend()
plt.xlabel('NN {Hyperparameter}')
plt.ylabel('AUC Score')
plt.title('Performance of model on train & cv data');


# In[44]:


# Naive Bayes Classifier 

classifier = GaussianNB()
classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]


# Calculating logloss score for our model
print(f'Train Logloss for the model -> {log_loss(y_train,y_train_prob)}')
print(f'Test Logloss for the model -> {log_loss(y_test,y_test_prob)}')

print('='*55)
# Calculating the AUC score for our model
print(f'Train AUC Score for the model -> {roc_auc_score(y_train, y_train_prob)}')
print(f'Test AUC Score for the model -> {roc_auc_score(y_test, y_test_prob)}')


# In[45]:


# Logistic Regression model 

estimator = LogisticRegression(penalty='l2', max_iter=250, random_state=42)
parameters = {'C':[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]}

# Performing GridSearchCV
clf = GridSearchCV(estimator, parameters, cv=10, n_jobs=-1, scoring='neg_log_loss', return_train_score=True)
clf.fit(X_train_dash, y_train)


# In[46]:


results = pd.DataFrame.from_dict(clf.cv_results_)
#results.head()

x = list(map(str,results['param_C'].values))
y1 = results['mean_train_score'].values
y2 = results['mean_test_score'].values

plt.plot(x, y1, label='Train')
plt.plot(x, y2, label='CV')
plt.legend()
plt.xlabel('C {Hyperparameter}')
plt.ylabel('Logloss Score')
plt.title('Performance of model on train & cv data');


# In[47]:


# Using Logistic Regression with l2 norm

classifier = LogisticRegression(C=1, penalty='l2', max_iter=250, random_state=42)
classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]

# Calculating logloss score for our model
print(f'Train Logloss for the model -> {log_loss(y_train,y_train_prob)}')
print(f'Test Logloss for the model -> {log_loss(y_test,y_test_prob)}')

print('='*55)
# Calculating the AUC score for our model
print(f'Train AUC Score for the model -> {roc_auc_score(y_train, y_train_prob)}')
print(f'Test AUC Score for the model -> {roc_auc_score(y_test, y_test_prob)}')


# In[48]:


# Decision Tree 

classifier = DecisionTreeClassifier(criterion='gini',min_samples_split=3,random_state=42)
classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]

# Calculating logloss score for the model
print(f'Train Logloss for the model -> {log_loss(y_train,y_train_prob)}')
print(f'Test Logloss for the model -> {log_loss(y_test,y_test_prob)}')

print('='*55)
# Calculating the AUC score for the model
print(f'Train AUC Score for the model -> {roc_auc_score(y_train, y_train_prob)}')
print(f'Test AUC Score for the model -> {roc_auc_score(y_test, y_test_prob)}')


# In[49]:


# Random Forest Classifier

classifier = RandomForestClassifier(n_estimators=500,
                                      max_depth=None,
                                      min_samples_split=2,
                                      n_jobs=-1,
                                      class_weight='balanced',
                                      random_state=42)
classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]

# Calculating logloss score for our model
print(f'Train Logloss for the model -> {log_loss(y_train,y_train_prob)}')
print(f'Test Logloss for the model -> {log_loss(y_test,y_test_prob)}')

print('='*55)
# Calculating the AUC score for our model
print(f'Train AUC Score for the model -> {roc_auc_score(y_train, y_train_prob)}')
print(f'Test AUC Score for the model -> {roc_auc_score(y_test, y_test_prob)}')


# In[51]:


# XGBoost 

classifier = XGBClassifier(n_estimators=500,
                           max_depth=5,
                           learning_rate=0.15,
                           colsample_bytree=1,
                           subsample=1,
                           reg_alpha = 0.3,
                           gamma=10,
                           n_jobs=-1,
                           eval_metric='logloss',
                           use_label_encoder=False)

classifier.fit(X_train_dash, y_train)

y_train_pred = classifier.predict(X_train_dash)
y_train_prob = classifier.predict_proba(X_train_dash)[:,1]
y_test_pred = classifier.predict(X_test_dash)
y_test_prob = classifier.predict_proba(X_test_dash)[:,1]


# Calculating logloss score for our model
print(f'Train Logloss for the model -> {log_loss(y_train,y_train_prob)}')
print(f'Test Logloss for the model -> {log_loss(y_test,y_test_prob)}')

print('='*55)
# Calculating the AUC score for our model
print(f'Train AUC Score for the model -> {roc_auc_score(y_train, y_train_prob)}')
print(f'Test AUC Score for the model -> {roc_auc_score(y_test, y_test_prob)}')


# In[ ]:


'''
From the above experiment we conclude that:

The Xgboost have the least gap between the train and test logloss.
As we know the divergence between train and test metrics depicts overfitting or underfitting of a model.
Clearly in the Xgboost we don't have any overfitting here.
And Xgboost has least logloss for test data. i.e. Test logloss is 0.154 .
So, we can proceed with Xgboost as our classifier here.
'''

