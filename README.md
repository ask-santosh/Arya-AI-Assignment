# Arya-AI-Assignment

### Approach to the assignment

This is a Binary classification problem. Training data is 3910 and 57 features. Primary work is  feature selection, preprocessing of data and finally train machine learning algorithm to predict the correct class from the given test data set.

### Feature Selection

I have used mutual information as a measure of feature importance. Here in this problem I calculated mutual information of each feature with target variable to rank them.

### Preprocessing

Used RandomForest Classifier for feature selection. I can skip the scaling part . There slight class imbalance but not enough to worry about.


2. Selected top 30 features with respect to their feature importance.
3. The best model I get is Xgboost.

# Contents and files


