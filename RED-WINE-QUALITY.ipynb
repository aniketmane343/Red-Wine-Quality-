#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("D:\\new project\\RED WINE QUALITY.csv")
df.head(5)


# In[3]:


#Shape
print(df.shape)
print("----------------------------")
print(df.info())
print("----------------------------")
print(df.describe())


# In[4]:


df.isnull().sum()


# In[5]:


corr=df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# 1.
# Logistic Regression
# Definition: Logistic regression is a machine learning algorithm for classification. In this algorithm, the probabilities describing the possible outcomes of a single trial are modelled using a logistic function.
# 
# Advantages: Logistic regression is designed for this purpose (classification), and is most useful for understanding the influence of several independent variables on a single outcome variable.
# 
# Disadvantages: Works only when the predicted variable is binary, assumes all predictors are independent of each other and assumes data is free of missing values.

# 2.
# Naïve Bayes
# Definition: Naive Bayes algorithm based on Bayes’ theorem with the assumption of independence between every pair of features. Naive Bayes classifiers work well in many real-world situations such as document classification and spam filtering.
# 
# Advantages: This algorithm requires a small amount of training data to estimate the necessary parameters. Naive Bayes classifiers are extremely fast compared to more sophisticated methods.
# 
# Disadvantages: Naive Bayes is is known to be a bad estimator.

# 3.
# Stochastic Gradient Descent
# Definition: Stochastic gradient descent is a simple and very efficient approach to fit linear models. It is particularly useful when the number of samples is very large. It supports different loss functions and penalties for classification.
# 
# Advantages: Efficiency and ease of implementation.
# 
# Disadvantages: Requires a number of hyper-parameters and it is sensitive to feature scaling.

# 4.
# K-Nearest Neighbours
# Definition: Neighbours based classification is a type of lazy learning as it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the k nearest neighbours of each point.
# 
# Advantages: This algorithm is simple to implement, robust to noisy training data, and effective if training data is large.
# 
# Disadvantages: Need to determine the value of K and the computation cost is high as it needs to compute the distance of each instance to all the training samples.
# 

# 5.
# Decision Tree
# Definition: Given a data of attributes together with its classes, a decision tree produces a sequence of rules that can be used to classify the data.
# 
# Advantages: Decision Tree is simple to understand and visualise, requires little data preparation, and can handle both numerical and categorical data.
# 
# Disadvantages: Decision tree can create complex trees that do not generalise well, and decision trees can be unstable because small variations in the data might result in a completely different tree being generated.

# 6.
# Random Forest
# Definition: Random forest classifier is a meta-estimator that fits a number of decision trees on various sub-samples of datasets and uses average to improve the predictive accuracy of the model and controls over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement.
# 
# Advantages: Reduction in over-fitting and random forest classifier is more accurate than decision trees in most cases.
# 
# Disadvantages: Slow real time prediction, difficult to implement, and complex algorithm.

# 7.
# Support Vector Machine
# Definition: Support vector machine is a representation of the training data as points in space separated into categories by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.
# 
# Advantages: Effective in high dimensional spaces and uses a subset of training points in the decision function so it is also memory efficient.
# 
# Disadvantages: The algorithm does not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.

# Conclusion
# 
# 
# Comparison Matrix
# 
# 
# Accuracy: (True Positive + True Negative) / Total Population
# 
# 
# Accuracy is a ratio of correctly predicted observation to the total observations. Accuracy is the most intuitive performance measure.
# 
# 
# True Positive: The number of correct predictions that the occurrence is positive
# 
# 
# True Negative: The number of correct predictions that the occurrence is negative
# 
# 
# F1-Score: (2 x Precision x Recall) / (Precision + Recall)
# 
# 
# F1-Score is the weighted average of Precision and Recall used in all types of classification algorithms. Therefore, this score takes both false positives and false negatives into account. F1-Score is usually more useful than accuracy, especially if you have an uneven class distribution.
# 
# 
# Precision: When a positive value is predicted, how often is the prediction correct?
# 
# 
# Recall: When the actual value is positive, how often is the prediction correct?

# ## Visualisation Of Data

# In[6]:


sns.violinplot(x='quality',data=df)


# In[7]:


sns.pairplot(data=df,hue='quality',height=2.5)


# In[8]:


figure, ax = plt.subplots(1,5, figsize = (24,6))
sns.boxplot(data = df, x = "quality", y="fixed acidity", ax = ax[0])
sns.boxplot(data = df, x = "quality", y="volatile acidity", ax = ax[1])
sns.boxplot(data = df, x = "quality", y="citric acid", ax = ax[2])
sns.boxplot(data = df, x = "quality", y="residual sugar", ax = ax[3])
sns.boxplot(data = df, x = "quality", y="chlorides", ax = ax[4])
plt.show()


# In[9]:


figure, ax = plt.subplots(1,6, figsize = (24,6))
sns.boxplot(data = df, x = "quality", y="free sulfur dioxide", ax = ax[0])
sns.boxplot(data = df, x = "quality", y="total sulfur dioxide", ax = ax[1])
sns.boxplot(data = df, x = "quality", y="density", ax = ax[2])
sns.boxplot(data = df, x = "quality", y="pH", ax = ax[3])
sns.boxplot(data = df, x = "quality", y="sulphates", ax = ax[4])
sns.boxplot(data = df, x = "quality", y="alcohol", ax = ax[5])
plt.show()


# ## Data Transformation

# In[10]:


#label binarization
y = df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0 )


# In[11]:


x = df[df.columns[:-1]]
scaler = StandardScaler()
x = scaler.fit_transform(x)


# Splitting Data

# In[12]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[13]:


x_train.shape , x_test.shape , y_train.shape , y_test.shape


# # Model Training
# 
# We will train different model after the evaluation of model we will select out best model for production.
# 
# 
# Logistic Regression
# KNeighborsClassifier
# SVM Model
# Decision Tree
# Random Forest Regressor
# 
# 

# ### 1.Logistic Regression

# In[14]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter=700)
logistic_model.fit(x_train,y_train)


# In[22]:


y_pred_Logistic=logistic_model.predict(x_test)
y_pred_Logistic


# In[15]:


# accuracy score on training data

x_train_prediction = logistic_model.predict(x_train)
training_data_accuray = accuracy_score(x_train_prediction,y_train)
print('Accuracy of Logistic Regression model on training data  : ', training_data_accuray)

# accuracy score on testing data
x_test_prediction = logistic_model.predict(x_test)
logistic_test_data_accuray = accuracy_score(x_test_prediction,y_test)
print('Accuracy of Logistic Regression model on test data      : ', logistic_test_data_accuray)


# In[38]:


# confusion matrix
confusion_matrix(y_test, y_pred_Logistic)


# In[39]:


# classification report
print(classification_report(y_test, y_pred_Logistic))


# ### KNN Model

# In[32]:


from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors=31)
kfitModel = KNN_model.fit(x_train, y_train)
print(kfitModel)


# In[33]:


# finding optimal values for k
from sklearn.model_selection import cross_val_score
cross_valid_scores = []
for k in range(1, 50):
  knn = KNeighborsClassifier(n_neighbors = k)
  scores = cross_val_score(knn,x, y, cv = 10, scoring = 'accuracy')
  cross_valid_scores.append(scores.mean())    

print("Optimal k with cross-validation: \t",np.argmax(cross_valid_scores))
plt.plot(range(1,50),cross_valid_scores)


# In[34]:


kX_train_prediction = kfitModel.predict(x_train)
training_data_accuray = accuracy_score(kX_train_prediction,y_train)
print('Accuracy on training data  : ', training_data_accuray)

# accuracy score on testing data
kX_test_prediction = kfitModel.predict(x_test)
kx_lgr_test_data_accuray = accuracy_score(kX_test_prediction,y_test)
print('Accuracy on test data      : ', kx_lgr_test_data_accuray)


# In[35]:


y_pred_KNN=kfitModel.predict(x_test)
y_pred_KNN


# In[36]:


# confusion matrix
confusion_matrix(y_test, y_pred_KNN)


# In[40]:


# classification report
print(classification_report(y_test, y_pred_KNN))


# # SVM Model

# In[41]:


from sklearn import svm
classifier_model = svm.SVC(kernel='linear')
svm_model=classifier_model.fit(x_train,y_train)


# In[43]:


y_pred_SVM=svm_model.predict(x_test)
y_pred_SVM


# In[44]:


# accuracy score on training data

x_train_prediction = classifier_model.predict(x_train)
training_data_accuray = accuracy_score(x_train_prediction,y_train)
print('Accuracy of SVM model on training data : ', training_data_accuray)

# accuracy score on testing data

x_test_prediction = classifier_model.predict(x_test)
svm_test_data_accuray = accuracy_score(x_test_prediction,y_test)
print('Accuracy of SVM model on test data    : ', svm_test_data_accuray)


# In[47]:


# confusion matrix
confusion_matrix(y_test, y_pred_SVM)


# In[48]:


# classification report
print(classification_report(y_test, y_pred_SVM))


# ### DecisionTreeClassifier

# In[49]:


from sklearn.tree import DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier(random_state = 0)
dtc=decision_tree_model.fit(x_train,y_train)


# In[50]:


y_pred_dtc=dtc.predict(x_test)
y_pred_dtc


# In[51]:


from sklearn.model_selection import GridSearchCV

grid_params = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [3, 5, 7, 10],
    'min_samples_split' : range(2, 10, 1),
    'min_samples_leaf' : range(2, 10, 1)
}

grid_search = GridSearchCV(decision_tree_model, grid_params, cv = 5, n_jobs = -1, verbose = 1)
grid_search.fit(x_train, y_train)


# In[52]:


dtc = grid_search.best_estimator_
y_pred = dtc.predict(x_test)
dtc_train_acc = accuracy_score(y_train, dtc.predict(x_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training Accuracy of Decesion Tree Model  is {dtc_train_acc}")
print(f"Test Accuracy of Decesion Tree Model      is {dtc_test_acc}")


# In[53]:


from sklearn import tree
plt.figure(figsize=(25,15))
tree.plot_tree(dtc,filled=True)


# In[54]:


# confusion matrix
confusion_matrix(y_test, y_pred_dtc)


# In[55]:


# classification report
print(classification_report(y_test, y_pred_dtc))


# ### Random Forest model

# In[56]:


from sklearn.ensemble import RandomForestClassifier
modelRF = RandomForestClassifier()
RF_model=modelRF.fit(x_train,y_train)


# In[58]:


y_pred_RF=RF_model.predict(x_test)
y_pred_RF


# In[59]:


# accuracy on test data
x_test_prediction = modelRF.predict(x_test)
kr_test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy : ', kr_test_data_accuracy)


# In[61]:


# confusion matrix
confusion_matrix(y_test, y_pred_RF)


# In[63]:


# classification report
print(classification_report(y_test, y_pred_RF))


# ## Models Best Scores

# In[64]:


models = ['Logistic Regression','KNN','SVC', 'Decision Tree', 'Random Forest']
scores = [logistic_test_data_accuray,kx_lgr_test_data_accuray, svm_test_data_accuray, dtc_test_acc, kr_test_data_accuracy]
models = pd.DataFrame({'Model' : models, 'Score' : scores})
models


# In[65]:


plt.figure(figsize = (18, 8))
sns.barplot(x = 'Model', y = 'Score', data = models)
plt.show()


# In[ ]:




