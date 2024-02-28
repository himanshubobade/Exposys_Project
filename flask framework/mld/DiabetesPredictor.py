
# Importing essential libraries
import numpy as np
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('kaggle_diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))




'''
## New Algorithms
import numpy as np
import pandas as pd

df = pd.read_csv('kaggle_diabetes.csv')
df.head()

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df['Glucose'].fillna(round(df['Glucose'].mean(),2), inplace=True)
df['BloodPressure'].fillna(round(df['BloodPressure'].mean(),2), inplace=True)
df['SkinThickness'].fillna(round(df['SkinThickness'].median(),2), inplace=True)
df['Insulin'].fillna(round(df['Insulin'].median(),2), inplace=True)
df['BMI'].fillna(round(df['BMI'].median(),2), inplace=True)
df.head()

X = df.iloc[:, :8].values
y = df.iloc[:, -1].values

#####################################################
################ Logistic Regression ################
#####################################################

from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression

clf_lrs = LogisticRegression()
clf_lrs.fit(X,y)
#clf_lrs.coef_
#clf_lrs.intercept_
import statsmodels.api as sn
X_cons = sn.add_constant(X)
import statsmodels.discrete.discrete_model as sm
logit = sm.Logit(y,X_cons).fit()
logit.summary()

clf_lr = LogisticRegression()
clf_lr.fit(X,y)
clf_lr.coef_

X_cons = sn.add_constant(X)
logit = sm.Logit(y,X_cons).fit()

logit.summary()

clf_lr.predict_proba(X)
y_pred = clf_lr.predict(X)
y_pred

from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)
from sklearn.metrics import accuracy_score
print("Accuracy Score LR",accuracy_score(y, y_pred))


#####################################################
####################### LDA #########################
#####################################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf_lda = LinearDiscriminantAnalysis()
clf_lda.fit(X, y)
y_pred_lda = clf_lda.predict(X) 
confusion_matrix(y, y_pred_lda)
from sklearn.metrics import precision_score, recall_score
precision_score(y, y_pred_lda)
recall_score(y, y_pred_lda)
from sklearn.metrics import roc_auc_score
roc_auc_score(y, y_pred_lda)
print("Accuracy score LDA",accuracy_score(y, y_pred_lda))
'''
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#print (X_train.shape,X_test.shape,y_train.shape, y_test.shape)
clf_LR = LogisticRegression()
clf_LR.fit(X_train,y_train)

y_test_pred = clf_LR.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test, y_test_pred)
print("Accuracy score LR",accuracy_score(y_test, y_test_pred))
'''
'''

#####################################################
####################### KNN #########################
#####################################################
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s= scaler.transform(X_train)

scaler = preprocessing.StandardScaler().fit(X_test)
X_test_s= scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
clf_knn_1 = KNeighborsClassifier(n_neighbors=1)
clf_knn_1.fit(X_train_s, y_train)

confusion_matrix(y_test, clf_knn_1.predict(X_test_s))
accuracy_score(y_test, clf_knn_1.predict(X_test_s))

clf_knn_3 = KNeighborsClassifier(n_neighbors=3)
clf_knn_3.fit(X_train_s, y_train)
accuracy_score(y_test, clf_knn_3.predict(X_test_s))

from sklearn.model_selection import GridSearchCV
params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20,30]}
grid_search_cv = GridSearchCV(KNeighborsClassifier(), params)
grid_search_cv.fit(X_train_s, y_train)
optimised_KNN = grid_search_cv.best_estimator_
y_test_pred = optimised_KNN.predict(X_test_s) 
confusion_matrix(y_test, y_test_pred)
print("Accuracy KNN",accuracy_score(y_test, y_test_pred))

#####################################################
################### Random Forest ###################
#####################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=180)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy Random Forest:",metrics.accuracy_score(y_test, y_pred))

#####################################################
################### Decision Tree ###################
#####################################################

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy Decision tree:",metrics.accuracy_score(y_test, y_pred))

#####################################################
####################### SVM #########################
#####################################################

from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=109)

from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy for SVM:",metrics.accuracy_score(y_test, y_pred))
'''