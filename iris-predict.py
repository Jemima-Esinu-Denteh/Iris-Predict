import pandas as pd
from sklearn import datasets
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from IPython.display import display
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
pd.set_option('display.width', 1200)
pd.set_option('display.max_columns', 100)
print(pd.concat([iris_df.head(3), iris_df.tail(3)]))
print('targets: {}'.format(iris.target_names))
print(iris_df.describe())
print(iris_df['target'].value_counts())

X = iris_df.drop(['target'], axis=1) # number of features/attributes
y = iris_df['target'] # Ground truth or label or target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Classification using Logistic Regression
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg_orig = LogisticRegression()
logreg_orig.fit(X_train, y_train)
print('Training Score: {:.2f}'.format(logreg_orig.score(X_train, y_train)))
print('Test Score: {:.2f}'.format(logreg_orig.score(X_test, y_test)))

logreg_tuned = LogisticRegression()
logreg_tuned.fit(X_train, y_train)
y_pred = logreg_tuned.predict(X_test)
print('Test Accuracy: {:.3f}'.format(metrics.accuracy_score(y_pred, y_test)))

plt.figure()
y_test=y_test.to_numpy()
plt.plot(y_test,label="Actual")
plt.plot(y_pred,label="Predicted")
plt.tick_params(labelsize=16)
plt.legend(loc='best', prop={'size': 20})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.text(250, 1.1, r'1-',fontsize=20)
plt.legend(loc='best', prop={'size': 20})
plt.ylabel('Species: 1 - Setosa, 2- Versicolor, 3- Virginica',fontsize=16)
plt.xlabel('Number of Instances/Samples',fontsize=16)
plt.show()

param_grid = {'C': np.logspace(-5, 8, 15), 'max_iter': np.array(range(100, 1000, 100)), 'solver':('newton-cg', 'lbfgs', 'liblinear')}

# Instantiate a logistic regression classifier
logreg = LogisticRegression()

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg,param_grid , cv=5, random_state = 55)

# Fit it to the data
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print('Best cross-validation score: {:.2f}'.format(logreg_cv.best_score_))

# Logistic Regression on Best Set of Parameters
logreg_tuned = LogisticRegression(solver= 'lbfgs', max_iter = 900, C = 1389495.494373136,   random_state=55)
logreg_tuned.fit(X_train, y_train)
y_pred = logreg_tuned.predict(X_test)
print('Test Accuracy: {:.3f}'.format(metrics.accuracy_score(y_pred, y_test)))



