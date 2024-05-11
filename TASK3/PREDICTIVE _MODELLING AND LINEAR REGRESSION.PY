## CodeAlpha
## Task 3:

## Predictive Modeling with Linear Regression:

## Problem Statement


## Build a predictive  model using linear regression predict a numerical outcome based on one or more features. 
## Use a dataset with continuous variable.
## Solution: 

The goal of the problem statement is to use linear regression to build a predictive model that can estimate the value of a numerical outcome variable based on one or more continuous independent variables. This involves analyzing a dataset, fitting a linear equation to the data, and using the model to make predictions about future or unknown observations.

## Importing the important libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
## Loading the data winequality dataset
df=pd.read_csv("winequality-red.csv")
## Displaying the data
df
## Number of Rows and columns in the dataset
df.shape

##  Getting the number of the data type of the columns
df.info()
## Getting the statistical insights from the dataset
df.describe()
## Checking the null values present in the dataset
df.isnull().sum()
## Visualizing the quality of wine with catplot
sns.catplot(x="quality", data=df, kind="count")
plt.show()
# Bar plot to show the quality of wine with citric acid quantity
sns.barplot(x="quality",y="citric acid",data=df)
plt.show()
## Defining the correlation
correlation=df.corr()
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
## feature selection of the data
X=df.drop('quality',axis=1)
Y=df['quality'].apply(lambda y_value: 1 if y_value>7 else 0)
Y
## Splitting into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
## Loading of the  logistic regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Making predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Coefficients and Intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
