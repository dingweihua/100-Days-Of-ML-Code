# Importing the libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('../datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding Categorical data
# https://stackoverflow.com/questions/54345667/onehotencoder-categorical-features-deprecated-how-to-transform-specific-column
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder="passthrough")
X = ct.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# regression evaluation
print(r2_score(Y_test, y_pred))
