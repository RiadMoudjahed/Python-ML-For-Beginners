import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

print ("="*10 + "Showing the first 5 rows" + "="*10)
df = pd.read_csv('housing.csv') # reading the csv file
print(df.head())
print("\n")

print ("="*10 + "Showing the info of the dataframe" + "="*10)
print(df.info()) # showing the info of the dataframe
print("\n")
print ("="*10 + "Showing the null values of the dataframe" + "="*10)
print(df.isnull().sum()) # showing the null values of the dataframe
print("\n")

print ("="*10 + "Filling the null values with the median" + "="*10)
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median()) # filling the null values with the median
print(df.isnull().sum()) # showing the null values of the dataframe
print("\n")

print ("="*10 + "Converting categorical values to numerical values" + "="*10)
df = pd.get_dummies(df, columns=["ocean_proximity"]) # converting categorical values to numerical values
print(df.head())
print("\n")

print ("="*10 + "Splitting the data into training and testing sets" + "="*10)
X = df.drop(columns=["median_house_value"]) # dropping the median_house_value column
y = df["median_house_value"] # creating the target variable
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\n")

print ("="*10 + "Splitting the data into training and testing sets" + "="*10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # splitting the data into training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("\n")

print ("\t" + "Training the model...") 
model = LinearRegression()
model.fit(X_train, y_train) # training the model
print("\n")

print ("="*10 + "Predicting the test data" + "="*10) 
y_pred = model.predict(X_test) # predicting the test data
print("y_pred shape:", y_pred.shape)
print("\n")

print ("="*10 + "Calculating the accuracy of the model" + "="*10) # calculating model performance metrics
print (f"R2 Score: {r2_score(y_test, y_pred)}") # displaying R2 score for linear regression
print (f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}") # displaying MSE for linear regression
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}") # calculating and displaying RMSE for linear regression
print("\n")

print ("="*10 + "Random Forest Regressor" + "="*10)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42) # initializing Random Forest model with 100 trees
model_rf.fit(X_train, y_train) # training Random Forest model on training data
y_pred_rf = model_rf.predict(X_test) # making predictions with Random Forest model
print("y_pred_rf shape:", y_pred_rf.shape)
print("\n")

print ("="*10 + "Random Forest Regressor Results" + "="*10)
print(f"R2 Score: {r2_score(y_test, y_pred_rf)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf))}")
print("\n")

print ("="*10 + "Plotting the results" + "="*10)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
# plotting actual vs predicted values for linear regression
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Random Forest)')
plt.show()