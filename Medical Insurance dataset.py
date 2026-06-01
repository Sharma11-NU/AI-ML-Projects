""" 
Model type - Linear Regression 
2026-05-31
Kaggle Notebook 

"""

import pandas as pd
import seaborn as sns

# TIP: File path should be in "quotes" as in Python it is considered as a string.
insurance_data = pd.read_csv("/kaggle/input/datasets/mirichoi0218/insurance/insurance.csv")

insurance_data.head()
# Visualize
sns.scatterplot(x=insurance_data["bmi"], y=insurance_data["charges"], hue=insurance_data["smoker"])

"""
Encoding
yes = 1 and NO = 0
Female = 1 and Male = 0
x = another dataset with the required features 
y = Output column 
"""

X = insurance_data.drop(columns=["charges", "region"])
y = insurance_data["charges"]

X["sex"] = X["sex"].map({"female":1, "male":0})
X["smoker"] = X["smoker"].map({"yes":1, "no":0})

# Train the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42)

X_train.head()

# Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train) # .fit basically means to train model

# Now, let's predict the outcome
y_pred = model.predict(X_test)
y_pred

# Evaluate Model
"""
y_test = actual Value of the outcome
y_pred = predicted outcome by the model

y_test - y_pred = Mean squared error (MSE)
"""
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R squared value :",r2)

# Adjusted R-squared value

n = X.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1-((1-r2)*(n-1)/(n-p-1))
print("Adjusted r_square value:",adjusted_r2)
