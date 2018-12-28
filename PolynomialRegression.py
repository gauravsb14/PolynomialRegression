import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures


# import the data
df = pd.read_csv("/home/gaurav/AI/Proiba_ML/Regression/RegressionData/50_Startups.csv")
print(df.head())


# labels the categorical variables with dummy variables as 0,1 or 2.
labelencoder_df = LabelEncoder()
df["State"] = labelencoder_df.fit_transform(df["State"])
print(df.head())


# Defining x,y
X = df.iloc[:,0:4]
Y = df["Profit"]


# changing input to polynomial input
polynomial_features= PolynomialFeatures(degree=2)
X_poly = polynomial_features.fit_transform(X)

# creating training and testing split
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size = .1, random_state = 1)

# creating model
model = LinearRegression()

# training model
model.fit(X_train, Y_train)

# prediction
pred = model.predict(X_test)
print("prdeicted value : {}".format(pred))