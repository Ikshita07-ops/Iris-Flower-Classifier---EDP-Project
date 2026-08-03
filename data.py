import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading data 
df = pd.read_csv("Iris.csv")

#printing dataframe
df1 = df.to_string()
df2 = df.info()

#Cleaning data
#Dropping ID column

df = df.drop("Id", axis=1)

# removing null values
df.dropna(inplace=True)
print(df.isnull().sum())

# Drop duplicates
print(df.drop_duplicates())

# Unique values and counts
print(df["Species"].unique())
print(df["Species"].value_counts())

# Feature selection
x = df.drop("Species", axis=1)

# Label selection
y = df["Species"]

print(x.head())
print(y.head())

# Loaded and cleaned the iris dataset ,and removed the unnecessary 
# column(ID column), removed null values 
# and duplicates, understood and performed feature and label selection.


