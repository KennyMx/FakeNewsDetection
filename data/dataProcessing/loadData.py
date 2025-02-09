import pandas as pd
import numpy as np

dfFake = pd.read_csv('data/Fake.csv')
dfTrue = pd.read_csv('data/True.csv')

dfFake['label'] = 1
dfTrue['label'] = 0

df= pd.concat([dfFake, dfTrue], ignore_index=True)


print("Data Set Head\n: ", df.head()) # 5 rows
print("Data Set Info\n: ", df.info())  # data types, missing values
print("Data Set Describe\n: ", df.describe()) # summary statistics
print("Data Set Shape\n: ", df.shape) # rows, columns
print("Data Set Columns\n: ", df.columns) # column names
print("Data Set Label\n: ", df['label'].value_counts()) # count of each label
print("Missing Values\n: ", df.isnull().sum()) # count of missing values


 # df.to_csv('data/combinedNews.csv', index=False)