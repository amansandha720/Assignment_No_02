

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataSet = pd.read_csv('Iris.csv')
dataSet.head()

# Data preprocessing
X = dataSet.drop('Species',axis=1)
Y = dataSet['Species']

# Calculate summary statistics
summary_stats = X.describe()
print('Summary Statistics: ')
print(summary_stats)

# showing all the graphs in plot
sns.set(style = 'ticks')
sns.pairplot(dataSet,hue = 'Species')
plt.show()

# Describe dataset

dataSet.describe()

#Dataset Information

dataSet.info()

# Histogram for sepal length

plt.figure(figsize=(10,7))
x = dataSet['SepalLengthCm']

plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")

# Histogram for sepal width

plt.figure(figsize=(10,7))
x = dataSet['SepalWidthCm']

plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")

# Histogram for Petal Length

plt.figure(figsize=(10,7))
x = dataSet['PetalLengthCm']

plt.hist(x, bins = 20, color = "green")
plt.title("Petal Length in cm")
plt.xlabel("Petal_Length_cm")
plt.ylabel("Count")

# Histogram for Petal Width

plt.figure(figsize=(10,7))
x = dataSet['PetalWidthCm']

plt.hist(x, bins = 20, color = "green")
plt.title("Petal Width in cm")
plt.xlabel("Petal_Width_cm")
plt.ylabel("Count")

# Removing id from column

new_data = dataSet[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
print(new_data.head())

# Box plot for total iris data

plt.figure(figsize=(10,7))
new_data.boxplot()

# Box plot for sepal Length data

plt.figure(figsize=(10,7))
new_data.boxplot(column='SepalLengthCm')

# Box plot for sepal Width data

plt.figure(figsize=(10,7))
new_data.boxplot(column='SepalWidthCm')