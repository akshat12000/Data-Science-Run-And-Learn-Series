import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Importing the dataset
dataset = pd.read_csv('income.csv')

plt.scatter(dataset['Age'], dataset['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(dataset[['Age', 'Income($)']])
dataset['cluster'] = y_predicted
dataset1 = dataset[dataset.cluster==0]
dataset2 = dataset[dataset.cluster==1]
dataset3 = dataset[dataset.cluster==2]
plt.scatter(dataset1.Age, dataset1['Income($)'], color='green')
plt.scatter(dataset2.Age, dataset2['Income($)'], color='red')
plt.scatter(dataset3.Age, dataset3['Income($)'], color='black')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

scaler = MinMaxScaler()
scaler.fit(dataset[['Income($)']])
dataset['Income($)'] = scaler.transform(dataset[['Income($)']])
scaler.fit(dataset[['Age']])
dataset['Age'] = scaler.transform(dataset[['Age']])

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(dataset[['Age', 'Income($)']])
dataset['cluster'] = y_predicted
dataset1 = dataset[dataset.cluster==0]
dataset2 = dataset[dataset.cluster==1]
dataset3 = dataset[dataset.cluster==2]
plt.scatter(dataset1.Age, dataset1['Income($)'], color='green')
plt.scatter(dataset2.Age, dataset2['Income($)'], color='red')
plt.scatter(dataset3.Age, dataset3['Income($)'], color='black')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='*', label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

k_rng = range(1, 10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(dataset[['Age', 'Income($)']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()