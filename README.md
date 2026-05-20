# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the customer dataset.
2. Select Annual Income and Spending Score as features for clustering.
3. Use the Elbow Method to determine the optimal number of clusters.
4. Apply the K-Means clustering algorithm to the dataset.
5. Train the model and predict cluster labels for customers.
6. Add the cluster labels to the dataset.
7. Visualize the customer clusters and centroids using a scatter plot.
8. Display the clustered data and stop the program.

## Program:
```

Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Iswarya S
RegisterNumber:  212225040135

# Implementation of K-Means Clustering for Customer Segmentation

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Step 2: Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Step 3: Display First 5 Rows
print(df.head())

# Step 4: Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 5: Find Optimal Number of Clusters using Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8,6))

plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.show()

# Step 6: Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)

# Step 7: Fit the Model
y_kmeans = kmeans.fit_predict(X)

# Step 8: Add Cluster Labels
df['Cluster'] = y_kmeans

# Step 9: Display Clustered Data
print(df.head())

# Step 10: Visualize Clusters
plt.figure(figsize=(10,7))

plt.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=y_kmeans
)

# Plot Centroids
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=200,
    marker='X'
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means Clustering")

plt.show()

```

## Output:
<img width="571" height="101" alt="Screenshot 2026-05-20 110214" src="https://github.com/user-attachments/assets/0128d300-919c-4cca-97f5-4e587992150b" />
<img width="787" height="520" alt="Screenshot 2026-05-20 110226" src="https://github.com/user-attachments/assets/1f231145-7c9a-495b-9014-02492dfa9c23" />
<img width="506" height="224" alt="Screenshot 2026-05-20 110243" src="https://github.com/user-attachments/assets/03c7f8b7-5019-4874-9a37-89ad9d2299b7" />
<img width="837" height="578" alt="Screenshot 2026-05-20 110359" src="https://github.com/user-attachments/assets/d770a9f7-60da-42cb-89ec-9508260af8de" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
