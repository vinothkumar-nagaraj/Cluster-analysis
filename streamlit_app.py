# Clustering Analysis Using Unsupervised Machine Learning in Streamlit

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import streamlit as st

# Title and Objective
st.title('Clustering Analysis Using Unsupervised Machine Learning')
st.write('## Objective:')
st.write('Explore and apply clustering techniques using the K-Means algorithm to group similar data points.')

# Dataset
st.write('#### Dataset Link:')
st.write('[Customer Segmentation Tutorial Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)')

# Load the dataset from GitHub
url = 'https://raw.githubusercontent.com/vinothkumar-nagaraj/Cluster-analysis/main/Mall_Customers.csv'
df = pd.read_csv(url)

# Display the first few rows of the dataset
st.write('### Data Exploration:')
st.write('#### First Few Rows of the Dataset')
st.dataframe(df.head())

# Summary statistics
st.write('#### Summary Statistics')
st.dataframe(df.describe())

# Check for missing values
st.write('#### Missing Values:')
st.write(df.isnull().sum())

# Data Visualization
st.write('### Visualize the Distribution of Key Variables')

# Age distribution
st.write('#### Age Distribution')
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30, color='blue')
plt.title('Distribution of Age')
st.pyplot(plt)

# Annual Income distribution
st.write('#### Annual Income Distribution')
plt.figure(figsize=(10, 6))
sns.histplot(df['Annual Income (k$)'], kde=True, bins=30, color='green')
plt.title('Distribution of Annual Income')
st.pyplot(plt)

# Spending Score distribution
st.write('#### Spending Score Distribution')
plt.figure(figsize=(10, 6))
sns.histplot(df['Spending Score (1-100)'], kde=True, bins=30, color='red')
plt.title('Distribution of Spending Score')
st.pyplot(plt)

# Data Processing
st.write('### Data Processing')
st.write('#### Handling Missing Values:')
df.fillna("", inplace=True)
st.write('No missing values were found.')

# Normalization
scaler = StandardScaler()
df[['Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
st.write('Data normalization done using StandardScaler.')

# K-Means Clustering
st.write('### K-Means Clustering')
st.write('#### Elbow Method to Determine Optimal Number of Clusters')

# Elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    wcss.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
st.pyplot(plt)

# Applying K-Means to the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])

# Visualization of Clusters
st.write('#### Visualization of Clusters')
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', data=df)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
st.pyplot(plt)

# Findings
st.write('### Findings:')
st.write('The K-Means algorithm was used to segment customers based on their Annual Income and Spending Score.')
st.write('We used the Elbow method to determine the optimal number of clusters and plotted the clusters using a scatterplot.')
st.write('Key processes before clustering included normalization and scaling using StandardScaler.')
