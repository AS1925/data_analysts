import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Check columns
print(df.columns)

# Select features
X = df[['Annual_Income_k_USD', 'Spending_Score_1_100']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)

# Create cluster column
df['Cluster'] = kmeans.fit_predict(X)

# Scatter plot
plt.scatter(
    df['Annual_Income_k_USD'],
    df['Spending_Score_1_100'],
    c=df['Cluster']
)

# Labels
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

# Show graph
plt.show()
df.to_csv("segmented_customers.csv", index=False)