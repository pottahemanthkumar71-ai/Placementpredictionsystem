import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    r"C:\Users\cool\PycharmProjects\Placement prediction sysytem\data\placement_data.csv"
)

features = [
    "CGPA",
    "ATTENDANCEPERCENTAGE",
    "PROJECTS",
    "INTERNSHIPS",
    "CODING TEST SCORE"
]

X = df[features].dropna()

print("Selected features")
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

X["cluster"] = clusters

print("\nCluster assignments")
print(X.head(10))

centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(
    centers,
    columns=features
)

print("\nCluster centers")
print(centers_df)

print("\nStudents in each cluster")
print(X["cluster"].value_counts())

# Visualization
plt.scatter(
    X["CGPA"],
    X["CODING TEST SCORE"],
    c=X["cluster"],
    cmap="viridis",
    s=50
)

plt.xlabel("CGPA")
plt.ylabel("Coding Test Score")
plt.title("K-Means Clustering")
plt.show()