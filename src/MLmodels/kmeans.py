import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\ADMIN\PycharmProjects\PlacementPredictionSystem\data\placement_data.csv")


features = [
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "CodingTestScore"
]

x = df[features].dropna()

print("Selected Features:")
print(x.head())

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(x_scaled)



x["Cluster"]=clusters

print("\nCluster Assignments:")
print(x.head(10))



centers_scaled = kmeans.cluster_centers_ppu


centers = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(
    centers,
    columns=features
)

print("\nCluster Centers:")
print(centers_df)



print("\nStudents in Each Cluster:")
print(x["Cluster"].value_counts().sort_index())


plt.figure(figsize=(8, 6))

plt.scatter(
    x["CGPA"],
    x["AttendancePercent"],
    c=x["Cluster"],
    cmap="viridis",
    s=50
)

plt.xlabel("CGPA")
plt.ylabel("Coding Test Score")
plt.title("K-Means Clustering of Students")

plt.show()