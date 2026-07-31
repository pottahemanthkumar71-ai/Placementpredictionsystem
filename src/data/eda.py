from src.data.load_data import load_data
import matplotlib.pyplot as plt

def basic_ede(df):
    print(df.head())
    print("first 5 rows")
    print(df.head())
    print("last 5 rows")
    print(df.tail())
    print("print 25 to 35 rows")
    print(df.iloc[25:36])
    print(" print coloumn names")
    print(df.columns)
    print("datatypes")
    print(df.dtypes)
    print("Complete Information")
    print(df.info())
    print("no of null values")
    missing=df.isnull().sum()
    print(missing[missing > 0])
    print("no of duplicate records")
    print(df.duplicated().sum())
    print("Target variable status")
    count=df["PlacementStatus"].value_counts()
    plt.figure(figsize = (6,5))
    plt.bar(count.index,count.values)
    plt.title("Placment prediction")

    plt.xlabel("Placement Status")
    plt.ylabel("Number of Records")
    plt.savefig(r"C:\Users\scs87\PycharmProjects\Placementpredictsystem\results\placement_status_bar.png")
    plt.show()

if __name__ == "__main__":
    df=load_data()
    basic_ede(df)
