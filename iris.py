from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

print(df.head())
print()
print("Shape:", df.shape)
print("Columns:", df.columns)
X = df.drop("species", axis=1)
y = df["species"]

print("X:")
print(X.head())

print("\ny:")
print(y.head())
from sklearn.model_selection import train_test_split

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

print("Model trained!")
pred = model.predict(X_test)

print(pred[:10])
from sklearn.metrics import accuracy_score
print("Calculating accuracy...")
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)
