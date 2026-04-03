import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data(path):
    data = pd.read_csv(path)
    data = data.drop("Loan_ID", axis=1)
    data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})
    data.ffill(inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    return data

def train_model(data):
    X = data.drop("Loan_Status", axis=1)
    y = data["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, X.columns