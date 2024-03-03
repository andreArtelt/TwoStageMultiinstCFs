import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_attrition_data():
    # Load and prepare data
    df = pd.read_csv ("data/Attrition.csv")

    # split dataset in features and target variable
    features_desc = ['DailyRate', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome',
                     'MonthlyRate', 'PerformanceRating', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

    feature_types = ["int", "int", "int", "int", "int", "real", "real", "int", "int", "int", "int"]

    X = df[features_desc].to_numpy() # Features

    y = df.Attrition.map(dict(Yes=1, No=0))
    y = pd.DataFrame(y).to_numpy()  # Target variable

    X = StandardScaler().fit_transform(X)

    feature_ranges = [(np.min(X[:, i]), np.max(X[:, i])) for i in range(X.shape[1])]
    feature_types = ["real" for _ in range(X.shape[1])]

    return X, y, np.zeros(y.shape), features_desc, feature_types, feature_ranges



# Note: Many .csv files in the data directory were downloaded from https://github.com/tailequy/fairness_dataset/tree/main/experiments/data
# Paper: https://arxiv.org/pdf/2110.00530.pdf



def load_lawSchool_dataset(use_gender_as_sensitive_attribute=True):
    # Load data
    df = pd.read_csv("data/law_school_clean.csv")
    
    # Extract label and sensitive attribute
    y = df["pass_bar"].to_numpy().flatten().astype(int)
    if use_gender_as_sensitive_attribute is True:
        y_sensitive = df["male"].to_numpy().flatten().astype(int)
    else:
        y_sensitive = df["race"]
        y_sensitive = (y_sensitive == "White").astype(int).to_numpy().flatten().astype(int)

    del df["pass_bar"]

    # Remove other columns and create final data set
    del df["male"]
    del df["race"]

    X = df.to_numpy()

    X = StandardScaler().fit_transform(X)
    features_desc = [f"f_{i}" for i in range(X.shape[1])]
    feature_ranges = [(np.min(X[:, i]), 10.*np.max(X[:,i])) for i in range(X.shape[1])]
    feature_types = ["real" for _ in range(X.shape[1])]
    return X, y, y_sensitive, features_desc, feature_types, feature_ranges


# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
def load_creditCardClients_dataset():
    # Load data
    df = pd.read_csv("data/credit-card-clients.csv")

    # Extract label and sensitive attribute (AGE could also be used as a sensitive attribute)
    y_sensitive = df["SEX"].to_numpy().flatten().astype(int) - 1  # Transform it to {0, 1}
    y = df["default payment"].to_numpy().flatten().astype(int)

    del df["SEX"]
    del df["default payment"]

    # Remove other "meaningless" columns and create final data set
    # [MARRIAGE, AGE]
    del df["MARRIAGE"]
    del df["AGE"]

    X = df.to_numpy()

    X = StandardScaler().fit_transform(X)
    features_desc = [f"f_{i}" for i in range(X.shape[1])]
    feature_ranges = [(np.min(X[:,i]), 10.*np.max(X[:,i])) for i in range(X.shape[1])]
    feature_types = ["real" for _ in range(X.shape[1])]
    return X, y, y_sensitive, features_desc, feature_types, feature_ranges
