from prefect import flow, task
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import bentoml

@task(retries=5)
def load_data(data_path):
    # Load the data, sample such that the target classes are equal size
    df = pd.read_csv(data_path)
    df = pd.concat(
        [df[df.isFraud == 0].sample(n=len(df[df.isFraud == 1])), df[df.isFraud == 1]],
        axis=0,
    )
    return df

@task
def split_data(df, enc):
    # Select the features and target, and generate train/test split
    X = df[["ProductCD", "P_emaildomain", "R_emaildomain", "card4", "M1", "M2", "M3"]]
    X = pd.DataFrame(
        enc.transform(X).toarray(), columns=enc.get_feature_names_out().reshape(-1)
    )
    X["TransactionAmt"] = df[["TransactionAmt"]].to_numpy()
    y = df.isFraud

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

@task
def train_ohe(df):
    # Use one-hot encoding to encode the categorical features
    X = df[["ProductCD", "P_emaildomain", "R_emaildomain", "card4", "M1", "M2", "M3"]]
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(X)
    return enc

@task
def train_xgb(X_train, y_train):
    # Train the model
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        nthread=4,
        scale_pos_weight=1,
        seed=27,
    )
    model = xgb.fit(X_train, y_train)
    return model

@task
def save_model(model, enc, version, stage):
    # Save the model with BentoML
    saved_model = bentoml.sklearn.save_model(
        "fraud_classifier",
        model,
        labels={"owner": "Cerebrium", "stage": f"{stage}"},
        metadata={f"version": f"{version}"},
        custom_objects={"ohe_encoder": enc},
        signatures={
            "predict": {
                "batchable": True,
                "batch_dim": 0,
            }
        },
    )
    print(saved_model)


@flow
def train_flow(
    data_path, 
    version, 
    stage
):
    df = load_data(data_path)
    enc = train_ohe(df)
    X_train, _, y_train, _ = split_data(df, enc)
    model = train_xgb(X_train, y_train)
    save_model(model, enc, version, stage)

if __name__ == "__main__":
    train_flow(
        data_path="data/train_transaction.csv",
        version="1.0.1",
        stage="prod"
    )