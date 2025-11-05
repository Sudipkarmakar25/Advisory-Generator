# train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import joblib

DATA = "sample_data.csv"
MODEL_OUT = "crop_model.joblib"
ENC_OUT = "encoders.joblib"

def load_data(path=DATA):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    cat_cols = ["crop_name","location","weather","soiltype"]
    enc = OrdinalEncoder()
    df_cat = enc.fit_transform(df[cat_cols])
    X_num = df[["temperature","humidity","rainfall"]].values
    X = np.hstack([df_cat, X_num])
    y = df["label"].values
    return X, y, enc

def main():
    df = load_data()
    X, y, enc = preprocess(df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    print("Train score:", clf.score(X_train, y_train))
    print("Val score:", clf.score(X_val, y_val))

    # save model and encoders
    joblib.dump(clf, MODEL_OUT)
    joblib.dump(enc, ENC_OUT)
    print("Saved model to", MODEL_OUT)

if __name__ == "__main__":
    main()
