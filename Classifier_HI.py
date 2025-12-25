"""
classification_pipeline_hi.py

Workflow:
1. Load nested-list dataset.
2. Fit all candidate distributions (still for comparison).
3. Force "power" distribution as the winner.
4. Label last 5 cycles of each unit as "unsafe" (1), rest as "safe" (0).
5. Build dataset: X = 15 sensors, y = [0/1].
6. Train classification models.
7. Evaluate with accuracy, precision, recall, f1, auc.
8. Generate confusion matrices.
"""

import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    xgb_available=True
except ImportError:
    xgb_available=False

try:
    from lightgbm import LGBMClassifier
    lgbm_available=True
except ImportError:
    lgbm_available=False


# ---------------- Load data ----------------
def load_preprocessed_nestedlist(path):
    with open(path, 'r') as f:
        data = json.load(f)
    units=[]
    for uid, unit_list in enumerate(data,start=1):
        sensors=np.array(unit_list,dtype=float)
        cycles=np.arange(1,sensors.shape[0]+1)
        units.append({'id':uid,'cycles':cycles,'sensors':sensors})
    return units


# ---------------- Label safe/unsafe ----------------
def build_classification_dataset(units):
    rows=[]
    for u in units:
        T=len(u['cycles'])
        for i,t in enumerate(u['cycles']):
            row={'unit':u['id'],'cycle':int(t)}
            for s in range(u['sensors'].shape[1]):
                row[f's{s+1}']=float(u['sensors'][i,s])
            row['label']=1 if i>=T-5 else 0  # last 5 cycles unsafe
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------- Plots ----------------
def plot_confusion_matrix(y_true,y_pred,model_name):
    cm=confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Safe","Unsafe"],yticklabels=["Safe","Unsafe"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs("plots/classification",exist_ok=True)
    plt.savefig(f"plots/classification/{model_name}_confusion.png"); plt.close()


# ---------------- Train & evaluate ----------------
def train_and_evaluate(df,target_col="label"):
    gsplit=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    train_idx,test_idx=next(gsplit.split(df,groups=df['unit']))
    train_df,test_df=df.iloc[train_idx],df.iloc[test_idx]

    X_cols=[c for c in df.columns if c.startswith('s')]
    X_train,X_test=train_df[X_cols].values,test_df[X_cols].values
    y_train,y_test=train_df[target_col].values,test_df[target_col].values

    models={
        "LogisticRegression":LogisticRegression(max_iter=1000),
        "RandomForest":RandomForestClassifier(n_estimators=200,n_jobs=-1),
        "ExtraTrees":ExtraTreesClassifier(n_estimators=200,n_jobs=-1),
        "GradientBoosting":GradientBoostingClassifier(n_estimators=200),
        "SVC":SVC(probability=True),
        "KNN":KNeighborsClassifier(n_neighbors=5),
        "MLP":MLPClassifier(hidden_layer_sizes=(64,32),max_iter=300)
    }
    if xgb_available:
        models["XGB"]=XGBClassifier(n_estimators=300,use_label_encoder=False,eval_metric="logloss")
    if lgbm_available:
        models["LGBM"]=LGBMClassifier(n_estimators=300)

    scaler=StandardScaler().fit(X_train)
    X_train_s,X_test_s=scaler.transform(X_train),scaler.transform(X_test)

    results=[]
    for name,model in models.items():
        print(f"Training {name}...")
        if name in ["SVC","MLP","KNN"]:
            model.fit(X_train_s,y_train); y_pred=model.predict(X_test_s); y_prob=model.predict_proba(X_test_s)[:,1]
        else:
            model.fit(X_train,y_train); y_pred=model.predict(X_test); y_prob=model.predict_proba(X_test)[:,1]
        acc=accuracy_score(y_test,y_pred)
        prec=precision_score(y_test,y_pred)
        rec=recall_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_prob)
        results.append({"model":name,"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"auc":auc})
        plot_confusion_matrix(y_test,y_pred,name)
    return pd.DataFrame(results)


# ---------------- Main ----------------
if __name__=="__main__":
    input_path="Data/train_FD001in.json"
    print("Loading data...")
    units=load_preprocessed_nestedlist(input_path)
    print(f"Loaded {len(units)} units.")

    # Build classification dataset
    df=build_classification_dataset(units)
    df.to_csv("ml_dataset_classification.csv",index=False)
    print("Saved ml_dataset_classification.csv")

    # Train classifiers
    results_df=train_and_evaluate(df)
    results_df.to_csv("ml_classification_results.csv",index=False)
    print("Classification Results:")
    print(results_df)
