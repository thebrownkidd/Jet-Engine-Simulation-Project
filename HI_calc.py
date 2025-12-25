"""
full_pipeline_hi_ml.py

- Fit multiple degradation distributions (with learnable start A and t0).
- Generate per-unit plots of distribution fits.
- User picks winning distribution in terminal.
- Compute HI for s2 and s5.
- Train multiple ML regressors with 15 sensors -> [HI2, HI5].
- Generate ML evaluation plots.

Outputs:
- distribution_scores.csv
- per_fit_records.csv
- plots/distributions/unitX_fits.png
- plots/distributions/rmse_boxplot.png
- ml_dataset_hi_s2_s5.csv
- ml_model_results.csv
- plots/ml/...
"""

import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# optional external libs
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

try:
    from lightgbm import LGBMRegressor
    lgbm_available = True
except ImportError:
    lgbm_available = False


# ---------------- Load data ----------------
def load_preprocessed_nestedlist(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if not (isinstance(data, list) and isinstance(data[0], list) and isinstance(data[0][0], list)):
        raise ValueError("Expected [unit][cycle][sensor_vector] format")
    units = []
    for uid, unit_list in enumerate(data, start=1):
        sensors = np.array(unit_list, dtype=float)
        cycles = np.arange(1, sensors.shape[0] + 1)
        units.append({'id': uid, 'cycles': cycles, 'sensors': sensors})
    return units


# ---------------- Degradation distributions (with A, t0) ----------------
def exp_model(t, A, t0, k):
    out = np.ones_like(t) * A
    mask = t >= t0
    out[mask] = A * np.exp(-k * (t[mask] - t0))
    return out

def stretched_model(t, A, t0, tau, beta):
    out = np.ones_like(t) * A
    mask = t >= t0
    x = (t[mask] - t0) / np.maximum(1e-8, tau)
    out[mask] = A * np.exp(-(x ** beta))
    return out

def power_model(t, A, t0, T, alpha):
    out = np.ones_like(t) * A
    mask = t >= t0
    denom = np.maximum(1e-8, T - t0)
    x = (t[mask] - t0) / denom
    x = np.clip(x, 0, 1)
    out[mask] = A * (1.0 - (x ** alpha))
    out[mask & (t >= T)] = 0.0
    return out

def gompertz_model(t, A, t0, a, b):
    out = np.ones_like(t) * A
    mask = t >= t0
    x = t[mask] - t0
    out[mask] = A * np.exp(-a * (np.exp(b * x) - 1.0))
    return out

def logistic_model(t, A, t0, c, d):
    out = np.ones_like(t) * A
    mask = t >= t0
    out[mask] = A / (1 + np.exp(c * (t[mask] - t0) - d))
    return out

def linear_model(t, A, t0, T):
    out = np.ones_like(t) * A
    mask = t >= t0
    out[mask] = A * (1 - (t[mask] - t0) / (T - t0 + 1e-8))
    out[mask & (t >= T)] = 0.0
    return out


# ---------------- Helpers ----------------
def sensor_to_hi_proxy(y_raw):
    T = len(y_raw)
    start_med = np.median(y_raw[:max(1,int(0.1*T))])
    end_med = np.median(y_raw[max(0,int(0.9*T)):])
    increasing = end_med > start_med
    y_min, y_max = np.min(y_raw), np.max(y_raw)
    if np.isclose(y_max, y_min):
        return np.ones_like(y_raw)
    if increasing:
        y_scaled = 1.0 - (y_raw - y_min) / (y_max - y_min)
    else:
        y_scaled = (y_raw - y_min) / (y_max - y_min)
    return np.clip(y_scaled, 0.0, 1.0)


def fit_model(t, y, model_name):
    if len(t) < 5: return None
    if len(y) >= 7:
        wl = min(9, len(y)//2*2+1)
        try: y_smooth = savgol_filter(y, wl, polyorder=2)
        except: y_smooth = y
    else: y_smooth = y
    y_smooth = np.clip(y_smooth, 1e-8, 1.0)

    A0 = y_smooth[0]
    t0_guess = float(t[int(0.1*len(t))])

    try:
        if model_name == 'exp':
            p0, bounds = [A0, t0_guess, 0.01], ([0, t[0], 1e-8],[2, t[-1], 10])
            popt,_ = curve_fit(lambda tt,A,t0,k:exp_model(tt,A,t0,k),t,y_smooth,p0=p0,bounds=bounds)
            y_hat = exp_model(t,*popt)
        elif model_name == 'stretched':
            p0, bounds = [A0, t0_guess, (t[-1]-t0_guess)/4,1.0], ([0,t[0],1e-8,0.1],[2,t[-1],(t[-1]-t0_guess)*10,5])
            popt,_ = curve_fit(lambda tt,A,t0,tau,beta:stretched_model(tt,A,t0,tau,beta),t,y_smooth,p0=p0,bounds=bounds)
            y_hat = stretched_model(t,*popt)
        elif model_name == 'power':
            T_guess=float(t[-1])
            p0, bounds = [A0,t0_guess,T_guess,1.0],([0,t[0],t0_guess+1e-6,0.1],[2,t[-1],t[-1],5])
            popt,_ = curve_fit(lambda tt,A,t0,T,alpha:power_model(tt,A,t0,T,alpha),t,y_smooth,p0=p0,bounds=bounds)
            y_hat = power_model(t,*popt)
        elif model_name == 'gompertz':
            p0, bounds = [A0,t0_guess,0.001,0.01],([0,t[0],1e-12,1e-6],[2,t[-1],10,1])
            popt,_ = curve_fit(lambda tt,A,t0,a,b:gompertz_model(tt,A,t0,a,b),t,y_smooth,p0=p0,bounds=bounds)
            y_hat = gompertz_model(t,*popt)
        elif model_name == 'logistic':
            p0, bounds = [A0,t0_guess,0.01,1],([0,t[0],1e-6,-5],[2,t[-1],5,5])
            popt,_ = curve_fit(lambda tt,A,t0,c,d:logistic_model(tt,A,t0,c,d),t,y_smooth,p0=p0,bounds=bounds)
            y_hat = logistic_model(t,*popt)
        elif model_name == 'linear':
            p0, bounds = [A0,t0_guess,t[-1]],([0,t[0],t0_guess+1e-6],[2,t[-1],t[-1]])
            popt,_ = curve_fit(lambda tt,A,t0,T:linear_model(tt,A,t0,T),t,y_smooth,p0=p0,bounds=bounds)
            y_hat = linear_model(t,*popt)
        else:
            return None
    except: return None

    rmse = np.sqrt(np.mean((y-y_hat)**2))
    return {'model': model_name,'params':popt,'y_hat':y_hat,'rmse':rmse}


# ---------------- Evaluation ----------------
def evaluate_distributions(units, candidate_models):
    records=[]
    for u in units:
        t=u['cycles']
        for s_idx in [1,4]:  # sensors 2 and 5
            y_scaled=sensor_to_hi_proxy(u['sensors'][:,s_idx])
            for m in candidate_models:
                res=fit_model(t,y_scaled,m)
                if res is None:
                    records.append({'unit':u['id'],'sensor':s_idx+1,'model':m,'rmse':np.nan,'params':None})
                else:
                    records.append({'unit':u['id'],'sensor':s_idx+1,'model':m,'rmse':res['rmse'],'params':res['params']})
        print(f"Fitted unit {u['id']}")
    df=pd.DataFrame(records)
    scores=df.groupby("model")["rmse"].mean().reset_index().sort_values("rmse")
    return scores,df


def plot_distribution_fits(units, per_fit_df, candidate_models):
    os.makedirs("plots/distributions",exist_ok=True)
    for u in units:
        t=u['cycles']
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for j,s_idx in enumerate([1,4]): # sensors 2,5
            y_scaled=sensor_to_hi_proxy(u['sensors'][:,s_idx])
            axes[j].plot(t,y_scaled,'k-',lw=2,label="Scaled proxy")
            for m in candidate_models:
                row=per_fit_df[(per_fit_df['unit']==u['id'])&(per_fit_df['sensor']==s_idx+1)&(per_fit_df['model']==m)]
                if row.empty or pd.isna(row.iloc[0]['rmse']): continue
                params=row.iloc[0]['params']
                if m=='exp': y_hat=exp_model(t,*params)
                elif m=='stretched': y_hat=stretched_model(t,*params)
                elif m=='power': y_hat=power_model(t,*params)
                elif m=='gompertz': y_hat=gompertz_model(t,*params)
                elif m=='logistic': y_hat=logistic_model(t,*params)
                elif m=='linear': y_hat=linear_model(t,*params)
                axes[j].plot(t,y_hat,'--',label=f"{m} (RMSE={row.iloc[0]['rmse']:.3f})")
            axes[j].set_title(f"Unit {u['id']} Sensor {s_idx+1}")
            axes[j].legend()
        plt.tight_layout()
        plt.savefig(f"plots/distributions/unit{u['id']}_fits.png"); plt.close()


def plot_rmse_boxplots(per_fit_df):
    plt.figure(figsize=(8,6))
    sns.boxplot(data=per_fit_df,x="model",y="rmse")
    plt.title("RMSE per distribution")
    plt.savefig("plots/distributions/rmse_boxplot.png"); plt.close()


# ---------------- HI computation ----------------
def compute_hi(units, model_name, per_fit_df):
    hi_targets={}
    for u in units:
        t=u['cycles']
        hi_targets[u['id']]={'cycles':t}
        for s_idx in [1,4]:
            row=per_fit_df[(per_fit_df['unit']==u['id'])&(per_fit_df['sensor']==s_idx+1)&(per_fit_df['model']==model_name)]
            if not row.empty and pd.notna(row.iloc[0]['rmse']):
                params=row.iloc[0]['params']
                if model_name=='exp': y_hat=exp_model(t,*params)
                elif model_name=='stretched': y_hat=stretched_model(t,*params)
                elif model_name=='power': y_hat=power_model(t,*params)
                elif model_name=='gompertz': y_hat=gompertz_model(t,*params)
                elif model_name=='logistic': y_hat=logistic_model(t,*params)
                elif model_name=='linear': y_hat=linear_model(t,*params)
            else:
                y_hat=np.linspace(1,0,len(t))
            hi_targets[u['id']][f'hi_s{s_idx+1}']=y_hat
    return hi_targets


def build_ml_dataset(units, hi_targets):
    rows=[]
    for u in units:
        for i,t in enumerate(u['cycles']):
            row={'unit':u['id'],'cycle':float(t)}
            for s in range(u['sensors'].shape[1]):
                row[f's{s+1}']=float(u['sensors'][i,s])
            row['hi_s2']=float(hi_targets[u['id']]['hi_s2'][i])
            row['hi_s5']=float(hi_targets[u['id']]['hi_s5'][i])
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------- ML training ----------------
def plot_true_vs_pred(y_true,y_pred,model_name,target_cols):
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for i,col in enumerate(target_cols):
        ax=axes[i]
        ax.scatter(y_true[:,i],y_pred[:,i],alpha=0.4,s=10)
        ax.plot([0,1],[0,1],'r--')
        ax.set_title(f"{model_name} - {col}")
    plt.tight_layout()
    os.makedirs("plots/ml",exist_ok=True)
    plt.savefig(f"plots/ml/{model_name}_true_vs_pred.png"); plt.close()

def plot_unit_timeseries(test_df,y_pred,model_name,target_cols):
    chosen_units=test_df['unit'].unique()[:3]
    for uid in chosen_units:
        df_u=test_df[test_df['unit']==uid].reset_index(drop=True)
        pred_u=y_pred[df_u.index]
        t=df_u['cycle'].values
        plt.figure(figsize=(10,5))
        for i,col in enumerate(target_cols):
            plt.plot(t,df_u[col].values,label=f"True {col}",lw=2)
            plt.plot(t,pred_u[:,i],'--',label=f"Pred {col}")
        plt.title(f"{model_name} - Unit {uid}")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"plots/ml/{model_name}_unit{uid}.png"); plt.close()

def train_and_evaluate(df,target_cols=['hi_s2','hi_s5']):
    gsplit=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    train_idx,test_idx=next(gsplit.split(df,groups=df['unit']))
    train_df,test_df=df.iloc[train_idx],df.iloc[test_idx]
    X_cols=[c for c in df.columns if c.startswith('s') and c not in target_cols]
    X_train,X_test=train_df[X_cols].values,test_df[X_cols].values
    y_train,y_test=train_df[target_cols].values,test_df[target_cols].values

    models={
        'Linear':MultiOutputRegressor(LinearRegression()),
        'Ridge':MultiOutputRegressor(Ridge()),
        'Lasso':MultiOutputRegressor(Lasso()),
        'RandomForest':MultiOutputRegressor(RandomForestRegressor(n_estimators=200,n_jobs=-1)),
        'ExtraTrees':MultiOutputRegressor(ExtraTreesRegressor(n_estimators=200,n_jobs=-1)),
        'GradientBoosting':MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200)),
        'SVR':MultiOutputRegressor(SVR(C=1.0)),
        'KNN':MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5)),
        'MLP':MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(64,32),max_iter=300))
    }
    if xgb_available:
        models['XGB']=MultiOutputRegressor(XGBRegressor(n_estimators=300))
    if lgbm_available:
        models['LGBM']=MultiOutputRegressor(LGBMRegressor(n_estimators=300))

    scaler=StandardScaler().fit(X_train)
    X_train_s,X_test_s=scaler.transform(X_train),scaler.transform(X_test)

    results=[]
    for name,model in models.items():
        print(f"Training {name}...")
        if name in ['SVR','MLP','KNN']:
            model.fit(X_train_s,y_train); y_pred=model.predict(X_test_s)
        else:
            model.fit(X_train,y_train); y_pred=model.predict(X_test)
        rmse=np.sqrt(mean_squared_error(y_test,y_pred))
        r2=r2_score(y_test,y_pred,multioutput='uniform_average')
        results.append({'model':name,'rmse':rmse,'r2':r2})
        plot_true_vs_pred(y_test,y_pred,name,target_cols)
        plot_unit_timeseries(test_df,y_pred,name,target_cols)
    return pd.DataFrame(results)


# ---------------- Main ----------------
if __name__=="__main__":
    input_path="Data/train_FD001in.json"
    print("Loading data...")
    units=load_preprocessed_nestedlist(input_path)

    candidate_models=['exp','stretched','power','gompertz','logistic','linear']
    scores,per_fit_df=evaluate_distributions(units,candidate_models)
    scores.to_csv("distribution_scores.csv",index=False)
    per_fit_df.to_csv("per_fit_records.csv",index=False)
    print(scores)

    plot_distribution_fits(units,per_fit_df,candidate_models)
    plot_rmse_boxplots(per_fit_df)

    # Ask user to choose distribution
    winner=input("Enter winning distribution (exp / stretched / power / gompertz / logistic / linear): ").strip()

    hi_targets=compute_hi(units,winner,per_fit_df)
    df=build_ml_dataset(units,hi_targets)
    df.to_csv("ml_dataset_hi_s2_s5.csv",index=False)

    results_df=train_and_evaluate(df)
    results_df.to_csv("ml_model_results.csv",index=False)
    print(results_df)
