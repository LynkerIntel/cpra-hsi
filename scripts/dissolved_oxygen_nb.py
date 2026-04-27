import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay, partial_dependence

    import os
    from xgboost import XGBRegressor
    import numpy as np
    from sklearn.model_selection import ShuffleSplit, cross_validate, learning_curve


    return (
        PartialDependenceDisplay,
        ShuffleSplit,
        XGBRegressor,
        cross_validate,
        os,
        partial_dependence,
        pd,
        plt,
    )


@app.cell
def _(os):
    # =========================================================
    # PATHS
    # =========================================================
    ### Please update this to your folder where the same training data (year 2020) is located
    #train_file = r'D:\OMT\ML_Ali\ML_trainingtesting\Station_Data_Absolute_Analysis2020.xlsx'
    train_file = "/Users/dillonragar/Downloads/Station_Data_Absolute_Analysis2020.xlsx"

    # Here please change to folder where the new data (other locations) is located
    #newdata_file = r'D:\OMT\ML_Ali\ML_trainingtesting\Station_Data_Absolute_Analysis2022.xlsx'

    output_folder = "./ml_out"
    os.makedirs(output_folder, exist_ok=True)
    return (train_file,)


@app.cell
def _():
    # =========================================================
    # FEATURES (WITHOUT TRACER)
    # =========================================================
    features = [
        'Temp_C',
        'Depth_m',
        'Velocity_ms',
        'Month',
        'DayOfYear'
    ]
    return (features,)


@app.cell
def _(features, pd, train_file):
    # =========================================================
    # LOAD TRAINING DATA
    # =========================================================
    def load_station_excel(file_path, cols):
        xl = pd.ExcelFile(file_path)
        all_data = []

        for sheet in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)

            if all(c in df.columns for c in cols):
                df = df[cols].copy()
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df['Month'] = df['DateTime'].dt.month
                df['DayOfYear'] = df['DateTime'].dt.dayofyear
                df['Station'] = sheet
                all_data.append(df)

        df = pd.concat(all_data, ignore_index=True)
        return df.dropna().reset_index(drop=True)

    train_cols = [
        'DateTime',
        'Dissolved_Oxygen',
        'Temp_C',
        'Depth_m',
        'Velocity_ms'
    ]

    df_train = load_station_excel(train_file, train_cols)

    X_train = df_train[features]
    y_train = df_train['Dissolved_Oxygen']
    return X_train, y_train


@app.cell
def _(ShuffleSplit, XGBRegressor, X_train, cross_validate, y_train):
    # =========================================================
    # TRAIN XGBOOST MODEL (SAME SETTINGS)
    # =========================================================
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        reg_lambda=5,
        reg_alpha=1,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    cv_results = cross_validate(
        xgb, X_train, y_train,
        cv=cv,
        scoring={'r2': 'r2', 'rmse': 'neg_root_mean_squared_error'},
        return_train_score=True
    )

    xgb.fit(X_train, y_train)

    print(
        f"XGB Training CV → "
        f"R² = {cv_results['test_r2'].mean():.3f}, "
        f"RMSE = {-cv_results['test_rmse'].mean():.3f} mg/L"
    )
    return (xgb,)


@app.cell
def _(features, pd, xgb):
    # =========================================================
    # FEATURE IMPORTANCE
    # =========================================================
    booster = xgb.get_booster()
    importance_types = ["gain", "weight", "cover"]
    importance_df = pd.DataFrame({"feature": features})
    for imp_type in importance_types:
        scores = booster.get_score(importance_type=imp_type)
        importance_df[imp_type] = importance_df["feature"].map(scores).fillna(0.0)

    importance_df = importance_df.sort_values("gain", ascending=False).reset_index(drop=True)
    print("Feature importance (sorted by gain):")
    print(importance_df.to_string(index=False))
    return


@app.cell
def _(
    PartialDependenceDisplay,
    X_train,
    features,
    partial_dependence,
    plt,
    xgb,
):
    # =========================================================
    # PARTIAL DEPENDENCE — quantify DO swing per feature
    # =========================================================

    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 4), sharey=True)
    PartialDependenceDisplay.from_estimator(
        xgb, X_train, features=features, ax=axes, grid_resolution=50
    )
    fig.suptitle("Partial dependence of predicted DO on each feature")
    fig.tight_layout()

    # Per-feature DO swing (max - min of partial dependence curve)
    swings = {}
    for f in features:
        pd_result = partial_dependence(xgb, X_train, [f], grid_resolution=50)
        avg = pd_result["average"][0]
        swings[f] = float(avg.max() - avg.min())
    print("Predicted DO swing across each feature's range (mg/L):")
    for f, s in sorted(swings.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {f:12s} {s:.3f}")
    fig
    return


@app.cell
def _(os, xgb):
    model_path = os.path.join("ml_out", "xgb_dissolved_oxygen.json")
    xgb.save_model(model_path)
    print(f"Model saved to {model_path}")
    return


if __name__ == "__main__":
    app.run()
