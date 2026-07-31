import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

### Path for saving everything
validation_path = Path(__file__).resolve().parent

### Access model directory and set models
contact_model_path = Path(__file__).resolve().parent.parent / "models" / "contact_xgb_model.pkl"
contact_model = joblib.load(contact_model_path)

hard_hit_model_path = Path(__file__).resolve().parent.parent / "models" / "hard_hit_model.pkl"
hard_hit_model = joblib.load(hard_hit_model_path)

bb_model_path = Path(__file__).resolve().parent.parent / "models" / "bb_model.pkl"
bb_model = joblib.load(bb_model_path)

### Access modeling data
model_data_path = Path(__file__).resolve().parent.parent / "data" / "ModelData.csv"
model_data = pd.read_csv(model_data_path)

### Features used for predictions
features = ['BatSpeed', 'AttackAngle', 'VBA', 'TTC', 'ReleaseSpeed', 'PitchTypeSpecific', 
            'PitcherHand', 'BatterHand', 'PitchZone', 'PitchType']

### Add predictions to the dataframe
model_data['pContact'] = contact_model.predict_proba(model_data[features])[:, 1]
model_data['pHardHit'] = hard_hit_model.predict_proba(model_data[features])[:, 1]

### Ball-in-play data for hard-hit rate and batted ball trajectory
bip_data = model_data[model_data["Outcome"] == "hit_into_play"].copy()

### Convert categorial batted ball representations to probabilities
bb_probs = bb_model.predict_proba(bip_data[features])

bb_prob_df = pd.DataFrame(
    bb_probs,
    columns = bb_model.named_steps['lr'].classes_,
    index = bip_data.index
).rename(columns = {
    "isGB": "pGB",
    "isLD": "pLD",
    "isFB": "pFB",
    "isPU": "pPU"
})

bip_data = bip_data.join(bb_prob_df)

### Function for assigning batted ball types
def hit_class(df):

    df = df.copy()

    # using mlb definitions
    conditions = ([df['LaunchAngle'] < 10,
        (df['LaunchAngle'] >= 10) & (df['LaunchAngle'] <= 25),
        (df['LaunchAngle'] > 25) & (df['LaunchAngle'] <= 50),
        df['LaunchAngle'] > 50])
    
    labels = ['isGB', 'isLD', 'isFB', 'isPU']
    
    for label, cond in zip(labels, conditions):
        df[label] = np.where(cond, 1, 0)

    return df

bip_data = hit_class(bip_data)

### Model accuracy function
def model_accuracy(df, bucket_size, target):

    data = df[df[target].notna()].copy()

    # Create variables for column access depending on target
    if target.startswith("is"):
        p_target = "p" + target[2:]
    else:
        p_target = "p" + target

    pred_rate = f'Predicted_{target}_Rate'
    true_rate = f'True_{target}_Rate'

    # Buckets are based on %s, so a bucket size of 0.1 means were are incrementing in steps of 10%
    bins = np.arange(0, 1 + bucket_size, bucket_size)

    data['Bucket'] = pd.cut(
        data[p_target],
        bins = bins,
        include_lowest = True,
        right = False
    )

    data_summary = (data.groupby('Bucket', observed = False)
                       .agg(Count = (target, 'size'),
                            **{pred_rate: (p_target, 'mean'),
                               true_rate: (target, 'mean')
                            }
                        )
    )

    # Weighted average of each bucket to determine the MAE
    data_summary['Err'] = abs(data_summary[pred_rate] - data_summary[true_rate])
    mae = round(
        (data_summary['Err'] * data_summary['Count']).sum() / data_summary['Count'].sum(), 4
    )

    data_summary = data_summary.round(4)
    data_summary["Mean_MAE"] = round(mae, 4)

    print(f'\n{target} Summary:')
    print(data_summary)
    print(mae)

    return data_summary

### Save each of the validation results
# - save contact output
with open(validation_path / "contact_results.csv", "w") as f:
    for bucket_size in [0.1, 0.05, 0.025]:
        f.write(f"Bucket Size, {bucket_size}\n")
        model_accuracy(model_data, bucket_size, "Contact").to_csv(f)
        f.write("\n")

with open(validation_path / "hard_hit_results.csv", "w") as f:
    for bucket_size in [0.1, 0.05, 0.025]:
        f.write(f"Bucket Size, {bucket_size}\n")
        model_accuracy(bip_data, bucket_size, "HardHit").to_csv(f)
        f.write("\n")

with open(validation_path / "batted_ball_results.csv", "w") as f:
    for bucket_size in [0.1, 0.05, 0.025]:
        f.write(f"Bucket Size, {bucket_size}\n\n")

        for each in ["isGB", "isLD", "isFB", "isPU"]:
            f.write(f"{each}\n")
            model_accuracy(bip_data, bucket_size, each).to_csv(f)
            f.write("\n")


### Encodings for pitch types
pitch_type_specific_labels = {
    0: "FF",
    1: "SI_FT",
    2: "FC",
    3: "SL",
    4: "CU_KC",
    5: "CH",
    6: "FS",
    7: "ST"
}

pitch_type_labels = {
    0: "FB_group",
    1: "Breaking_group",
    2: "Offspeed_group"
}

pitcher_hand_labels = {
    0: "RH_pitcher",
    1: "LH_pitcher"
}

batter_hand_labels = {
    0: "RH_batter",
    1: "LH_batter"
}

### SHAP plot function
def shap_plot(model, data, title, filename):
    sample_size = min(10000, len(data))
    X = data[features].sample(sample_size, random_state = 42)

    X_processed = model.named_steps["prep"].transform(X)
    feature_names = []

    for f in model.named_steps["prep"].get_feature_names_out():
        f = f.replace("num__", "").replace("cat__", "")

        if f.startswith("PitchTypeSpecific_"):
            code = int(float(f.split("_")[-1]))
            f = f"PitchTypeSpecific_{pitch_type_specific_labels[code]}"

        elif f.startswith("PitchType_"):
            code = int(float(f.split("_")[-1]))
            f = f"PitchType_{pitch_type_labels[code]}"

        elif f.startswith("PitcherHand_"):
            code = int(float(f.split("_")[-1]))
            f = pitcher_hand_labels[code]

        elif f.startswith("BatterHand_"):
            code = int(float(f.split("_")[-1]))
            f = batter_hand_labels[code]

        feature_names.append(f)


    estimator = model.steps[-1][1]

    if isinstance(estimator, LogisticRegression):

        masker = shap.maskers.Independent(
            X_processed,
            max_samples = sample_size
        )

        explainer = shap.Explainer(estimator, masker)
        shap_values = explainer(X_processed)

    else:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_processed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    plt.figure(figsize = (10, 8))

    if isinstance(estimator, LogisticRegression):

        class_names = {
            "isGB": "Ground Ball",
            "isLD": "Line Drive",
            "isFB": "Fly Ball",
            "isPU": "Pop Up"
        }

        legend_names = [
            class_names[c]
            for c in bb_model.named_steps["lr"].classes_
        ]

        shap.summary_plot(
            shap_values,
            feature_names = feature_names,
            class_names = legend_names,
            max_display = 15,
            show = False
        )

    else:
        shap.summary_plot(
            shap_values,
            X_processed,
            feature_names=feature_names,
            max_display=15,
            show=False
        )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(validation_path / filename, dpi = 300)
    plt.close()

# - save each SHAP figure
shap_plot(contact_model, model_data, "Contact Rate SHAP", "contact_shap.png")
shap_plot(hard_hit_model, bip_data, "Hard Hit Rate SHAP", "hard_hit_shap.png")
shap_plot(bb_model, bip_data, "Batted Ball Profile SHAP", "batted_ball_shap.png")