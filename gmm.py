import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib

### GMM Classification
# - need bat tracking data for this 
bat_tracking_df = pd.read_csv("data/SwingShapeData.csv")

# - only training on bat tracking data
gmm_train_features = ["bat_speed", "attack_angle", "vba", "ttc"]

# - there shouldn't be any missing data but just in case
X = bat_tracking_df[gmm_train_features].dropna()

# - scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# - find an appropriate amount of clusters
results = []
selection_text = []

for each in range(2, 9):

    gmm = GaussianMixture(
        n_components = each,
        covariance_type = "full",
        random_state = 42
    )

    gmm.fit(X_scaled)

    # assign clusters
    labels = gmm.predict(X_scaled)

    temp = bat_tracking_df.loc[X.index].copy()
    temp["Cluster"] = labels

    # cluster avg. summary
    cluster_summary = (
        temp.groupby("Cluster")[["bat_speed", "attack_angle", "vba", "ttc"]].mean().round(2)
    )

    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    # metrics
    results.append({
        "Clusters": each,
        "BIC": gmm.bic(X_scaled),
        "AIC": gmm.aic(X_scaled),
        "Sizes": cluster_sizes.tolist()
    })

    # text output
    selection_text.append(f"{each} Clusters")
    selection_text.append(cluster_summary.to_string())
    selection_text.append("")

results_df = pd.DataFrame(results)
selection_text.append("Model Selectioin Summary")
selection_text.append(results_df.to_string(index = False))

# report
with open("validation/gmm_model_selection.txt", "w") as f:
    f.write("\n".join(selection_text))

print("GMM model saved to validation/gmm_model_selection.txt")

# - fit gmm model
gmm = GaussianMixture(n_components = 4, random_state = 42)
gmm.fit(X_scaled)

# - predict cluster labs and append to dataframe
bat_tracking_df['GMM_Cluster'] = gmm.predict(scaler.transform(bat_tracking_df[gmm_train_features]))
bat_tracking_df['GMM_Cluster'] = bat_tracking_df['GMM_Cluster'].astype(str)

# - save cluster results and model
bat_tracking_df.to_csv("data/clusters.csv")
joblib.dump(gmm, 'models/gmm.joblib')
print("Saved cluster data to data/clusters.csv and GMM model to models/gmm.joblib")

# - save scaler for app
joblib.dump(scaler, "models/gmm_scaler.joblib")
print("Saved scaler to models/gmm_scaler.joblib")