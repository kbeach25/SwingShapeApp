import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

### Create folder for the models
os.makedirs("models", exist_ok = True)

df = pd.read_csv("data/ModelData.csv")

features = ['BatSpeed', 'AttackAngle', 'VBA', 'TTC', 'ReleaseSpeed', 
            'PitchTypeSpecific', 'PitcherHand', 'BatterHand', 
            'PitchZone', 'PitchType']

### Split features into numerical and categorical
num_cols = df[features].select_dtypes(include='number').columns
cat_cols = [c for c in features if c not in num_cols]

### Contact Model
# - column transformer
contact_prep = ColumnTransformer([
    ('num', 'passthrough', num_cols), 
    ('cat', OneHotEncoder(handle_unknown = 'ignore'), cat_cols)
    ])

# - keep nonempty cells
contact_train = df[df['Contact'].notna()]

# - train xgb model
contact_xgb_model = Pipeline([
    ('prep', contact_prep),
    ('xgb', XGBClassifier(
        n_estimators = 200,
        max_depth = 6,
        learning_rate = 0.05, 
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state = 42,
        n_jobs = -1,
        eval_metric = 'logloss'
    ))
])

# - fit 
contact_xgb_model.fit(contact_train[features], contact_train['Contact'])

# - save
joblib.dump(contact_xgb_model, 'models/contact_xgb_model.pkl')

### Hard Hit Model
# - only need balls in play
bip_description_vals = ['hit_into_play']
bip_df = df[df['Outcome'].isin(bip_description_vals)]

# - using the same features as the contact model, split into numerical and categorical
bip_num_cols = bip_df[features].select_dtypes(include = 'number').columns
bip_cat_cols = [c for c in features if c not in num_cols]

# - column transformer
hard_hit_prep = ColumnTransformer([
    ('num', 'passthrough', bip_num_cols),
    ('cat', OneHotEncoder(handle_unknown = 'ignore'), bip_cat_cols)
     ])

# - fit 
hard_hit_model = Pipeline([
    ('prep', hard_hit_prep),
    ('lr', LogisticRegression(max_iter = 1000))
])

# - save
joblib.dump(hard_hit_model, 'models/hard_hit_model.pkl')

### Batted Ball Profile Model
# - hit classification function
def hit_class(df):
    df = df.copy()

    # using mlb definitions to define batted ball types 
    conditions = [df['LaunchAngle'] < 10,
                  (df['LaunchAngle'] >= 10) & (df['LaunchAngle'] <= 25),
                  (df['LaunchAngle'] > 25) & (df['LaunchAngle'] <= 50),
                  df['LaunchAngle'] > 50]
    
    labels = ['isGB', 'isLD', 'isFB', 'isPU']

    for label, cond in zip(labels, conditions):
        df[label] = np.where(cond, 1, 0)

    return df


# - basically the same data as the hard hit model, but need LaunchAngle to always have a value
bb_profile_df = df[df['Outcome'].isin(bip_description_vals)]
bb_profile_df = bb_profile_df.dropna(subset = ['LaunchAngle'])

# - assign batted ball profiles to each row
complete_bb_df = hit_class(bb_profile_df)

# - derive single combined target column
complete_bb_df['BattedBallType'] = complete_bb_df[['isGB', 'isLD', 'isFB', 'isPU']].idxmax(axis = 1)

# - split into numerical and categorical
bb_num_cols = ['BatSpeed', 'AttackAngle', 'VBA', 'TTC', 'ReleaseSpeed']
bb_cat_cols = ['PitchTypeSpecific', 'PitcherHand', 'BatterHand', 'PitchZone', 'PitchType']

# - split into predictors and target
X = complete_bb_df[features]
y = complete_bb_df['BattedBallType']

# - split into train and test 
X_train, X_text, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

# - column transformer
bb_prep = ColumnTransformer([
    ('num', 'passthrough', bb_num_cols),
    ('cat', OneHotEncoder(handle_unknown = 'ignore'), bb_cat_cols)
])

# - model
bb_model = Pipeline([
    ('prep', bb_prep),
    ('lr', LogisticRegression(max_iter = 10000))
])

# - fit
bb_model.fit(X_train, y_train)

# - save
joblib.dump(bb_model, 'models/bb_model.pkl')