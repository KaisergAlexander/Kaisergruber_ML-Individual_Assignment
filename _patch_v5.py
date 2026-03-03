"""
Patch v5: clean feature set + deep XGBoost search with subsample/colsample_bytree.

Changes:
  1. Cell 21 — revert NUMERICAL_FEATURES to clean set (remove redundant features)
  2. Cell 29 — update markdown
  3. Cell 30 — deep grid: n_estimators=[1000,1500], lr=[0.01,0.03], max_depth=[5,6],
               subsample=0.8, colsample_bytree=0.8 fixed
  4. Cell 31 — add subsample/colsample_bytree to final pipeline
"""

import json

NB_PATH = "70299_Complaints_Notebook.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

def src(cell):
    return "".join(cell["source"])

def set_src(cell, text):
    lines = text.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

print(f"Starting patch. Total cells: {len(cells)}")

# --------------------------------------------------------------------------
# 1. Cell 21 — clean NUMERICAL_FEATURES
# --------------------------------------------------------------------------
c21 = cells[21]
assert "NUMERICAL_FEATURES" in src(c21)

new21_src = """\
#Define column lists AFTER feature_engineering transforms X
#These must match exactly what feature_engineering() outputs.

NUMERICAL_FEATURES = [
    'response_time_days',
    'received_month',
    'received_year',
    'received_weekday',
    'has_narrative',
    'narrative_length',
    'zip_region',
    'is_older_american',
    'is_servicemember',
    'timely_response',
]

CATEGORICAL_FEATURES = [
    'Product',
    'Sub-product',
    'Issue',
    'Sub-issue',
    'Company public response',
    'Company',
    'State',
    'Consumer consent provided?',
    'Submitted via',
    'Company response to consumer',
]

print('Numerical features:', len(NUMERICAL_FEATURES))
print('Categorical features:', len(CATEGORICAL_FEATURES))\
"""
set_src(c21, new21_src)
assert "narrative_length" in src(c21)
assert "zip_region" in src(c21)
assert "narrative_length_log" not in src(c21)
assert "got_monetary_relief" not in src(c21)
print("[1] Cell 21 — NUMERICAL_FEATURES cleaned (redundant features removed)")

# --------------------------------------------------------------------------
# 2. Cell 29 — update markdown
# --------------------------------------------------------------------------
c29 = cells[29]
old29 = src(c29)
# Replace whatever feature description is in there
import re
new29 = re.sub(
    r'\*\*Feature engineering enhancements \(this branch\):\*\*.*?(?=\n\n\*\*Hyperparameters)',
    "**Feature engineering (this branch):**\n"
    "- `narrative_length`: character count of complaint text — captures consumer motivation\n"
    "  more precisely than a binary presence flag\n"
    "- `zip_region`: first digit of ZIP code (0–9 broad US region) — recovers geographic\n"
    "  signal from a column previously discarded entirely\n",
    old29,
    flags=re.DOTALL
)
set_src(c29, new29)
print("[2] Cell 29 — markdown updated")

# --------------------------------------------------------------------------
# 3. Cell 30 — deep XGBoost grid with subsample/colsample_bytree
# --------------------------------------------------------------------------
new_cell30 = """\
# XGBoost: deep hyperparameter search — many trees, small steps, stochastic sampling.
# subsample and colsample_bytree add controlled randomness (Lecture 6 — Boosting):
#   subsample: fraction of training rows used per tree (reduces variance)
#   colsample_bytree: fraction of features used per tree (reduces correlation between trees)

n_estimators_list  = [1000, 1500]
max_depth_list     = [5, 6]
learning_rate_list = [0.01, 0.03]

xgb_tuning_results = []

print('XGBoost — 5-fold CV F1 (deep search + stochastic sampling):')
print(f'{"n_est":>6}  {"depth":>5}  {"lr":>6}  {"Mean F1":>10}  {"Std F1":>8}')
print('-' * 48)

for n_est in n_estimators_list:
    for depth in max_depth_list:
        for lr in learning_rate_list:
            xgb_pipe = Pipeline([
                ('feature_engineering', feature_transformer),
                ('preprocessor', preprocessor),
                ('model', XGBClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    learning_rate=lr,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss',
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ))
            ])
            scores = cross_val_score(xgb_pipe, X_train, y_train, cv=5, scoring='f1')
            xgb_tuning_results.append({
                'n_estimators': n_est,
                'max_depth': depth,
                'learning_rate': lr,
                'mean_f1': scores.mean(),
                'std_f1': scores.std()
            })
            print(f'{n_est:>6}  {depth:>5}  {lr:>6.3f}  '
                  f'{scores.mean():>10.4f}  {scores.std():>8.4f}')

xgb_tuning_df = pd.DataFrame(xgb_tuning_results)
best_row   = xgb_tuning_df.loc[xgb_tuning_df['mean_f1'].idxmax()]
best_n_est = int(best_row['n_estimators'])
best_depth = int(best_row['max_depth'])
best_lr    = float(best_row['learning_rate'])
print(f'\\nBest: n_estimators={best_n_est}, max_depth={best_depth}, learning_rate={best_lr}')
print(f'Best CV F1: {best_row["mean_f1"]:.4f}')\
"""
set_src(cells[30], new_cell30)
print("[3] Cell 30 — deep grid: n_estimators=[1000,1500], lr=[0.01,0.03], subsample=0.8")

# --------------------------------------------------------------------------
# 4. Cell 31 — add subsample/colsample_bytree
# --------------------------------------------------------------------------
new_cell31 = """\
# Train best XGBoost on the full training set
xgb_best_pipeline = Pipeline([
    ('feature_engineering', feature_transformer),
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        n_estimators=best_n_est,
        max_depth=best_depth,
        learning_rate=best_lr,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

xgb_best_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_best_pipeline.predict(X_test)

xgb_f1  = f1_score(y_test, y_pred_xgb)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_pre = precision_score(y_test, y_pred_xgb)
xgb_rec = recall_score(y_test, y_pred_xgb)

print('=== XGBoost ===')
print(f'  F1 Score  : {xgb_f1:.4f}')
print(f'  Precision : {xgb_pre:.4f}')
print(f'  Recall    : {xgb_rec:.4f}')
print(f'  Accuracy  : {xgb_acc:.4f}')
print()
print(classification_report(y_test, y_pred_xgb, target_names=['Not Disputed', 'Disputed']))\
"""
set_src(cells[31], new_cell31)
print("[4] Cell 31 — subsample=0.8 and colsample_bytree=0.8 added")

# --------------------------------------------------------------------------
# Clear outputs and save
# --------------------------------------------------------------------------
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook saved. Total cells: {len(cells)}")
print("Done.")
