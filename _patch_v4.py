"""
Patch v4: company response binary flags + more trees with lower learning rate.

Changes:
  1. Cell 21 — add got_monetary_relief and got_any_relief to NUMERICAL_FEATURES
  2. Cell 29 — update markdown
  3. Cell 30 — new grid: n_estimators=[700,1000], lr=[0.03,0.05], max_depth=[5,7], gamma=0 fixed
  4. Cell 31 — remove gamma param (fixed at 0)
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
# 1. Cell 21 — add got_monetary_relief and got_any_relief
# --------------------------------------------------------------------------
c21 = cells[21]
assert "NUMERICAL_FEATURES" in src(c21)

new21 = src(c21).replace(
    "    'is_older_american',\n"
    "    'is_servicemember',\n"
    "    'timely_response',\n",
    "    'got_monetary_relief',\n"
    "    'got_any_relief',\n"
    "    'is_older_american',\n"
    "    'is_servicemember',\n"
    "    'timely_response',\n"
)
set_src(c21, new21)
assert "got_monetary_relief" in src(c21)
assert "got_any_relief" in src(c21)
print("[1] Cell 21 — got_monetary_relief and got_any_relief added to NUMERICAL_FEATURES")

# --------------------------------------------------------------------------
# 2. Cell 29 — update markdown
# --------------------------------------------------------------------------
c29 = cells[29]
new29 = src(c29).replace(
    "**Feature engineering enhancements (this branch):**\n"
    "- `narrative_length` + `narrative_length_log`: raw and log-scaled character count;\n"
    "  log compression handles the heavy right-skew of complaint text lengths\n"
    "- `response_time_log`: log-scaled days from receipt to company assignment\n"
    "- `is_weekend`: binary flag for Saturday/Sunday submission\n"
    "- `zip_region`: first digit of ZIP code (0–9 geographic region)",
    "**Feature engineering enhancements (this branch):**\n"
    "- `narrative_length` + `narrative_length_log`: raw and log-scaled character count\n"
    "- `response_time_log`: log-scaled response time\n"
    "- `is_weekend`: binary flag for weekend submission\n"
    "- `zip_region`: first digit of ZIP code (0–9 broad region)\n"
    "- `got_monetary_relief`: 1 if company provided monetary relief (strongest de-escalator)\n"
    "- `got_any_relief`: 1 if company provided any form of relief (monetary or non-monetary)"
)
set_src(c29, new29)
print("[2] Cell 29 — markdown updated")

# --------------------------------------------------------------------------
# 3. Cell 30 — new grid: more trees, lower LR, drop gamma loop (fix at 0)
# --------------------------------------------------------------------------
new_cell30 = """\
# XGBoost: hyperparameter tuning — more trees with lower learning rate.
# Smaller steps + more rounds is a classic XGBoost technique for squeezing
# out additional performance once the feature set is well-developed.

n_estimators_list  = [700, 1000]
max_depth_list     = [5, 7]
learning_rate_list = [0.03, 0.05]

xgb_tuning_results = []

print('XGBoost — 5-fold CV F1 (deep tree search):')
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
print("[3] Cell 30 — XGBoost grid updated: n_estimators=[700,1000], lr=[0.03,0.05]")

# --------------------------------------------------------------------------
# 4. Cell 31 — remove gamma parameter
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
print("[4] Cell 31 — gamma removed, grid params updated")

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
