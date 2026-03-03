"""
Patch v3: log transforms + is_weekend + gamma tuning in XGBoost grid.

Changes:
  1. Cell 21 — add narrative_length_log, response_time_log, is_weekend to NUMERICAL_FEATURES
  2. Cell 29 — update markdown to describe the new features
  3. Cell 30 — add gamma=[0, 0.1] to grid, extend n_estimators=[500, 700]
  4. Cell 31 — add gamma=best_gamma to final pipeline
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
# 1. Cell 21 — add new features to NUMERICAL_FEATURES
# --------------------------------------------------------------------------
c21 = cells[21]
assert "NUMERICAL_FEATURES" in src(c21)

new21 = src(c21).replace(
    "    'narrative_length',\n"
    "    'zip_region',\n",
    "    'narrative_length',\n"
    "    'narrative_length_log',\n"
    "    'response_time_log',\n"
    "    'is_weekend',\n"
    "    'zip_region',\n"
)
set_src(c21, new21)
assert "narrative_length_log" in src(c21)
assert "response_time_log" in src(c21)
assert "is_weekend" in src(c21)
print("[1] Cell 21 — narrative_length_log, response_time_log, is_weekend added")

# --------------------------------------------------------------------------
# 2. Cell 29 — update markdown
# --------------------------------------------------------------------------
c29 = cells[29]
new29 = src(c29).replace(
    "**Feature engineering enhancements (this branch):**\n"
    "- `narrative_length`: character count of the complaint text (0 if none); captures\n"
    "  consumer motivation more precisely than a binary presence flag\n"
    "- `zip_region`: first digit of ZIP code (0–9 geographic region); recovers the\n"
    "  regional signal that was previously discarded entirely",
    "**Feature engineering enhancements (this branch):**\n"
    "- `narrative_length` + `narrative_length_log`: raw and log-scaled character count;\n"
    "  log compression handles the heavy right-skew of complaint text lengths\n"
    "- `response_time_log`: log-scaled days from receipt to company assignment\n"
    "- `is_weekend`: binary flag for Saturday/Sunday submission\n"
    "- `zip_region`: first digit of ZIP code (0–9 geographic region)"
)
set_src(c29, new29)
print("[2] Cell 29 — markdown updated")

# --------------------------------------------------------------------------
# 3. Cell 30 — XGBoost grid with gamma tuning
# --------------------------------------------------------------------------
new_cell30 = """\
# XGBoost: hyperparameter tuning over n_estimators, max_depth, learning_rate,
# and gamma (minimum loss reduction — tree complexity regularisation).

n_estimators_list = [500, 700]
max_depth_list    = [5, 7]
learning_rate_list = [0.05, 0.1]
gamma_list        = [0, 0.1]     # 0 = no regularisation; 0.1 = prune weak splits

xgb_tuning_results = []

print('XGBoost — 5-fold CV F1 (gamma regularisation tuning):')
print(f'{"n_est":>6}  {"depth":>5}  {"lr":>5}  {"gamma":>6}  {"Mean F1":>10}  {"Std F1":>8}')
print('-' * 58)

for n_est in n_estimators_list:
    for depth in max_depth_list:
        for lr in learning_rate_list:
            for gamma in gamma_list:
                xgb_pipe = Pipeline([
                    ('feature_engineering', feature_transformer),
                    ('preprocessor', preprocessor),
                    ('model', XGBClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        gamma=gamma,
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
                    'gamma': gamma,
                    'mean_f1': scores.mean(),
                    'std_f1': scores.std()
                })
                print(f'{n_est:>6}  {depth:>5}  {lr:>5.2f}  {gamma:>6.2f}  '
                      f'{scores.mean():>10.4f}  {scores.std():>8.4f}')

xgb_tuning_df = pd.DataFrame(xgb_tuning_results)
best_row   = xgb_tuning_df.loc[xgb_tuning_df['mean_f1'].idxmax()]
best_n_est = int(best_row['n_estimators'])
best_depth = int(best_row['max_depth'])
best_lr    = float(best_row['learning_rate'])
best_gamma = float(best_row['gamma'])
print(f'\\nBest: n_estimators={best_n_est}, max_depth={best_depth}, '
      f'learning_rate={best_lr}, gamma={best_gamma}')
print(f'Best CV F1: {best_row["mean_f1"]:.4f}')\
"""
set_src(cells[30], new_cell30)
print("[3] Cell 30 — XGBoost grid updated with gamma tuning and n_estimators up to 700")

# --------------------------------------------------------------------------
# 4. Cell 31 — add best_gamma to final pipeline
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
        gamma=best_gamma,
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
print("[4] Cell 31 — best_gamma added to final XGBoost pipeline")

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
print("Done — all patches applied.")
