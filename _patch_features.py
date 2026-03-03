"""
Patch 70299_Complaints_Notebook.ipynb — Feature Engineering Enhancement strategy.

Changes:
  1. Cell 0   - remove 70299_Threshold.json row from submission table
  2. Cell 21  - add narrative_length and zip_region to NUMERICAL_FEATURES
  3. Cell 29  - update XGBoost markdown to describe new features
  4. Cell 30  - tune min_child_weight in [1, 3] (narrowed grid stays fast)
  5. Cell 31  - rename y_pred_xgb_default -> y_pred_xgb, remove threshold comment
  6. Cell 32  - update CM to use y_pred_xgb
  7. Delete cells 33, 34, 35 (threshold markdown + 2 code cells)
  8. Cell 38 (→35) - already uses xgb_f1; rename xgb_f1_default->xgb_f1 in cell 31 fixes this
  9. Cell 42 (→39) - remove threshold file saving
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
# 1. Cell 0 — remove Threshold.json row
# --------------------------------------------------------------------------
c0 = cells[0]
new0 = src(c0).replace(
    "| `70299_Threshold.json` | Optimal decision threshold for F1 maximisation |\n", ""
)
set_src(c0, new0)
assert "Threshold.json" not in src(c0)
print("[1] Cell 0 — Threshold.json row removed")

# --------------------------------------------------------------------------
# 2. Cell 21 — add narrative_length and zip_region to NUMERICAL_FEATURES
# --------------------------------------------------------------------------
c21 = cells[21]
assert "NUMERICAL_FEATURES" in src(c21), "Cell 21 is not the features cell"
new21 = src(c21).replace(
    "    'has_narrative',\n"
    "    'is_older_american',\n"
    "    'is_servicemember',\n"
    "    'timely_response',\n"
    "]",
    "    'has_narrative',\n"
    "    'narrative_length',\n"
    "    'zip_region',\n"
    "    'is_older_american',\n"
    "    'is_servicemember',\n"
    "    'timely_response',\n"
    "]"
)
set_src(c21, new21)
assert "narrative_length" in src(c21)
assert "zip_region" in src(c21)
print("[2] Cell 21 — narrative_length and zip_region added to NUMERICAL_FEATURES")

# --------------------------------------------------------------------------
# 3. Cell 29 — update XGBoost markdown
# --------------------------------------------------------------------------
c29 = cells[29]
new29 = src(c29).replace(
    "**Hyperparameters tuned via 5-fold cross-validation:**\n"
    "- `n_estimators`: number of boosting rounds\n"
    "- `max_depth`: maximum tree depth (limits overfitting)\n"
    "- `learning_rate`: shrinkage per round (lower = more robust, more trees needed)\n"
    "- `scale_pos_weight` fixed at `n_negative / n_positive` = 4.02",
    "**Feature engineering enhancements (this branch):**\n"
    "- `narrative_length`: character count of the complaint text (0 if none); captures\n"
    "  consumer motivation more precisely than a binary presence flag\n"
    "- `zip_region`: first digit of ZIP code (0–9 geographic region); recovers the\n"
    "  regional signal that was previously discarded entirely\n\n"
    "**Hyperparameters tuned via 5-fold cross-validation:**\n"
    "- `n_estimators`: number of boosting rounds\n"
    "- `max_depth`: maximum tree depth (limits overfitting)\n"
    "- `learning_rate`: shrinkage per round (lower = more robust, more trees needed)\n"
    "- `min_child_weight`: minimum sample weight per leaf (regularises against overfitting)\n"
    "- `scale_pos_weight` fixed at `n_negative / n_positive` ≈ 4.02"
)
set_src(c29, new29)
print("[3] Cell 29 — XGBoost markdown updated")

# --------------------------------------------------------------------------
# 4. Cell 30 — XGBoost grid with min_child_weight tuning
# --------------------------------------------------------------------------
new_cell30 = """\
# XGBoost: hyperparameter tuning over n_estimators, max_depth, learning_rate,
# and min_child_weight (regularisation — Lecture 6 Boosting).

n_estimators_list     = [300, 500]
max_depth_list        = [5, 7]
learning_rate_list    = [0.05, 0.1]
min_child_weight_list = [1, 3]   # higher = more conservative, reduces overfitting

xgb_tuning_results = []

print('XGBoost — 5-fold CV F1 (min_child_weight tuning):')
print(f'{"n_est":>6}  {"depth":>5}  {"lr":>5}  {"mcw":>5}  {"Mean F1":>10}  {"Std F1":>8}')
print('-' * 55)

for n_est in n_estimators_list:
    for depth in max_depth_list:
        for lr in learning_rate_list:
            for mcw in min_child_weight_list:
                xgb_pipe = Pipeline([
                    ('feature_engineering', feature_transformer),
                    ('preprocessor', preprocessor),
                    ('model', XGBClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        min_child_weight=mcw,
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
                    'min_child_weight': mcw,
                    'mean_f1': scores.mean(),
                    'std_f1': scores.std()
                })
                print(f'{n_est:>6}  {depth:>5}  {lr:>5.2f}  {mcw:>5}  '
                      f'{scores.mean():>10.4f}  {scores.std():>8.4f}')

xgb_tuning_df = pd.DataFrame(xgb_tuning_results)
best_row  = xgb_tuning_df.loc[xgb_tuning_df['mean_f1'].idxmax()]
best_n_est = int(best_row['n_estimators'])
best_depth = int(best_row['max_depth'])
best_lr    = float(best_row['learning_rate'])
best_mcw   = int(best_row['min_child_weight'])
print(f'\\nBest: n_estimators={best_n_est}, max_depth={best_depth}, '
      f'learning_rate={best_lr}, min_child_weight={best_mcw}')
print(f'Best CV F1: {best_row["mean_f1"]:.4f}')\
"""
set_src(cells[30], new_cell30)
print("[4] Cell 30 — XGBoost grid updated with min_child_weight tuning")

# --------------------------------------------------------------------------
# 5. Cell 31 — update training cell
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
        min_child_weight=best_mcw,
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
print("[5] Cell 31 — best_mcw added, y_pred_xgb_default renamed to y_pred_xgb")

# --------------------------------------------------------------------------
# 6. Cell 32 — update confusion matrix
# --------------------------------------------------------------------------
new_cell32 = """\
# Confusion matrix — XGBoost
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_xgb),
    display_labels=['Not Disputed', 'Disputed']
).plot(ax=ax, colorbar=False)
ax.set_title('Confusion Matrix — XGBoost')
plt.tight_layout()
plt.show()\
"""
set_src(cells[32], new_cell32)
print("[6] Cell 32 — y_pred_xgb_default -> y_pred_xgb")

# --------------------------------------------------------------------------
# 7. Delete cells 33, 34, 35 (threshold section)
# --------------------------------------------------------------------------
assert "Decision Threshold" in src(cells[33]) or "Threshold" in src(cells[33]), \
    f"Cell 33 unexpected: {src(cells[33])[:80]}"
assert "StratifiedKFold" in src(cells[34]), \
    f"Cell 34 unexpected: {src(cells[34])[:80]}"
assert "best_threshold" in src(cells[35]), \
    f"Cell 35 unexpected: {src(cells[35])[:80]}"

del cells[35]
del cells[34]
del cells[33]
print("[7] Cells 33-35 deleted (threshold optimisation section)")

# --------------------------------------------------------------------------
# 8. Cell 42 → now cell 39 after deletion — remove threshold saving
# --------------------------------------------------------------------------
target_idx = None
for i, c in enumerate(cells):
    if "THRESHOLD_FILENAME" in src(c) and "PICKLE_FILENAME" in src(c):
        target_idx = i
        break

assert target_idx is not None, "Could not find pickle/threshold cell"
print(f"[8] Found pickle/threshold cell at new index {target_idx}")

new_save = """\
# Save the best model pipeline as a Pickle file
PICKLE_FILENAME = '70299_Pipeline.pkl'

with open(PICKLE_FILENAME, 'wb') as f:
    pickle.dump(best_model_pipeline, f)

print(f'Pipeline saved : {PICKLE_FILENAME}  ({os.path.getsize(PICKLE_FILENAME)/1024:.1f} KB)')\
"""
set_src(cells[target_idx], new_save)
print(f"[8] Cell {target_idx} — threshold saving removed")

# --------------------------------------------------------------------------
# Clear all outputs
# --------------------------------------------------------------------------
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

nb["cells"] = cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook saved. Total cells: {len(cells)}")
print("Done — all patches applied.")
