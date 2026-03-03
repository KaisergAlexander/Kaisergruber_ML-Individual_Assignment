"""
Patch 70299_Complaints_Notebook.ipynb on optimize/threshold branch.

Changes:
  1. Cell 0  - remove 70299_Threshold.json row from submission table
  2. Cell 29 - update markdown: scale_pos_weight is now tuned, not fixed
  3. Cell 30 - replace XGBoost grid with scale_pos_weight tuning + regularisation
  4. Cell 31 - add best_spw, rename _default variables, add subsample/colsample_bytree
  5. Cell 32 - rename y_pred_xgb_default -> y_pred_xgb, update title
  6. Delete cells 33, 34, 35 (threshold optimisation section)
  7. Cell 42 (→39 post-delete) - remove threshold saving; save pipeline only
"""

import json, re, copy

NB_PATH = "70299_Complaints_Notebook.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
def src(cell):
    return "".join(cell["source"])

def set_src(cell, text):
    cell["source"] = [line + "\n" for line in text.split("\n")]
    # Fix last line (no trailing newline)
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

# --------------------------------------------------------------------------
# 1. Cell 0 — remove Threshold.json row from submission table
# --------------------------------------------------------------------------
c0 = cells[0]
old0 = src(c0)
new0 = old0.replace(
    "| `70299_Threshold.json` | Optimal decision threshold for F1 maximisation |\n",
    ""
)
set_src(c0, new0)
assert "Threshold.json" not in src(c0), "Cell 0 patch failed"
print("[1] Cell 0 patched — Threshold.json row removed")

# --------------------------------------------------------------------------
# 2. Cell 29 — update XGBoost description to mention scale_pos_weight tuning
# --------------------------------------------------------------------------
c29 = cells[29]
old29 = src(c29)
new29 = old29.replace(
    "- `scale_pos_weight` fixed at `n_negative / n_positive` = 4.02",
    "- `scale_pos_weight`: class-imbalance penalty (tuned; lower values improve precision)"
)
set_src(c29, new29)
print("[2] Cell 29 patched — scale_pos_weight description updated")

# --------------------------------------------------------------------------
# 3. Cell 30 — replace XGBoost grid
# --------------------------------------------------------------------------
new_cell30 = """\
# XGBoost: hyperparameter tuning over n_estimators, max_depth, learning_rate,
# and scale_pos_weight (Lecture 6 — Boosting).
# subsample & colsample_bytree fixed at 0.8 for regularisation (reduces overfitting).

n_estimators_list    = [300, 500]
max_depth_list       = [5, 7]
learning_rate_list   = [0.05, 0.1]
scale_pos_weight_list = [1.5, 2.0, 3.0, 4.0]   # lower = more precise, higher = more recall

xgb_tuning_results = []

print('XGBoost — 5-fold CV F1 (scale_pos_weight tuning):')
print(f'{"n_est":>6}  {"depth":>5}  {"lr":>5}  {"spw":>5}  {"Mean F1":>10}  {"Std F1":>8}')
print('-' * 58)

for n_est in n_estimators_list:
    for depth in max_depth_list:
        for lr in learning_rate_list:
            for spw in scale_pos_weight_list:
                xgb_pipe = Pipeline([
                    ('feature_engineering', feature_transformer),
                    ('preprocessor', preprocessor),
                    ('model', XGBClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        scale_pos_weight=spw,
                        subsample=0.8,
                        colsample_bytree=0.8,
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
                    'scale_pos_weight': spw,
                    'mean_f1': scores.mean(),
                    'std_f1': scores.std()
                })
                print(f'{n_est:>6}  {depth:>5}  {lr:>5.2f}  {spw:>5.1f}  '
                      f'{scores.mean():>10.4f}  {scores.std():>8.4f}')

xgb_tuning_df = pd.DataFrame(xgb_tuning_results)
best_row  = xgb_tuning_df.loc[xgb_tuning_df['mean_f1'].idxmax()]
best_n_est = int(best_row['n_estimators'])
best_depth = int(best_row['max_depth'])
best_lr    = float(best_row['learning_rate'])
best_spw   = float(best_row['scale_pos_weight'])
print(f'\\nBest: n_estimators={best_n_est}, max_depth={best_depth}, '
      f'learning_rate={best_lr}, scale_pos_weight={best_spw}')
print(f'Best CV F1: {best_row["mean_f1"]:.4f}')\
"""
set_src(cells[30], new_cell30)
print("[3] Cell 30 patched — XGBoost grid updated with scale_pos_weight tuning")

# --------------------------------------------------------------------------
# 4. Cell 31 — update training cell
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
        scale_pos_weight=best_spw,
        subsample=0.8,
        colsample_bytree=0.8,
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
print("[4] Cell 31 patched — best_spw added, variables renamed, subsample/colsample_bytree added")

# --------------------------------------------------------------------------
# 5. Cell 32 — update confusion matrix cell
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
print("[5] Cell 32 patched — y_pred_xgb_default -> y_pred_xgb")

# --------------------------------------------------------------------------
# 6. Delete cells 33, 34, 35 (threshold tuning markdown + 2 code cells)
# --------------------------------------------------------------------------
# Verify before deleting
assert "Decision Threshold" in src(cells[33]), "Cell 33 is not the threshold markdown!"
assert "StratifiedKFold" in src(cells[34]),     "Cell 34 is not the OOF code!"
assert "best_threshold" in src(cells[35]),       "Cell 35 is not the threshold CM!"

del cells[35]
del cells[34]
del cells[33]
print("[6] Cells 33-35 deleted (threshold optimisation section removed)")

# --------------------------------------------------------------------------
# 7. Cell 42 is now cell 39 after deletion.
#    Find it by looking for PICKLE_FILENAME and THRESHOLD_FILENAME.
# --------------------------------------------------------------------------
target_idx = None
for i, c in enumerate(cells):
    if "THRESHOLD_FILENAME" in src(c) and "PICKLE_FILENAME" in src(c):
        target_idx = i
        break

assert target_idx is not None, "Could not find the pickle-saving cell!"
print(f"[7] Found pickle/threshold cell at new index {target_idx}")

new_cell_save = """\
# Save the best model pipeline as a Pickle file
PICKLE_FILENAME = '70299_Pipeline.pkl'

with open(PICKLE_FILENAME, 'wb') as f:
    pickle.dump(best_model_pipeline, f)

print(f'Pipeline saved : {PICKLE_FILENAME}  ({os.path.getsize(PICKLE_FILENAME)/1024:.1f} KB)')\
"""
set_src(cells[target_idx], new_cell_save)
print(f"[7] Cell {target_idx} patched — threshold saving removed")

# --------------------------------------------------------------------------
# Write back
# --------------------------------------------------------------------------
# Clear all outputs to avoid stale data
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

nb["cells"] = cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook saved. Total cells: {len(cells)}")
print("Done — all patches applied successfully.")
