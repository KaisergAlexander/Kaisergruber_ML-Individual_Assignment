"""
Patch 70299_ModelTesting_Notebook.ipynb:
  - Remove Step 3b cells (threshold load markdown + code) at indices 7 and 8
  - Revert Step 5 predict cell to use loaded_pipeline.predict() instead of predict_proba >= threshold
  - Update Step 0 description: remove mention of threshold file
"""

import json

NB_PATH = "70299_ModelTesting_Notebook.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

def src(cell):
    return "".join(cell["source"])

def set_src(cell, text):
    lines = text.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

# ── Verify cells 7 and 8 are the threshold cells ──────────────────────────
assert "Step 3b" in src(cells[7]) or "Threshold" in src(cells[7]), \
    f"Cell 7 is unexpected: {src(cells[7])[:80]}"
assert "THRESHOLD_PATH" in src(cells[8]), \
    f"Cell 8 is unexpected: {src(cells[8])[:80]}"

del cells[8]
del cells[7]
print("[1] Deleted Step 3b cells (threshold markdown + load code)")

# ── After deletion: old cell 12 is now cell 10 ────────────────────────────
# Find the predict cell
pred_idx = None
for i, c in enumerate(cells):
    if "predict_proba" in src(c) or "OPTIMAL_THRESHOLD" in src(c):
        pred_idx = i
        break

assert pred_idx is not None, "Could not find predict cell"
print(f"[2] Found predict cell at new index {pred_idx}")

new_pred = """\
# Columns to exclude from the feature matrix
COLS_TO_EXCLUDE = ['Consumer disputed?', 'Complaint ID']

X_val = df_test.drop(
    columns=[c for c in COLS_TO_EXCLUDE if c in df_test.columns]
)

# Generate binary predictions using the pipeline directly
val_predictions = loaded_pipeline.predict(X_val)

print(f'Predictions generated  : {len(val_predictions)}')
print(f'Unique output values   : {list(set(val_predictions.tolist()))}  (must be 0 and/or 1)')
print(f'Predicted dispute rate : {val_predictions.mean():.1%}')
print()
print('First 20 predictions:')
if 'Complaint ID' in df_test.columns:
    preview = pd.DataFrame({
        'Complaint ID'    : df_test['Complaint ID'].values[:20],
        'Prediction (0/1)': val_predictions[:20]
    })
else:
    preview = pd.DataFrame({'Prediction (0/1)': val_predictions[:20]})
print(preview.to_string(index=False))\
"""
set_src(cells[pred_idx], new_pred)
print(f"[2] Cell {pred_idx} patched — reverted to predict()")

# ── Update Step 0 title cell (cell 0) to remove threshold mention ─────────
c0 = cells[0]
old0 = src(c0)
new0 = old0.replace(
    "| 3 | Load the serialised pipeline from `70299_Pipeline.pkl` |\n"
    "| 4 | Load the external validation dataset (`complaints_modeltesting100.csv`) |\n"
    "| 5 | Generate binary predictions (0 = Not Disputed, 1 = Disputed) |\n"
    "| 6 | Validate output format and summarise results |",
    "| 3 | Load the serialised pipeline from `70299_Pipeline.pkl` |\n"
    "| 4 | Load the external validation dataset (`complaints_modeltesting100.csv`) |\n"
    "| 5 | Generate binary predictions (0 = Not Disputed, 1 = Disputed) |\n"
    "| 6 | Validate output format and summarise results |"
)
# Remove "Step 3b" from the steps table if present
new0 = "\n".join(
    line for line in new0.split("\n")
    if "3b" not in line and "Threshold" not in line and "threshold" not in line
)
set_src(c0, new0)
print("[3] Cell 0 updated — threshold references removed from steps table")

# Clear outputs
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

nb["cells"] = cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nTesting notebook saved. Total cells: {len(cells)}")
print("Done.")
