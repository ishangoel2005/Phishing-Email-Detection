import shap
import json
import numpy as np
import scipy.sparse as sp
import joblib
import matplotlib.pyplot as plt
import os

print("Loading model...")
# The file raw_lightgbm.pkl is in model_artifacts
model = joblib.load("model_artifacts/raw_lightgbm.pkl")

print("Loading features...")
# `features.npz` is in artifacts/
X = sp.load_npz("artifacts/features.npz")

print("Loading feature names...")
with open("artifacts/feature_names.json", "r") as f:
    feature_names_dict = json.load(f)

# Combine feature names based on how they were hstacked in features.py
# X_text (tfidf) then X_numeric_sparse
feature_names = []
if "tfidf" in feature_names_dict:
    feature_names.extend(feature_names_dict["tfidf"])
if "numeric" in feature_names_dict:
   feature_names.extend(feature_names_dict["numeric"])

print(f"X shape: {X.shape}, Feature names len: {len(feature_names)}")

# SHAP can be very slow, so we will use a small subset of the training data
# Using 500 samples
NUM_SAMPLES = min(500, X.shape[0])
# We have to use dense subset for TreeExplainer sometimes, although lightgbm may support sparse.
# Let's convert to dense for the plot subset
# SHAP TreeExplainer prefers dense arrays or pd.DataFrame
X_subset = X[:NUM_SAMPLES].toarray()

print(f"Calculating SHAP values for {NUM_SAMPLES} samples...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_subset)

print("Plotting SHAP summary graph...")
plt.figure()
# If binary classification, lightgbm returns list of two arrays depending on shap version
if isinstance(shap_values, list):
    # Usually [0] is negative class, [1] is positive class
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

shap.summary_plot(shap_values_to_plot, X_subset, feature_names=feature_names, show=False)

plt.tight_layout()
out_file = "model_artifacts/shap_summary_plot.png"
plt.savefig(out_file, bbox_inches='tight')
print(f"SHAP graph saved to {out_file}")
