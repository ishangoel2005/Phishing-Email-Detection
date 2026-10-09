# src/model.py
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
import joblib
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
import argparse
import warnings
warnings.filterwarnings("ignore")

# NEW imports for progress bar & plotting
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files

# Use LightGBM callbacks
from lightgbm.callback import early_stopping, log_evaluation, record_evaluation

# -------------------------
# Configurable hyperparams
# -------------------------
N_SPLITS = 5
RANDOM_STATE = 42

LGB_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 1000,   # large; will rely on best_iteration from early stopping in CV
    'random_state': RANDOM_STATE,
    'verbosity': -1
}

# -------------------------
# Utilities
# -------------------------
def eval_at_thresholds(y_true, probs, thresholds):
    out = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        out.append({
            'threshold': float(t),
            'precision': float(precision_score(y_true, preds, zero_division=0)),
            'recall': float(recall_score(y_true, preds, zero_division=0)),
            'f1': float(f1_score(y_true, preds, zero_division=0))
        })
    return out

def plot_metrics_per_fold(evals_result, out_path_prefix, fold_idx):
    """
    evals_result: dict collected by record_evaluation (format: {'valid_0': {'auc': [...], 'binary_logloss': [...]}, 'train': {...}})
    Save two plots: AUC per iter and binary_logloss per iter for train & valid_0.
    """
    # find keys for train/valid
    # evals_result typically like: {'training': {'auc': [...], 'binary_logloss': [...]}, 'valid_0': {...}}
    keys = list(evals_result.keys())
    # unify names
    train_key = keys[0]
    val_key = keys[1] if len(keys) > 1 else None

    # plot AUC
    if 'auc' in evals_result[train_key]:
        plt.figure(figsize=(8,4))
        plt.plot(evals_result[train_key]['auc'], label=f'{train_key} auc')
        if val_key and 'auc' in evals_result[val_key]:
            plt.plot(evals_result[val_key]['auc'], label=f'{val_key} auc')
        plt.xlabel('Iteration')
        plt.ylabel('AUC')
        plt.title(f'Fold {fold_idx} AUC per iteration')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_path_prefix}_fold{fold_idx}_auc.png")
        plt.close()

    # plot binary_logloss
    if 'binary_logloss' in evals_result[train_key]:
        plt.figure(figsize=(8,4))
        plt.plot(evals_result[train_key]['binary_logloss'], label=f'{train_key} logloss')
        if val_key and 'binary_logloss' in evals_result[val_key]:
            plt.plot(evals_result[val_key]['binary_logloss'], label=f'{val_key} logloss')
        plt.xlabel('Iteration')
        plt.ylabel('Binary Logloss')
        plt.title(f'Fold {fold_idx} Binary Logloss per iteration')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_path_prefix}_fold{fold_idx}_logloss.png")
        plt.close()

def save_confusion_and_report(y_true, probs, threshold, out_dir):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, output_dict=True, zero_division=0)

    # save classification report JSON
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    # save confusion matrix plot
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix (OOF)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["legit (0)","phish (1)"])
    plt.yticks(tick_marks, ["legit (0)","phish (1)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

# -------------------------
# Training function
# -------------------------
def main(features_path: str, labels_path: str, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading features and labels ...")
    X = load_npz(features_path)
    y = np.load(labels_path)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(y))
    fold_best_iters = []
    fold = 0

    # For aggregate plotting later we can store per-fold evals if desired
    all_evals = []

    print("Starting cross-validated LightGBM training...")
    # Wrap outer loop with tqdm for folds
    for train_idx, val_idx in tqdm(skf.split(X, y), total=N_SPLITS, desc="Folds"):
        fold += 1
        print(f"\n Fold {fold}/{N_SPLITS}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # prepare evals_result container for this fold
        evals_result = {}
        # create a tqdm progress bar for iterations (n_estimators)
        pbar = tqdm(total=LGB_PARAMS['n_estimators'], desc=f"Fold{fold} iters", leave=False, unit="iter")

        # define a callback to update tqdm each iteration
        def _tqdm_callback(env):
            # env.iteration starts at 0, but update by 1 each call
            pbar.update(1)
        _tqdm_callback.order = 0  # ensure it runs early

        # build LightGBM classifier and fit with callbacks to record evals and show progress
        lgb_clf = lgb.LGBMClassifier(**LGB_PARAMS)
        # Prepare callbacks: record_evaluation to capture metrics, early_stopping, our tqdm callback, and suppress default logging
        callbacks = [
            record_evaluation(evals_result),
            early_stopping(stopping_rounds=50, verbose=False),
            _tqdm_callback,
            # log_evaluation with 0 to suppress per-iteration prints (we use tqdm)
            log_evaluation(period=0)
        ]

        lgb_clf.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_metric=['auc', 'binary_logloss'],
            callbacks=callbacks
        )

        # finish the bar for this fold
        pbar.close()

        # store best iteration if available
        best_iter = getattr(lgb_clf, "best_iteration_", None)
        if best_iter is None:
            best_iter = lgb_clf.n_estimators
        fold_best_iters.append(int(best_iter))
        print(f"  Best iter: {best_iter}")

        # Predict on validation (probabilities)
        val_probs = lgb_clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs

        # Save per-fold eval plots (AUC and logloss per iter)
        # evals_result format: {'training': {'auc': [...], 'binary_logloss': [...]}, 'valid_1': {...} ... }
        # Save fold evals to list for potential aggregate plotting
        all_evals.append(evals_result)
        out_path_prefix = str(out_dir / "evals")
        try:
            plot_metrics_per_fold(evals_result, out_path_prefix, fold)
            print(f"  Saved fold {fold} metric plots to {out_dir}")
        except Exception as e:
            print("  Warning: failed to plot per-fold metrics:", e)

    # -------------------------
    # OOF evaluation
    # -------------------------
    print("\nComputing OOF metrics...")
    roc_auc = roc_auc_score(y, oof_probs)
    pr_auc = average_precision_score(y, oof_probs)
    precision_at_0_5 = precision_score(y, (oof_probs >= 0.5).astype(int), zero_division=0)
    recall_at_0_5 = recall_score(y, (oof_probs >= 0.5).astype(int), zero_division=0)
    f1_at_0_5 = f1_score(y, (oof_probs >= 0.5).astype(int), zero_division=0)

    # choose best threshold by maximizing F1 on OOF
    precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
    f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = np.nanargmax(f1s)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'precision_at_0.5': float(precision_at_0_5),
        'recall_at_0.5': float(recall_at_0_5),
        'f1_at_0.5': float(f1_at_0_5),
        'best_threshold_f1': best_threshold,
        'best_f1': float(f1s[best_idx])
    }

    print("OOF ROC-AUC: {:.4f}, PR-AUC: {:.4f}, best F1: {:.4f} @ {:.4f}".format(
        roc_auc, pr_auc, metrics['best_f1'], metrics['best_threshold_f1']
    ))

    # Save OOF preds and metrics
    np.save(out_dir / "oof_preds.npy", oof_probs)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved oof_preds.npy and metrics.json to {out_dir}")

    # Save classification report and confusion matrix (using chosen threshold)
    try:
        save_confusion_and_report(y, oof_probs, best_threshold, out_dir)
        print(f"Saved classification_report.json and confusion_matrix.png to {out_dir}")
    except Exception as e:
        print("Warning: failed to save classification report/CM:", e)

    # Additionally, save a PR curve and ROC curve plot for OOF
    try:
        # PR curve
        plt.figure(figsize=(6,4))
        plt.plot(recalls, precisions, label=f'PR curve (AP={pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall (OOF)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "oof_pr_curve.png")
        plt.close()

        # ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, oof_probs)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.4f})')
        plt.plot([0,1],[0,1], linestyle='--', color='gray')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve (OOF)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "oof_roc_curve.png")
        plt.close()
        print(f"Saved OOF PR and ROC curve plots to {out_dir}")
    except Exception as e:
        print("Warning: failed to plot OOF PR/ROC curves:", e)

    # -------------------------
    # Final model training on full data
    # -------------------------
    median_best_iter = int(np.median(fold_best_iters))
    final_params = LGB_PARAMS.copy()
    final_params['n_estimators'] = median_best_iter if median_best_iter > 0 else final_params.get('n_estimators', 100)
    print(f"\nTraining final LightGBM on full data with n_estimators={final_params['n_estimators']} ...")

    final_clf = lgb.LGBMClassifier(**final_params)
    final_clf.fit(X, y)
    # Save raw final LGBM
    joblib.dump(final_clf, out_dir / "raw_lightgbm.pkl")
    print(f"Saved uncalibrated LightGBM to {out_dir/'raw_lightgbm.pkl'}")

    # -------------------------
    # Calibration
    # -------------------------
    print("Calibrating classifier with CalibratedClassifierCV (cv=5, method='isotonic') ...")
    calibrated = CalibratedClassifierCV(base_estimator=lgb.LGBMClassifier(**final_params),
                                       method='isotonic',
                                       cv=N_SPLITS)
    calibrated.fit(X, y)  # this will train internal models and calibrators
    # Save calibrated model
    joblib.dump(calibrated, out_dir / "model.pkl")
    print(f"Saved calibrated classifier to {out_dir/'model.pkl'}")

    # Also save chosen threshold for inference
    (out_dir / "decision_threshold.txt").write_text(str(best_threshold))
    print(f"Saved decision threshold ({best_threshold}) to {out_dir/'decision_threshold.txt'}")

    print("Training complete. Artifacts saved in:", out_dir)

# -------------------------
# CLI or hard-coded path section
# -------------------------
if __name__ == "__main__":
    # You can edit these defaults if you want
    INPUT_FEATURES = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/features.npz"
    INPUT_LABELS = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/labels.npy"
    OUTPUT_DIR = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/model_artifacts"

    # If INPUT_FEATURES is empty, script will accept CLI args (unchanged behavior)
    if INPUT_FEATURES:
        main(INPUT_FEATURES, INPUT_LABELS, OUTPUT_DIR)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--features", default="artifacts/features.npz")
        parser.add_argument("--labels", default="artifacts/labels.npy")
        parser.add_argument("--output_dir", default="artifacts")
        args = parser.parse_args()
        main(args.features, args.labels, args.output_dir)