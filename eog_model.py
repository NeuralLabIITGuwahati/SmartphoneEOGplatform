"""
EOG Eye Movement Classification — Model Training
=================================================
Contains the model:
    EEGNet      — compact, channel-aware, best for small datasets

Folder structure expected before running:
    processed/
    ├── subject_01.npz
    ├── subject_02.npz
    └── ...
    results/        ← created automatically

Run:
    python eog_model.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import random
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PROCESSED_DIR  = "processed"
RESULTS_DIR    = "results"
TRAIL_TYPE = "" # For testing purpose, we an name different trials
SAMPLING_RATE  = 128
N_TIMEPOINTS   = 256        # 2 seconds × 128 Hz
N_CHANNELS     = 2          # horizontal + vertical
NUM_CLASSES    = 6
BATCH_SIZE     = 32
MAX_EPOCHS     = 100

CLASS_NAMES = ["Left", "Right", "Up", "Down", "Blink", "No Movement"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# MODEL 1 — EEGNet (fixed for channels_last)
# ─────────────────────────────────────────────

def build_eegnet(
    n_timepoints = N_TIMEPOINTS,
    n_channels   = N_CHANNELS,
    num_classes  = NUM_CLASSES,
    fs           = SAMPLING_RATE,
    F1           = 8,       # number of temporal filters
    D            = 2,       # depth multiplier for depthwise conv
    dropout_rate = 0.5
):
    """
    EEGNet adapted for 2-channel EOG, using channels_last format (default in TF).

    The key fix:
        OLD (broken): Reshape to (1, 2, 256)
                      TF treated 256 as the channel dim → 131K params, zero dimensions
        NEW (fixed) : Reshape to (256, 2, 1)
                      height=timepoints, width=EOG_channels, depth=1 ✓

    All kernels are oriented as (time_axis, channel_axis).
    Expected total parameters: ~2,000
    """

    F2            = F1 * D          # 8 × 2 = 16 total filters after depthwise
    kernel_length = fs // 2         # 64 samples = 500ms at 128 Hz

    # ── Input ────────────────────────────────────────────────────────────────
    inputs = layers.Input(shape=(n_timepoints, n_channels))
    # shape: (256, 2)

    # Reshape to (256, 2, 1)
    # TF channels_last 2D conv sees: height=256 (timepoints), width=2 (EOG channels), depth=1
    x = layers.Reshape((n_timepoints, n_channels, 1))(inputs)
    # shape: (256, 2, 1)

    # ── Block 1a — Temporal Convolution ──────────────────────────────────────
    # Kernel (64, 1): slides along HEIGHT (timepoints) only.
    # Does NOT touch WIDTH (EOG channels).
    # Learns frequency content of the EOG signal —
    # slow saccade waveforms, sharp blink spikes.
    # Parameters: 64 × 1 × 1 × 8 = 512

    x = layers.Conv2D(
        filters     = F1,
        kernel_size = (kernel_length, 1),   # (64, 1) — along timepoints
        padding     = "same",
        use_bias    = False
    )(x)
    # shape: (256, 2, 8)

    x = layers.BatchNormalization()(x)

    # ── Block 1b — Depthwise Convolution across EOG channels ─────────────────
    # Kernel (1, 2): slides along WIDTH (EOG channels) only.
    # Combines H and V channel information explicitly.
    # Each of the 8 temporal filters spawns D=2 channel filters → 16 total.
    # padding='valid' collapses the width dim: 2 → 1
    # Parameters: 1 × 2 × 8 × 2 = 32

    x = layers.DepthwiseConv2D(
        kernel_size          = (1, n_channels),  # (1, 2) — across EOG channels
        depth_multiplier     = D,
        padding              = "valid",           # collapses width: 2 → 1
        use_bias             = False,
        depthwise_constraint = tf.keras.constraints.MaxNorm(1.)
    )(x)
    # shape: (256, 1, 16)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)

    # Pool along timepoints: (4, 1) → 256 → 64
    x = layers.AveragePooling2D(pool_size=(4, 1))(x)
    # shape: (64, 1, 16)

    x = layers.Dropout(dropout_rate)(x)

    # ── Block 2 — Separable Convolution ──────────────────────────────────────
    # Kernel (16, 1): slides along timepoints.
    # Separable = depthwise + pointwise — 8-10x fewer params than Conv2D.
    # Parameters: depthwise(16×1×16) + pointwise(1×1×16×16) = 256 + 256 = 512

    x = layers.SeparableConv2D(
        filters     = F2,
        kernel_size = (16, 1),   # (16, 1) — along timepoints
        padding     = "same",
        use_bias    = False
    )(x)
    # shape: (64, 1, 16)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)

    # Pool along timepoints: (8, 1) → 64 → 8
    x = layers.AveragePooling2D(pool_size=(8, 1))(x)
    # shape: (8, 1, 16)

    x = layers.Dropout(dropout_rate)(x)

    # ── Classifier ───────────────────────────────────────────────────────────
    # Flatten (8, 1, 16) → (128,)
    # Direct Dense output — no hidden layer keeps params minimal.
    # Parameters: 128 × 6 + 6 = 774

    x = layers.Flatten()(x)

    outputs = layers.Dense(
        num_classes,
        activation        = "softmax",
        kernel_constraint = tf.keras.constraints.MaxNorm(0.25)
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EEGNet_EOG")
    return model


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_all_subjects(processed_dir):
    """
    Loads all subject .npz files.

    Returns:
        X        : (total_windows, 256, 2)
        y        : (total_windows,)
        subjects : (total_windows,)
    """
    all_X, all_y, all_subjects = [], [], []

    subject_files = sorted([
        f for f in os.listdir(processed_dir) if f.endswith(".npz")
    ])

    if not subject_files:
        raise FileNotFoundError(f"No .npz files found in: {processed_dir}")

    for subj_idx, fname in enumerate(subject_files):
        path = os.path.join(processed_dir, fname)
        data = np.load(path, allow_pickle=True)
        all_X.append(data["X"])
        all_y.append(data["y"])
        all_subjects.extend([subj_idx] * len(data["y"]))
        print(f"Loaded: {fname}  |  windows: {len(data['y'])}")

    X        = np.concatenate(all_X, axis=0).astype(np.float32)
    y        = np.concatenate(all_y, axis=0).astype(np.int32)
    subjects = np.array(all_subjects)

    print(f"\nTotal windows : {len(X)}")
    print(f"X shape       : {X.shape}")
    print(f"Class dist    : {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, subjects


# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────

def augment_with_scaling(X, y, scale_range=(0.8, 1.2), n_augments=2):
    """
    Randomly scales amplitude of each window.
    scale_range=(0.9, 1.1) means ±10% amplitude variation.
    This directly simulates electrode placement differences
    and individual EOG amplitude variation across subjects.
    """
    X_aug, y_aug = [X], [y]

    for _ in range(n_augments):
        scales  = np.random.uniform(
            scale_range[0], scale_range[1], size=(len(X), 1, 1)
        )
        X_scaled = (X * scales).astype(np.float32)
        X_aug.append(X_scaled)
        y_aug.append(y)

    return np.concatenate(X_aug), np.concatenate(y_aug)


def augment_with_noise(X, y, noise_std=0.03, n_augments=1):
    """
    Adds Gaussian noise scaled to the signal's std.
    noise_std=0.03 means noise is 3% of signal amplitude.
    After z-score normalisation, signal std≈1, so noise_std=0.03
    adds realistic electrode noise without distorting waveform shape.
    """
    X_aug, y_aug = [X], [y]

    for _ in range(n_augments):
        noise   = np.random.normal(0, noise_std, size=X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy.astype(np.float32))
        y_aug.append(y)

    return np.concatenate(X_aug), np.concatenate(y_aug)


def shift_window(window, shift):
    """
    Shifts signal by shift samples and fills with zeros.
    Positive shift → signal moves right (delayed onset)
    Negative shift → signal moves left (earlier onset)
    
    window : (256, 2)
    shift  : int, positive or negative
    """
    shifted = np.zeros_like(window)

    if shift > 0:
        shifted[shift:, :] = window[:-shift, :]   # move right, pad left
    elif shift < 0:
        shifted[:shift, :] = window[-shift:, :]   # move left, pad right
    else:
        shifted = window.copy()

    return shifted


def augment_with_shifts(X, y, max_shift=15, n_augments=1):
    """
    Creates time-shifted copies using zero-padding instead of circular roll.
    max_shift=15 samples = ~117ms at 128Hz — realistic reaction time variation.
    """
    X_aug, y_aug = [X], [y]

    for _ in range(n_augments):
        X_shifted = np.array([
            shift_window(X[i], np.random.randint(-max_shift, max_shift))
            for i in range(len(X))
        ])
        X_aug.append(X_shifted)
        y_aug.append(y)

    return np.concatenate(X_aug), np.concatenate(y_aug)


# ─────────────────────────────────────────────
# RESULT GENERATION
# ─────────────────────────────────────────────

def plot_loso_fold_results(fold_results, model_name, save_dir):
    df = pd.DataFrame(fold_results)

    # Bar plot: accuracy per fold / subject
    plt.figure(figsize=(10, 4))
    plt.bar(df["test_subject"].astype(str), df["accuracy"])
    plt.ylim(0, 1)
    plt.xlabel("Held-out Subject")
    plt.ylabel("Accuracy")
    plt.title(f"LOSO Fold-wise Accuracy — {model_name} {TRAIL_TYPE}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_{TRAIL_TYPE}_fold_accuracy_bar.png"), dpi=120)
    plt.show()

    # Boxplot: spread across folds
    plt.figure(figsize=(5, 4))
    plt.boxplot(df["accuracy"], vert=True)
    plt.ylabel("Accuracy")
    plt.title(f"LOSO Accuracy Distribution — {model_name} {TRAIL_TYPE}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_{TRAIL_TYPE}_fold_accuracy_boxplot.png"), dpi=120)
    plt.show()
    
    # Save table too
    df.to_csv(os.path.join(save_dir, f"{model_name}_{TRAIL_TYPE}_fold_results.csv"), index=False)


def generate_classification_report(
    y_true,
    y_pred,
    class_names,
    results_dir,
    model_name="EEGNet"
):
    """
    Generates, prints, saves, and visualizes
    aggregated classification report.

    Saves:
    - CSV table
    - PNG table image
    - TXT report

    Parameters
    ----------
    y_true : array-like
        Ground truth labels

    y_pred : array-like
        Predicted labels

    class_names : list
        List of class names

    results_dir : str or Path
        Directory to save outputs

    model_name : str
        Name used in filenames
    """

    # ---------------------------------------------------------
    # Generate report dictionary
    # ---------------------------------------------------------
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4
    )

    # ---------------------------------------------------------
    # Convert to DataFrame
    # ---------------------------------------------------------
    report_df = pd.DataFrame(report_dict).transpose()

    # ---------------------------------------------------------
    # Print nicely
    # ---------------------------------------------------------
    print("\n=== Aggregated Classification Report ===")
    print(report_df.round(4))

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------
    csv_path = os.path.join(
        results_dir,
        f"{model_name}_{TRAIL_TYPE}_classification_report.csv"
    )

    report_df.to_csv(csv_path)

    # ---------------------------------------------------------
    # Save TXT version
    # ---------------------------------------------------------
    txt_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    txt_path = os.path.join(
        results_dir,
        f"{model_name}_{TRAIL_TYPE}_classification_report.txt"
    )

    with open(txt_path, "w") as f:
        f.write(txt_report)

    # ---------------------------------------------------------
    # Create table figure
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axis("off")

    table = ax.table(
        cellText=report_df.round(3).values,
        colLabels=report_df.columns,
        rowLabels=report_df.index,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    plt.title(f"{model_name} - Classification Report - {TRAIL_TYPE}")
    plt.tight_layout()
    png_path = os.path.join(
        results_dir,
        f"{model_name}_{TRAIL_TYPE}_classification_report.png"
    )
    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nSaved:")
    print("CSV :", csv_path)
    print("TXT :", txt_path)
    print("PNG :", png_path)

    return report_df


def save_summary_dict(summary_dict, results_dir, model_name="EEGNet"):
    """
    Save summary dictionary as:
    - JSON
    - TXT
    """

    import os
    import json

    # -----------------------------------------
    # Save JSON
    # -----------------------------------------
    json_path = os.path.join(
        results_dir,
        f"{model_name}_{TRAIL_TYPE}_summary.json"
    )

    with open(json_path, "w") as f:
        json.dump(summary_dict, f, indent=4)

    # -----------------------------------------
    # Save TXT
    # -----------------------------------------
    txt_path = os.path.join(
        results_dir,
        f"{model_name}_{TRAIL_TYPE}_summary.txt"
    )

    with open(txt_path, "w") as f:
        for key, value in summary_dict.items():
            f.write(f"{key}: {value}\n")

    print("\n-----Summary-----")
    print(f" Mean accuracy : {summary_dict['mean_accuracy']}")
    print(f" Std deviation : {summary_dict['std_deviation']}")
    print(f" Range         : {summary_dict['range']}")
    
    print("\nSaved summary:")
    print("JSON :", json_path)
    print("TXT  :", txt_path)


# ─────────────────────────────────────────────
# LOSO CROSS-VALIDATION
# ─────────────────────────────────────────────

def cross_validate_loso(build_fn, X, y, subjects, model_name):
    """
    Leave-One-Subject-Out cross-validation.
    Trains and evaluates once per subject, reports mean ± std accuracy.

    Args:
        build_fn   : function that returns a compiled-ready model
                     e.g. build_eegnet or build_1d_cnn
        X          : (total_windows, 256, 2)
        y          : (total_windows,)
        subjects   : (total_windows,) — subject index per window
        model_name : string label for printing
    """
    unique_subjects = np.unique(subjects)
    fold_accuracies = []
    all_y_true, all_y_pred = [], []
    fold_results = []

    print(f"\n{'='*55}")
    print(f"LOSO Cross-Validation: {model_name}")
    print(f"{'='*55}")

    for fold, test_subj in enumerate(unique_subjects):

        # Split — one subject out
        test_mask  = subjects == test_subj
        train_mask = subjects != test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_test_cat  = to_categorical(y_test,  num_classes=NUM_CLASSES)

        # Build a fresh model for each fold
        model = build_fn()
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss      = "categorical_crossentropy",
            metrics   = ["accuracy"]
        )

        # Callbacks — no ModelCheckpoint in LOSO (11 checkpoints is messy)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor              = "val_loss",
                patience             = 15,
                restore_best_weights = True,
                verbose              = 0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor  = "val_loss",
                factor   = 0.5,
                patience = 7,
                min_lr   = 1e-5,
                verbose  = 0
            ),
            FoldSummaryCallback()
        ]
        
        # Compute class weights from this fold's training labels
        class_weights = compute_class_weight(
            class_weight = "balanced",
            classes      = np.unique(y_train),
            y            = y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        # Data augmentation by amplitude scaling
        X_aug, y_aug = augment_with_scaling(X_train, y_train,
                                     scale_range=(0.85, 1.15), n_augments=2)
        X_aug, y_aug = augment_with_noise(X_aug, y_aug,
                                   noise_std=0.03, n_augments=1)
        X_aug, y_aug = augment_with_shifts(X_aug, y_aug,
                                    max_shift=15, n_augments=1)
        y_aug_cat = to_categorical(y_aug, num_classes=NUM_CLASSES)
        
        # Train
        model.fit(
            X_aug, y_aug_cat,
            epochs           = MAX_EPOCHS,
            batch_size       = BATCH_SIZE,
            validation_split = 0.20,  # small val split for early stopping only
            callbacks        = callbacks,
            shuffle          = True,
            verbose          = 1,      # silent per fold — summary printed below
            class_weight     = class_weight_dict
        )

        # Evaluate
        _, acc     = model.evaluate(X_test, y_test_cat, verbose=0)
        y_pred_int = np.argmax(model.predict(X_test, verbose=0), axis=1)

        fold_accuracies.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred_int.tolist())
        fold_results.append({
            "fold": fold + 1,
            "test_subject": int(test_subj) + 1,
            "accuracy": float(acc),
            "test_windows": int(test_mask.sum())
        })

        print(f"  Fold {fold+1:02d} | Test subject: {test_subj:02d} "
              f"| Windows: {test_mask.sum():3d} | Accuracy: {acc:.4f}")
        
        # Free memory between folds
        tf.keras.backend.clear_session()

    # ── Summary ──────────────────────────────────────────────────────────────
    mean_acc = np.mean(fold_accuracies)
    std_acc  = np.std(fold_accuracies)

    summary_dict={
        "mean_accuracy": f"{mean_acc:.4f}",
        "std_deviation":f"{std_acc:.4f}",
        "range": f"{min(fold_accuracies):.4f} – {max(fold_accuracies):.4f}",
        "params": f"{build_eegnet().count_params()}"
    }
    save_summary_dict(
        summary_dict,
        RESULTS_DIR,
        model_name="EEGNet"
    )
    

    # Overall classification report across all folds
    generate_classification_report(
        y_true=all_y_true,
        y_pred=all_y_pred,
        class_names=CLASS_NAMES,
        results_dir=RESULTS_DIR,
        model_name=model_name
    )

    # Aggregated confusion matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        all_y_true, all_y_pred,
        display_labels = CLASS_NAMES,
        cmap           = "Blues",
        ax             = ax
    )
    ax.set_title(
        f"LOSO Confusion Matrix - {model_name} {TRAIL_TYPE}\n"
        f"Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}",
        fontsize=12
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{model_name}_{TRAIL_TYPE}_loso_confusion_matrix.png")
    plt.savefig(path, dpi=100)
    plt.show()
    print(f"Saved: {path}")
    
    # Plotting LOSO fold results
    plot_loso_fold_results(fold_results, model_name, RESULTS_DIR)

    return mean_acc, std_acc, fold_accuracies

class FoldSummaryCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        final_train_acc = self.model.history.history["accuracy"][-1]
        final_val_acc   = self.model.history.history["val_accuracy"][-1]
        best_val_loss   = min(self.model.history.history["val_loss"])
        n_epochs        = len(self.model.history.history["loss"])

        print(f"    Training finished at epoch {n_epochs}")
        print(f"    Final train acc : {final_train_acc:.4f}")
        print(f"    Final val acc   : {final_val_acc:.4f}")
        print(f"    Best val loss   : {best_val_loss:.4f}")
        gap = final_train_acc - final_val_acc
        if gap > 0.15:
            print(f"    ⚠ Overfit gap   : {gap:.4f}  — consider more dropout")
        else:
            print(f"    ✓ Overfit gap   : {gap:.4f}  — acceptable")

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    X, y, subjects = load_all_subjects(PROCESSED_DIR)

    results = []

    # LOSO for each model
    acc, std, folds = cross_validate_loso(build_eegnet,     X, y, subjects, "EEGNet")
    results.append(("EEGNet",      acc, std, build_eegnet().count_params()))

    # acc, std, folds = cross_validate_loso(build_1d_cnn,     X, y, subjects, "1D_CNN")
    # results.append(("1D-CNN",      acc, std, build_1d_cnn().count_params()))

    # acc, std, folds = cross_validate_loso(build_1d_cnn_lite, X, y, subjects, "1D_CNN_Lite")
    # results.append(("1D-CNN Lite", acc, std, build_1d_cnn_lite().count_params()))

    # Final comparison
    print(f"\n{'='*55}")
    print("FINAL COMPARISON (LOSO)")
    print(f"{'='*55}")
    print(f"{'Model':<20} {'Params':<10} {'Mean Acc':<12} {'Std'}")
    print("-" * 55)
    for name, acc, std, params in sorted(results, key=lambda x: -x[1]):
        print(f"{name:<20} {params:<10,} {acc:.4f}       ±{std:.4f}")
