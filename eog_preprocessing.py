"""
EOG Data Preprocessing Script
==============================
Folder structure expected:
    EOG Data/
    ├── Subject_01/
    │   ├── Up/
    │   │   ├── trial_1.csv
    │   │   ├── trial_2.csv
    │   │   ├── trial_3.csv
    │   │   └── trial_4.csv
    │   ├── Down/
    │   ├── Left/
    │   ├── Right/
    │   ├── Blink/
    │   └── No_Movement/
    ├── Subject_02/
    └── ...

Each CSV has columns: sample_index, time_ms, h_signal, v_signal, label
Label column has a value only in the first row.

Output:
    processed/
    ├── subject_01.npz
    ├── subject_02.npz
    └── ...
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# CONFIGURATION — adjust these to match your setup
# ─────────────────────────────────────────────

RAW_DATA_DIR   = "raw_data"          # path to your root data folder
PROCESSED_DIR  = "processed"         # where .npz files will be saved
SAMPLING_RATE  = 128                 # Hz — 7.8125ms per sample
WINDOW_SEC     = 2                   # seconds per window
MOVEMENT_WINDOW_INDEX = 1            # 0 = first window is movement, 1 = second window is movement

# Skip the first window of every trial (always the pre-movement centre-fixation baseline)
SKIP_FIRST_WINDOW = True

# Optionally skip the last window of every trial.
# Enable this if your plots show the last window looks like an abrupt signal cut-off
# rather than a clean return-to-centre. Check visually with verify_all_windows() first.
SKIP_LAST_WINDOW = False

LABEL_MAP = {
    "up":          2,
    "down":        3,
    "left":        0,
    "right":       1,
    "blink":       4,
    "no_movement": 5,
}

# Folder name → label key mapping
FOLDER_TO_LABEL = {
    "Up":          "up",
    "Down":        "down",
    "Left":        "left",
    "Right":       "right",
    "Blink":       "blink",
    "No Movement": "no_movement",
}

# Only horizontal classes get return window relabelling
# Up and Down return windows are discarded — too similar to blinks
OPPOSITE_CLASS = {
    "Left":  "right",
    "Right": "left",
}


# ─────────────────────────────────────────────
# STEP 1 — VERIFY ALL WINDOWS (run this first)
# ─────────────────────────────────────────────

def verify_all_windows(subject_folder, movement_folder="Up", trial_index=0):
    """
    Plots ALL non-overlapping windows from one trial so you can visually confirm:
      - Window 0  : centre-fixation baseline (should be flat) → will be skipped
      - Window 1+ : movement + return-to-centre cycles → will be kept
      - Last window: check if it looks like a clean return or an abrupt cut-off.
                     If cut-off, set SKIP_LAST_WINDOW = True.

    Usage:
        verify_all_windows("EOG Data/Subject_01", movement_folder="Up")
        verify_all_windows("EOG Data/Subject_01", movement_folder="Left", trial_index=1)
    """
    movement_path = os.path.join(subject_folder, movement_folder)
    trial_files   = sorted(glob.glob(os.path.join(movement_path, "*.csv")))

    if not trial_files:
        print(f"No CSV files found in {movement_path}")
        return

    filepath = trial_files[trial_index]
    df = pd.read_csv(filepath)
    h  = df["h_signal"].values
    v  = df["v_signal"].values

    ws            = WINDOW_SEC * SAMPLING_RATE          # 256 samples
    n_total       = len(h) // ws
    start_idx     = 1 if SKIP_FIRST_WINDOW else 0
    end_idx       = n_total - 1 if SKIP_LAST_WINDOW else n_total
    kept_indices  = list(range(start_idx, end_idx))

    fig, axes = plt.subplots(n_total, 2, figsize=(12, n_total * 2.2))
    fig.suptitle(
        f"All windows — {movement_folder} — {os.path.basename(filepath)}\n"
        f"Grey = skipped  |  Coloured = kept",
        fontsize=12
    )

    for i in range(n_total):
        start  = i * ws
        end    = start + ws
        kept   = i in kept_indices
        color_h = "steelblue" if kept else "lightgrey"
        color_v = "tomato"    if kept else "lightgrey"
        status  = "KEPT" if kept else "SKIPPED"

        axes[i, 0].plot(h[start:end], color=color_h)
        axes[i, 0].set_title(f"Window {i} [{status}] — H channel", fontsize=9)
        axes[i, 0].axhline(0, color="gray", linewidth=0.5)

        axes[i, 1].plot(v[start:end], color=color_v)
        axes[i, 1].set_title(f"Window {i} [{status}] — V channel", fontsize=9)
        axes[i, 1].axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    out = f"window_check_{movement_folder}_trial{trial_index}.png"
    plt.savefig(out, dpi=100)
    plt.show()

    print(f"Saved: {out}")
    print(f"Total windows in trial : {n_total}")
    print(f"Windows skipped        : {n_total - len(kept_indices)}  (indices: {[i for i in range(n_total) if i not in kept_indices]})")
    print(f"Windows kept           : {len(kept_indices)}  (indices: {kept_indices})")
    print(f"\nIf the last kept window looks like an abrupt cut-off, set SKIP_LAST_WINDOW = True and re-run.")


# ─────────────────────────────────────────────
# STEP 1b — REMOVE BAD WINDOWS AND SAVE CLEANED CSV
# ─────────────────────────────────────────────

def remove_windows_from_trial(
    subject_folder,
    movement_folder,
    trial_index,
    remove_indices,
    overwrite=False,
):
    """
    Removes specific windows from a raw trial CSV and saves the cleaned file.

    Window indices correspond exactly to those shown by verify_all_windows()
    (0 = first window, 1 = second window, …).

    The sample rows for each removed window are dropped from the DataFrame.
    Because preprocessing extracts windows sequentially from whatever rows
    remain, the pipeline needs no changes — bad windows simply no longer exist
    in the file.

    Args:
        subject_folder  : path to the subject folder
                          e.g. "EOG Data/Subject_01"
        movement_folder : class subfolder name
                          e.g. "Up", "Left", "Blink"
        trial_index     : 0-based index of the trial file to clean
                          (0 = trial_1.csv, 1 = trial_2.csv, …)
        remove_indices  : list of window indices (as shown in verify_all_windows)
                          to remove, e.g. [3, 5]
        overwrite       : if True  → overwrites the original CSV in place
                          if False → saves as <original_name>_cleaned.csv
                          alongside the original (safer default)

    Usage:
        # Inspect first
        verify_all_windows("EOG Data/Subject_01", movement_folder="Up", trial_index=0)

        # Then remove windows 3 and 5 from that same trial
        remove_windows_from_trial(
            subject_folder  = "EOG Data/Subject_01",
            movement_folder = "Up",
            trial_index     = 0,
            remove_indices  = [3, 5],
        )
    """
    if not remove_indices:
        print("[INFO] remove_indices is empty — nothing to do.")
        return

    movement_path = os.path.join(subject_folder, movement_folder)
    trial_files   = sorted(glob.glob(os.path.join(movement_path, "*.csv")))

    if not trial_files:
        print(f"[ERROR] No CSV files found in: {movement_path}")
        return

    if trial_index >= len(trial_files):
        print(f"[ERROR] trial_index={trial_index} is out of range "
              f"({len(trial_files)} trial(s) found).")
        return

    filepath = trial_files[trial_index]
    df       = pd.read_csv(filepath)
    ws       = WINDOW_SEC * SAMPLING_RATE          # 256 samples per window
    n_total  = len(df) // ws

    # Validate requested indices
    invalid = [i for i in remove_indices if i < 0 or i >= n_total]
    if invalid:
        print(f"[ERROR] The following window indices are out of range "
              f"(0–{n_total - 1}): {invalid}")
        return

    remove_indices = sorted(set(remove_indices))   # deduplicate and sort

    # Build the set of row indices to drop
    rows_to_drop = []
    for win_idx in remove_indices:
        start = win_idx * ws
        end   = start + ws                         # exclusive
        rows_to_drop.extend(range(start, end))

    df_cleaned = df.drop(index=rows_to_drop).reset_index(drop=True)

    # Decide output path
    if overwrite:
        out_path = filepath
    else:
        base, ext = os.path.splitext(filepath)
        out_path  = f"{base}_cleaned{ext}"

    df_cleaned.to_csv(out_path, index=False)

    # ── Summary ────────────────────────────────────────────────────────
    removed_count = len(rows_to_drop)
    print(f"\nFile     : {filepath}")
    print(f"Removed  : windows {remove_indices}  ({removed_count} rows / "
          f"{len(remove_indices)} window(s))")
    print(f"Before   : {len(df)} rows  ({n_total} windows)")
    print(f"After    : {len(df_cleaned)} rows  ({len(df_cleaned) // ws} windows)")
    print(f"Saved to : {out_path}")

    # ── Plot: before vs after ──────────────────────────────────────────
    h_before = df["h_signal"].values
    v_before = df["v_signal"].values
    h_after  = df_cleaned["h_signal"].values
    v_after  = df_cleaned["v_signal"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    fig.suptitle(
        f"Before vs After window removal — {movement_folder} — "
        f"{os.path.basename(filepath)}\n"
        f"Removed window(s): {remove_indices}",
        fontsize=11
    )

    axes[0, 0].plot(h_before, color="steelblue", linewidth=0.7)
    axes[0, 0].set_title("BEFORE — H channel")
    axes[0, 0].axhline(0, color="gray", linewidth=0.5)

    # Shade the removed windows on the before plot for easy identification
    for win_idx in remove_indices:
        axes[0, 0].axvspan(win_idx * ws, (win_idx + 1) * ws,
                           color="red", alpha=0.2, label=f"win {win_idx}")
        axes[0, 1].axvspan(win_idx * ws, (win_idx + 1) * ws,
                           color="red", alpha=0.2)

    axes[0, 1].plot(v_before, color="tomato", linewidth=0.7)
    axes[0, 1].set_title("BEFORE — V channel")
    axes[0, 1].axhline(0, color="gray", linewidth=0.5)

    axes[1, 0].plot(h_after, color="steelblue", linewidth=0.7)
    axes[1, 0].set_title("AFTER — H channel")
    axes[1, 0].axhline(0, color="gray", linewidth=0.5)

    axes[1, 1].plot(v_after, color="tomato", linewidth=0.7)
    axes[1, 1].set_title("AFTER — V channel")
    axes[1, 1].axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    out_fig = (f"removal_check_{movement_folder}_trial{trial_index}"
               f"_removed{'_'.join(str(i) for i in remove_indices)}.png")
    plt.savefig(out_fig, dpi=100)
    plt.show()
    print(f"Plot saved: {out_fig}")


# ─────────────────────────────────────────────
# STEP 2 — LOAD AND EXTRACT WINDOWS FROM ONE TRIAL
# ─────────────────────────────────────────────

def extract_windows_from_trial(filepath, folder_name):
    """
    Loads a single trial CSV and extracts windows.

    Directional classes:
        - keep only movement windows (strict mode)

    Blink + No_Movement:
        - skip first baseline window
        - keep all remaining windows
    """

    df = pd.read_csv(filepath)

    h = df["h_signal"].values
    v = df["v_signal"].values

    ws = WINDOW_SEC * SAMPLING_RATE
    n_total_windows = len(h) // ws

    windows = []

    # ─────────────────────────────────────────
    # Blink + No Movement
    # ─────────────────────────────────────────
    if folder_name in ["Blink", "No_Movement"]:

        # Skip first baseline window
        for i in range(1, n_total_windows):

            start = i * ws
            end   = start + ws

            w = np.stack([h[start:end], v[start:end]], axis=-1)
            windows.append(w)

    # ─────────────────────────────────────────
    # Directional movement classes
    # ─────────────────────────────────────────
    else:

        for i in range(n_total_windows):

            if i % 2 == MOVEMENT_WINDOW_INDEX:

                start = i * ws
                end   = start + ws

                w = np.stack([h[start:end], v[start:end]], axis=-1)
                windows.append(w)

    return np.array(windows, dtype=np.float32) # shape (n_windows, 256, 2)

def extract_windows_from_trial_with_return(filepath, folder_name):
    """
    Extended version of extract_windows_from_trial.
    
    Directional classes (Up, Down, Left, Right):
        - Odd windows  → intended movement (original behaviour)
        - Even windows → return movement, labelled as opposite direction
        - Window 0     → skipped (pure baseline)

    Blink + No_Movement:
        - Same as original — skip first window, keep the rest

    Returns:
        windows : np.ndarray of shape (n_windows, 256, 2)  ← same as before
        labels  : np.ndarray of shape (n_windows,)         ← integer labels
    """

    df = pd.read_csv(filepath)
    h  = df["h_signal"].values
    v  = df["v_signal"].values

    ws              = WINDOW_SEC * SAMPLING_RATE
    n_total_windows = len(h) // ws

    windows = []
    labels  = []

    # ── Blink and No Movement — unchanged from original ───────────────────
    if folder_name in ["Blink", "No_Movement", "No Movement"]:
        intended_label = LABEL_MAP[FOLDER_TO_LABEL[folder_name]]
        for i in range(1, n_total_windows):
            start = i * ws
            end   = start + ws
            w     = np.stack([h[start:end], v[start:end]], axis=-1)
            windows.append(w)
            labels.append(intended_label)

    # ── Directional classes — collect both intended and return ────────────
    else:
        intended_label = LABEL_MAP[FOLDER_TO_LABEL[folder_name]]
        opposite_key   = OPPOSITE_CLASS.get(folder_name)

        for i in range(n_total_windows):

            if i == 0:
                # Skip first window — pure baseline before any movement
                continue

            start = i * ws
            end   = start + ws
            w     = np.stack([h[start:end], v[start:end]], axis=-1)

            if i % 2 == MOVEMENT_WINDOW_INDEX:
                # Intended movement window
                windows.append(w)
                labels.append(intended_label)
            elif opposite_key is not None:
                # Return window — only for Left and Right
                windows.append(w)
                labels.append(LABEL_MAP[opposite_key])

    return (
        np.array(windows, dtype=np.float32),  # (n_windows, 256, 2)
        np.array(labels,  dtype=np.int32)     # (n_windows,)
    )

# ─────────────────────────────────────────────
# STEP 3 — PER-TRIAL Z-SCORE NORMALIZATION
# ─────────────────────────────────────────────

def normalize_trial_windows(windows):
    """
    Z-score normalize per channel across the trial's windows.
    Computes mean and std from the entire trial (all windows together),
    then normalizes each window with those values.

    Args:
        windows : np.ndarray of shape (n_windows, 256, 2)

    Returns:
        normalized : np.ndarray of same shape
        params     : dict with mean and std per channel (save these for Android)
    """
    normalized = windows.copy().astype(np.float32)
    params = {}

    for ch in range(2):
        ch_data = windows[:, :, ch]       # shape (n_windows, 256)
        mean    = ch_data.mean()
        std     = ch_data.std()

        normalized[:, :, ch] = (ch_data - mean) / (std + 1e-8)
        params[f"ch{ch}_mean"] = mean
        params[f"ch{ch}_std"]  = std

    return normalized, params


# ─────────────────────────────────────────────
# STEP 4 — PROCESS ONE SUBJECT
# ─────────────────────────────────────────────

def process_subject(subject_path):

    X, y, trial_ids, class_names = [], [], [], []

    for folder_name, label_key in FOLDER_TO_LABEL.items():
        movement_path = os.path.join(subject_path, folder_name)

        if not os.path.exists(movement_path):
            print(f"  [WARNING] Folder not found, skipping: {movement_path}")
            continue

        trial_files = sorted(glob.glob(os.path.join(movement_path, "*.csv")))

        if not trial_files:
            print(f"  [WARNING] No CSV files in: {movement_path}")
            continue

        for trial_num, filepath in enumerate(trial_files, start=1):

            # ── Unpack windows and labels together ─────────────────────────
            windows, window_labels = \
                extract_windows_from_trial_with_return(filepath, folder_name)

            if len(windows) == 0:
                print(f"  [WARNING] No windows extracted from: {filepath}")
                continue

            # ── Normalize (same as before) ─────────────────────────────────
            windows_norm, norm_params = normalize_trial_windows(windows)

            n = len(windows_norm)
            X.append(windows_norm)
            y.extend(window_labels.tolist())   # ← use returned labels directly
            trial_ids.extend([trial_num] * n)
            class_names.extend([
                list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(lbl)]
                for lbl in window_labels
            ])

            print(f"  {folder_name} | Trial {trial_num} | {n} windows | "
                  f"H: mean={norm_params['ch0_mean']:.3f} "
                  f"std={norm_params['ch0_std']:.3f}")

    X           = np.concatenate(X, axis=0)
    y           = np.array(y,         dtype=np.int32)
    trial_ids   = np.array(trial_ids, dtype=np.int32)
    class_names = np.array(class_names)

    return X, y, trial_ids, class_names


# ─────────────────────────────────────────────
# STEP 5 — PROCESS ALL SUBJECTS AND SAVE
# ─────────────────────────────────────────────

def preprocess_all():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    subject_folders = sorted([
        f for f in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, f))
    ])

    if not subject_folders:
        print(f"No subject folders found in: {RAW_DATA_DIR}")
        return

    summary = []

    for subj_folder in subject_folders:
        subj_path = os.path.join(RAW_DATA_DIR, subj_folder)
        print(f"\nProcessing: {subj_folder}")

        X, y, trial_ids, class_names = process_subject(subj_path)

        # Save as compressed numpy file
        save_path = os.path.join(PROCESSED_DIR, f"{subj_folder.lower()}.npz")
        np.savez_compressed(
            save_path,
            X=X,
            y=y,
            trial=trial_ids,
            class_name=class_names
        )

        print(f"  → Saved: {save_path} | Total windows: {len(X)}")
        summary.append((subj_folder, len(X), X.shape))

    # Print summary
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"{'Subject':<15} {'Windows':<10} {'Shape'}")
    for subj, n_windows, shape in summary:
        print(f"{subj:<15} {n_windows:<10} {shape}")
    print(f"\nAll files saved to: {PROCESSED_DIR}/")


# ─────────────────────────────────────────────
# QUICK SANITY CHECK — run after preprocessing
# ─────────────────────────────────────────────

def sanity_check(subject_npz_path):
    """
    Loads a processed .npz and prints a summary.
    Also plots one window per class to visually verify.

    Usage:
        sanity_check("processed/subject_01.npz")
    """
    data        = np.load(subject_npz_path, allow_pickle=True)
    X           = data["X"]
    y           = data["y"]
    class_names = data["class_name"]

    print(f"\nSanity check: {subject_npz_path}")
    print(f"  X shape      : {X.shape}   (windows, timesteps, channels)")
    print(f"  y shape      : {y.shape}")
    print(f"  dtype        : {X.dtype}")
    print(f"  y range      : {y.min()} – {y.max()}")
    print(f"\n  Class distribution:")

    for cls_name, cls_int in LABEL_MAP.items():
        count = (y == cls_int).sum()
        print(f"    {cls_name:<15} label={cls_int}  count={count}")

    # Plot one sample window per class
    fig, axes = plt.subplots(len(LABEL_MAP), 2, figsize=(12, len(LABEL_MAP) * 2.5))
    fig.suptitle(f"One sample window per class — {os.path.basename(subject_npz_path)}", fontsize=13)

    for row, (cls_name, cls_int) in enumerate(LABEL_MAP.items()):
        indices = np.where(y == cls_int)[0]
        if len(indices) == 0:
            axes[row, 0].set_title(f"{cls_name} — NO SAMPLES", color="red")
            continue

        sample = X[indices[0]]  # shape (256, 2)
        axes[row, 0].plot(sample[:, 0], color="steelblue")
        axes[row, 0].set_title(f"{cls_name} — H channel")
        axes[row, 0].axhline(0, color="gray", linewidth=0.5)

        axes[row, 1].plot(sample[:, 1], color="tomato")
        axes[row, 1].set_title(f"{cls_name} — V channel")
        axes[row, 1].axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=100)
    plt.show()
    print("\nSaved: sanity_check.png")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── STEP 1a: Run this first to confirm window alignment ──────────────
    # Uncomment and run. Look at the plot and identify which window index
    # (0, 2, 4... OR 1, 3, 5...) contains the movement deflection.
    # Then update MOVEMENT_WINDOW_INDEX at the top of this file.
    #
    # verify_all_windows("d:/Documents/Studies/MTech/MTP/EOG Model/raw_data/Subject_03", movement_folder="Right", trial_index=0)
    
    # ── STEP 1b: Remove bad windows from a specific trial ───────────────
    # First run verify_all_windows() to identify the bad window indices,
    # then pass them here.
    # By default saves as trial_1_cleaned.csv alongside the original.
    # Set overwrite=True to replace the original file in place.
    #
    # remove_windows_from_trial(
    #      subject_folder  = "d:/Documents/Studies/MTech/MTP/EOG Model/raw_data/Subject_03",
    #      movement_folder = "Right",
    #      trial_index     = 0,       # 0 = trial_1.csv, 1 = trial_2.csv, ...
    #      remove_indices  = [5,6,14],  # window indices as shown in the plot
    #      overwrite       = True,   # set True to replace original
    # )

    # ── STEP 2: Preprocess all subjects ─────────────────────────────────
    preprocess_all()

    # ── STEP 3: Sanity check one subject ────────────────────────────────
    # sanity_check("d:/Documents/Studies/MTech/MTP/EOG Model/processed_strict/subject_11.npz")
