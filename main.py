import kagglehub
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
from collections import defaultdict



from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# TORGO dysarthria baseline:
# - Downloads the dataset via kagglehub
# - Runs lightweight dataset checks (speaker overlap, mic distribution, random quality scan)
# - Extracts MFCC summary features per utterance
# - Trains and evaluates an SVM classifier (file-level split) as an initial baseline
# - Safely load a wav file at a fixed sample rate.
# - Returns (None, None) if loading fails or the file is empty.


def safe_load_wav(fp: str):
    try:
        y, sr = librosa.load(fp, sr=16000, mono=True)
        if y is None or sr is None:
            return None, None
        if len(y) == 0:
            return None, None
        return y, sr
    except Exception:
        return None, None

# Extract 13 MFCCs per file and summarize over time using mean and standard deviation.
# Final feature dimension: 26 per utterance.
def extract_mfcc_features(fp: str, n_mfcc: int = 13):
    y, sr = safe_load_wav(fp)
    if y is None or sr is None:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def main():
    path = kagglehub.dataset_download("pranaykoppula/torgo-audio")
    print("Dataset path:", path)

    items = sorted(os.listdir(path))
    print(f"Top-level items ({len(items)}):")
    for item in items:
        print(" -", item)


    for group in ["F_Con", "F_Dys", "M_Con", "M_Dys"]:
        group_path = os.path.join(path, group)
        wavs = []
        for root, _, files in os.walk(group_path):
            for f in files:
                if f.lower().endswith(".wav"):
                    wavs.append(os.path.join(root, f))
        print(group, "wav files:", len(wavs))

    # Dataset-level metadata checks (no audio decoding).
    # Used to verify speaker separation and inspect microphone distributions.
    print("\n=== Confound checks (dataset-level) ===")


    # Speaker ID is encoded in folder names like: wav_headMic_FC02S03 (speaker FC02, session S03).
    speaker_re = re.compile(r"wav_(?:headMic|arrayMic)_(?P<code>[FM]C?\d{2})S\d{2}")

    # Extract speaker ID from directory names (e.g., wav_headMic_FC02S03 → FC02).
    def parse_speaker_id_from_path(fp: str) -> str | None:
        # Expected folder: .../<GROUP>/wav_headMic_FC02S03/<file>.wav
        parts = fp.split(os.sep)
        for p in parts:
            m = speaker_re.match(p)
            if m:
                return m.group("code")
        return None

    # Infer microphone type from the directory name.
    def parse_mic_from_path(fp: str) -> str | None:
        # Returns 'headMic' or 'arrayMic' if present
        if f"wav_headMic_" in fp:
            return "headMic"
        if f"wav_arrayMic_" in fp:
            return "arrayMic"
        return None

    # Collect speakers and mic types per group (fast, no audio loading)
    group_to_speakers: dict[str, set[str]] = {}
    group_to_mics: dict[str, dict[str, int]] = {}
    group_to_files: dict[str, list[str]] = {}

    for group in ["F_Con", "F_Dys", "M_Con", "M_Dys"]:
        group_path = os.path.join(path, group)
        speakers = set()
        mic_counts = {"headMic": 0, "arrayMic": 0, "unknown": 0}
        files = []

        for root, _, fs in os.walk(group_path):
            for f in fs:
                if not f.lower().endswith(".wav"):
                    continue
                fp = os.path.join(root, f)
                files.append(fp)

                spk = parse_speaker_id_from_path(fp)
                if spk is not None:
                    speakers.add(spk)

                mic = parse_mic_from_path(fp)
                if mic is None:
                    mic_counts["unknown"] += 1
                else:
                    mic_counts[mic] += 1

        group_to_speakers[group] = speakers
        group_to_mics[group] = mic_counts
        group_to_files[group] = files

        print(f"{group}: speakers {len(speakers)} | mic counts {mic_counts}")

    # Check speaker overlap across label groups
    all_groups = ["F_Con", "F_Dys", "M_Con", "M_Dys"]
    speaker_to_groups: dict[str, set[str]] = {}
    for g in all_groups:
        for spk in group_to_speakers[g]:
            speaker_to_groups.setdefault(spk, set()).add(g)

    multi_group = {spk: gs for spk, gs in speaker_to_groups.items() if len(gs) > 1}

    if not multi_group:
        print("OK: No speaker IDs appear in more than one group (no obvious label mixing).")
    else:
        print("WARNING: Some speaker IDs appear in multiple groups (possible label mixing):")
        # Print up to 20 for readability
        for spk, gs in list(sorted(multi_group.items()))[:20]:
            print(" -", spk, "->", sorted(gs))

    # Also check within-sex overlap (Control vs Dys)
    f_overlap = group_to_speakers["F_Con"].intersection(group_to_speakers["F_Dys"])
    m_overlap = group_to_speakers["M_Con"].intersection(group_to_speakers["M_Dys"])
    print(f"Female Con∩Dys overlap: {len(f_overlap)}")
    if f_overlap:
        print("  examples:", sorted(list(f_overlap))[:10])
    print(f"Male Con∩Dys overlap: {len(m_overlap)}")
    if m_overlap:
        print("  examples:", sorted(list(m_overlap))[:10])

    # --- Data quality scan (random sample) ---
    RANDOM_SEED = 42
    SAMPLE_PER_GROUP = 50
    MIN_DUR_S = 0.5
    MAX_DUR_S = 20.0
    SILENCE_RMS = 1e-3
    CLIP_FRAC = 0.001  # fraction of samples near full scale

    random.seed(RANDOM_SEED)

    print("\n=== Data quality scan (random sample) ===")

    for group in ["F_Con", "F_Dys", "M_Con", "M_Dys"]:
        wavs = group_to_files.get(group, [])
        if not wavs:
            print(f"{group}: no wav files found")
            continue

        sample = random.sample(wavs, k=min(SAMPLE_PER_GROUP, len(wavs)))

        failed_load = 0
        too_short = 0
        too_long = 0
        near_silent = 0
        clipped = 0

        durations = []
        srs = []

        flagged_examples = []  # (issue, filepath)

        for fp in sample:
            y_, sr_ = safe_load_wav(fp)
            if y_ is None:
                failed_load += 1
                flagged_examples.append(("failed_load", fp))
                continue

            dur = len(y_) / float(sr_)
            rms = float(np.sqrt(np.mean(y_ * y_)))
            clip_fraction = float(np.mean(np.abs(y_) >= 0.999))

            durations.append(dur)
            srs.append(sr_)

            reasons = []
            if dur < MIN_DUR_S:
                too_short += 1
                reasons.append("too_short")
            if dur > MAX_DUR_S:
                too_long += 1
                reasons.append("too_long")
            if rms < SILENCE_RMS:
                near_silent += 1
                reasons.append("near_silent")
            if clip_fraction > CLIP_FRAC:
                clipped += 1
                reasons.append("clipped")

            if reasons:
                flagged_examples.append(("+".join(reasons), fp))

        if durations:
            d_min = float(np.min(durations))
            d_med = float(np.median(durations))
            d_max = float(np.max(durations))
        else:
            d_min = d_med = d_max = float("nan")

        sr_unique = sorted(set(srs))

        print(
            f"{group}: sampled {len(sample)} | failed {failed_load} | short {too_short} | long {too_long} | "
            f"silent {near_silent} | clipped {clipped} | dur(min/med/max) {d_min:.2f}/{d_med:.2f}/{d_max:.2f} | sr {sr_unique}"
        )

        if flagged_examples:
            print("  flagged examples:")
            for reason, fp in flagged_examples[:5]:
                print("   -", reason, "::", fp)

 

    # --- SVM Training and Evaluation ---
    print("\n=== SVM Training (MFCC features) ===")

    X = []
    y = []

    # Binary labels: 0 = control, 1 = dysarthric.
    label_map = {
        "F_Con": 0,
        "M_Con": 0,
        "F_Dys": 1,
        "M_Dys": 1
    }

    # Split data by speaker so that no speaker appears in both train and test.
    print("\n=== Speaker-level split ===")

    # Build a parallel speaker list aligned to X/y so we can split by speaker.
    speakers = []
    for group, files in group_to_files.items():
        for fp in files:
            spk = parse_speaker_id_from_path(fp)
            if spk is not None:
                speakers.append(spk)
            else:
                # Keep alignment by adding a placeholder; these rows may be skipped if features are None.
                speakers.append(None)

    # Rebuild feature, label, and speaker arrays together to ensure proper alignment
    # after skipping files with missing features or speaker IDs.
    X_list, y_list, spk_list = [], [], []
    for group, files in group_to_files.items():
        label = label_map[group]
        for fp in files:
            spk = parse_speaker_id_from_path(fp)
            feats = extract_mfcc_features(fp)
            if feats is None or spk is None:
                continue
            X_list.append(feats)
            y_list.append(label)
            spk_list.append(spk)

    X = np.array(X_list)
    y = np.array(y_list)
    speakers = np.array(spk_list)

    print("Feature matrix shape:", X.shape)

    # Map each speaker ID to the indices of its utterances.
    speaker_to_indices = defaultdict(list)
    for i, spk in enumerate(speakers):
        speaker_to_indices[spk].append(i)

    unique_speakers = list(speaker_to_indices.keys())
    random.seed(42)
    random.shuffle(unique_speakers)

    split_idx = int(0.8 * len(unique_speakers))
    train_speakers = set(unique_speakers[:split_idx])
    test_speakers = set(unique_speakers[split_idx:])

    train_idx = [i for spk in train_speakers for i in speaker_to_indices[spk]]
    test_idx = [i for spk in test_speakers for i in speaker_to_indices[spk]]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train speakers: {len(train_speakers)} | Test speakers: {len(test_speakers)}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    y_scores = svm.decision_function(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_scores))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()