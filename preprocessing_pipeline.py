# =============================================================================
# Preprocessing Pipeline — Dating App Behaviour Dataset
# WIA1006/WID3006 Machine Learning Assignment
# =============================================================================
# Pipeline overview
# ─────────────────────────────────────────────────────────────────────────────
#  DROPPED   : app_usage_time_label, swipe_right_label  ← derived from numeric
#              columns → would cause data leakage
#
#  NUMERICAL : app_usage_time_min, swipe_right_ratio, likes_received,
#              mutual_matches, profile_pics_count, bio_length,
#              message_sent_count, emoji_usage_rate, last_active_hour
#              → median impute → StandardScaler
#
#  ORDINAL   : income_bracket  (Very Low … Very High, 7 levels)
#              education_level (No Formal … Postdoc, 9 levels)
#              → mode impute → OrdinalEncoder
#
#  NOMINAL   : gender, sexual_orientation, location_type, swipe_time_of_day
#              → mode impute → OneHotEncoder
#
#  MULTI-TAG : interest_tags  (comma-separated free text)
#              → custom MultiLabelBinarizerTransformer
#
#  TARGET    : match_outcome  (10 classes)
#              → LabelEncoder  (integer codes for sklearn estimators)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.compose         import ColumnTransformer
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import (StandardScaler, OrdinalEncoder,
                                     OneHotEncoder, LabelEncoder)
from sklearn.decomposition   import PCA
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid", font_scale=1.1)
RANDOM_STATE = 42


# =============================================================================
# 0.  LOAD DATA
# =============================================================================
df = pd.read_csv("dating_app_behavior_dataset.csv")
print(f"Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")


# =============================================================================
# 1.  COLUMN DEFINITIONS
# =============================================================================

# Columns that are pure re-codings of a numeric partner → drop to avoid leakage
LEAKY_COLS = ["app_usage_time_label", "swipe_right_label"]

TARGET_COL  = "match_outcome"
MULTITAG_COL = "interest_tags"

NUMERIC_COLS = [
    "app_usage_time_min", "swipe_right_ratio", "likes_received",
    "mutual_matches",     "profile_pics_count", "bio_length",
    "message_sent_count", "emoji_usage_rate",   "last_active_hour",
]

# Ordinal columns with their LOW→HIGH ordering
ORDINAL_COLS = ["income_bracket", "education_level"]

INCOME_ORDER = [
    "Very Low", "Low", "Lower-Middle", "Middle",
    "Upper-Middle", "High", "Very High"
]
EDUCATION_ORDER = [
    "No Formal Education", "High School", "Diploma",
    "Associate's", "Bachelor's", "MBA",
    "Master's", "PhD", "Postdoc"
]

NOMINAL_COLS = [
    "gender", "sexual_orientation", "location_type", "swipe_time_of_day"
]


# =============================================================================
# 2.  CUSTOM TRANSFORMER — MultiLabelBinarizer for interest_tags
# =============================================================================

class MultiTagBinarizer(BaseEstimator, TransformerMixin):
    """
    Splits comma-separated tag strings and produces a binary indicator
    matrix.  Handles unseen tags in transform() gracefully.

    Parameters
    ----------
    sep : str
        Delimiter used between tags (default ',').
    min_freq : int
        Drop tags that appear in fewer than this many rows (helps reduce
        dimensionality from very rare interest combinations).
    """

    def __init__(self, sep: str = ",", min_freq: int = 50):
        self.sep      = sep
        self.min_freq = min_freq

    # ── helpers ──────────────────────────────────────────────────────────────
    def _split(self, series: pd.Series) -> list[list[str]]:
        return [
            [t.strip() for t in str(v).split(self.sep) if t.strip()]
            for v in series
        ]

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, X, y=None):
        col = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(X.flatten())
        split = self._split(col)

        # Count tag frequencies
        counter = {}
        for tags in split:
            for t in tags:
                counter[t] = counter.get(t, 0) + 1

        self.classes_ = sorted(
            t for t, freq in counter.items() if freq >= self.min_freq
        )
        self.tag_index_ = {t: i for i, t in enumerate(self.classes_)}
        return self

    # ── transform ────────────────────────────────────────────────────────────
    def transform(self, X, y=None):
        col   = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(X.flatten())
        split = self._split(col)
        mat   = np.zeros((len(split), len(self.classes_)), dtype=np.float32)
        for i, tags in enumerate(split):
            for t in tags:
                if t in self.tag_index_:
                    mat[i, self.tag_index_[t]] = 1.0
        return mat

    def get_feature_names_out(self, input_features=None):
        return np.array([f"tag_{t.replace(' ', '_')}" for t in self.classes_])


# =============================================================================
# 3.  FEATURE / TARGET SPLIT
# =============================================================================
X_raw = df.drop(columns=[TARGET_COL] + LEAKY_COLS)
y_raw = df[TARGET_COL]

# ── encode target ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(y_raw)

print("\nTarget classes (encoded → original):")
for code, label in enumerate(le.classes_):
    print(f"  {code:2d}  →  {label}")


# =============================================================================
# 4.  SUB-PIPELINES FOR EACH FEATURE TYPE
# =============================================================================

# ── 4a. Numerical: median impute → standard scale ─────────────────────────────
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

# ── 4b. Ordinal: mode impute → ordinal encode ─────────────────────────────────
ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[INCOME_ORDER, EDUCATION_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,            # unseen category → -1 sentinel
    )),
])

# ── 4c. Nominal: mode impute → one-hot encode ────────────────────────────────
nominal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",     # unseen category → all-zero row
        sparse_output=False,
        drop=None,                   # keep all dummies for interpretability
    )),
])

# ── 4d. Multi-tag: custom binarizer (fit-only, no impute needed) ──────────────
multitag_pipeline = Pipeline(steps=[
    ("binarizer", MultiTagBinarizer(sep=",", min_freq=50)),
])


# =============================================================================
# 5.  COLUMN TRANSFORMER — assemble all sub-pipelines
# =============================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num",      numeric_pipeline,  NUMERIC_COLS),
        ("ord",      ordinal_pipeline,  ORDINAL_COLS),
        ("nom",      nominal_pipeline,  NOMINAL_COLS),
        ("multitag", multitag_pipeline, [MULTITAG_COL]),
    ],
    remainder="drop",                  # discard any unlisted columns
    verbose_feature_names_out=True,
)


# =============================================================================
# 6.  FIT & TRANSFORM (on full X before split — for PCA analysis only)
# =============================================================================
print("\nFitting ColumnTransformer …")
X_processed = preprocessor.fit_transform(X_raw)

# ── Retrieve feature names ────────────────────────────────────────────────────
feature_names = preprocessor.get_feature_names_out()
print(f"Features after preprocessing : {X_processed.shape[1]}")
print(f"  numeric   : {len(NUMERIC_COLS)}")
print(f"  ordinal   : {len(ORDINAL_COLS)}")
nominal_features = preprocessor.named_transformers_["nom"]["encoder"] \
                               .get_feature_names_out(NOMINAL_COLS)
print(f"  one-hot   : {len(nominal_features)}")
tag_features = preprocessor.named_transformers_["multitag"]["binarizer"] \
                            .get_feature_names_out()
print(f"  multi-tag : {len(tag_features)}")


# =============================================================================
# 7.  TRAIN / TEST SPLIT  (80 / 20, stratified)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,                       # preserve class proportions
)

print(f"\nTrain set : {X_train.shape[0]:,} samples  "
      f"({X_train.shape[0]/len(y)*100:.0f}%)")
print(f"Test  set : {X_test.shape[0]:,} samples  "
      f"({X_test.shape[0]/len(y)*100:.0f}%)")

# Verify stratification
print("\nClass distribution — train vs test (%):")
train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index() * 100
test_dist  = pd.Series(y_test ).value_counts(normalize=True).sort_index() * 100
dist_check = pd.DataFrame({"train_%": train_dist.round(2),
                            "test_%":  test_dist.round(2),
                            "label":   le.classes_})
print(dist_check.to_string(index=True))


# =============================================================================
# 8.  PCA — EXPLAINED VARIANCE ANALYSIS
# =============================================================================
print("\nFitting full PCA for explained variance …")
pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_train)                 # fit on training split only

ev_ratio  = pca_full.explained_variance_ratio_
ev_cumsum = np.cumsum(ev_ratio)

# Find thresholds
thresh = {0.80: None, 0.90: None, 0.95: None, 0.99: None}
for t in thresh:
    thresh[t] = int(np.argmax(ev_cumsum >= t)) + 1

print("\nComponents needed to reach variance thresholds:")
for t, n in thresh.items():
    print(f"  {int(t*100):3d}%  →  {n:3d} components")

OPTIMAL_N = thresh[0.95]             # 95 % is the standard choice
print(f"\nSelected  : {OPTIMAL_N} components  (95% explained variance)")


# ── Plot 1: Scree + Cumulative Variance ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("PCA — Explained Variance Analysis", fontsize=14, fontweight="bold")

# Individual explained variance (scree)
n_show = min(60, len(ev_ratio))
axes[0].bar(range(1, n_show + 1), ev_ratio[:n_show] * 100,
            color="#4C72B0", alpha=0.8, width=0.8)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance (%)")
axes[0].set_title(f"Scree Plot (first {n_show} PCs)")
axes[0].axvline(OPTIMAL_N, color="red", linestyle="--",
                label=f"n={OPTIMAL_N} (95%)")
axes[0].legend()

# Cumulative explained variance
axes[1].plot(range(1, len(ev_cumsum) + 1), ev_cumsum * 100,
             color="#4C72B0", linewidth=2)
axes[1].fill_between(range(1, len(ev_cumsum) + 1), ev_cumsum * 100,
                     alpha=0.15, color="#4C72B0")
for t, n in thresh.items():
    axes[1].axhline(t * 100, linestyle=":", alpha=0.6,
                    label=f"{int(t*100)}% @ n={n}")
    axes[1].axvline(n, linestyle=":", alpha=0.4)
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance (%)")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend(fontsize=9)
axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

plt.tight_layout()
plt.savefig("pca_explained_variance.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# ── Plot 2: PCA 2D scatter (first 2 PCs), coloured by target ─────────────────
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_2d = pca_2d.fit_transform(X_train)

fig, ax = plt.subplots(figsize=(10, 7))
palette = sns.color_palette("tab10", len(le.classes_))
for cls_idx, cls_label in enumerate(le.classes_):
    mask = y_train == cls_idx
    ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
               s=8, alpha=0.35, color=palette[cls_idx], label=cls_label)
ax.set_xlabel(f"PC 1  ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC 2  ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA 2D Projection — Training Set (coloured by match_outcome)",
             fontsize=13, fontweight="bold")
ax.legend(markerscale=3, fontsize=9, bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig("pca_2d_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()


# =============================================================================
# 9.  FINAL PCA REDUCTION  (apply chosen n_components)
# =============================================================================
pca_final = PCA(n_components=OPTIMAL_N, random_state=RANDOM_STATE)
X_train_pca = pca_final.fit_transform(X_train)   # fit ONLY on train
X_test_pca  = pca_final.transform(X_test)         # apply to test

print(f"\nDimensionality after PCA:")
print(f"  Before : {X_train.shape[1]} features")
print(f"  After  : {X_train_pca.shape[1]} components")
print(f"  Variance retained : "
      f"{pca_final.explained_variance_ratio_.sum()*100:.2f}%")


# =============================================================================
# 10.  EXPORT — ready-to-use objects for modelling notebook
# =============================================================================
import pickle, os

artifacts = {
    "preprocessor":  preprocessor,     # ColumnTransformer (fitted on full X)
    "pca":           pca_final,        # PCA fitted on X_train
    "label_encoder": le,               # LabelEncoder for target
    "X_train":       X_train,          # preprocessed, pre-PCA
    "X_test":        X_test,
    "X_train_pca":   X_train_pca,      # preprocessed + PCA-reduced
    "X_test_pca":    X_test_pca,
    "y_train":       y_train,
    "y_test":        y_test,
    "feature_names": feature_names,    # names after ColumnTransformer
}

with open("preprocessing_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\n✅  Artifacts saved to  preprocessing_artifacts.pkl")
print("    Keys:", list(artifacts.keys()))
print("\nPreprocessing pipeline complete.")


# =============================================================================
# 11.  PIPELINE ARCHITECTURE SUMMARY
# =============================================================================
print("""
╔══════════════════════════════════════════════════════════════════════╗
║              PREPROCESSING PIPELINE — ARCHITECTURE                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  RAW INPUT                                                          ║
║  ├─ DROP  : app_usage_time_label, swipe_right_label (leaky)         ║
║  │                                                                   ║
║  ├─ NUMERIC (9 cols)                                                 ║
║  │   └─ SimpleImputer(median) → StandardScaler                      ║
║  │                                                                   ║
║  ├─ ORDINAL (2 cols)                                                 ║
║  │   ├─ income_bracket  : Very Low … Very High (7 levels)           ║
║  │   └─ education_level : No Formal … Postdoc   (9 levels)          ║
║  │   └─ SimpleImputer(mode) → OrdinalEncoder                        ║
║  │                                                                   ║
║  ├─ NOMINAL (4 cols)                                                 ║
║  │   └─ SimpleImputer(mode) → OneHotEncoder(drop=None)              ║
║  │                                                                   ║
║  └─ MULTI-TAG (1 col : interest_tags)                               ║
║      └─ MultiTagBinarizer(sep=',', min_freq=50)                     ║
║                                                                      ║
║  COLUMN TRANSFORMER (parallel) → DENSE MATRIX                       ║
║                                                                      ║
║  TRAIN / TEST SPLIT  80/20  stratified on match_outcome             ║
║                                                                      ║
║  PCA  (fit on X_train only, transform both)                         ║
║  └─ n_components = 95% explained variance                            ║
╚══════════════════════════════════════════════════════════════════════╝
""")
