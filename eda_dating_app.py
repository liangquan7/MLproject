# =============================================================================
# EDA — Dating App Behavior Dataset
# WIA1006/WID3006 Machine Learning Assignment
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE   = "muted"
FIG_DIR   = "."          # change to your output folder if needed
SAVE_FIGS = True         # set False to only display

def savefig(name):
    if SAVE_FIGS:
        plt.savefig(f"{FIG_DIR}/{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# =============================================================================
# 1. LOAD & BASIC INSPECTION
# =============================================================================
df = pd.read_csv("dating_app_behavior_dataset.csv")

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)

print("\n── HEAD ──")
print(df.head())

print("\n── DTYPES & NULL COUNTS ──")
info = pd.DataFrame({
    "dtype":     df.dtypes,
    "nulls":     df.isnull().sum(),
    "null_%":    (df.isnull().mean() * 100).round(2),
    "nunique":   df.nunique(),
})
print(info)

print("\n── NUMERICAL SUMMARY ──")
print(df.describe().T.round(2))

print("\n── CATEGORICAL FEATURE VALUE COUNTS ──")
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    print(f"\n{col}:\n{df[col].value_counts()}")


# =============================================================================
# 2. TARGET VARIABLE — match_outcome
# =============================================================================
TARGET = "match_outcome"
outcome_counts = df[TARGET].value_counts()
outcome_pct    = df[TARGET].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Target Variable: match_outcome Distribution", fontsize=14, fontweight="bold")

# Bar chart
bars = axes[0].barh(outcome_counts.index, outcome_counts.values,
                    color=sns.color_palette(PALETTE, len(outcome_counts)))
axes[0].set_xlabel("Count")
axes[0].set_title("Count per Outcome")
for bar, pct in zip(bars, outcome_pct.values):
    axes[0].text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va="center", fontsize=9)

# Pie chart
axes[1].pie(outcome_counts, labels=outcome_counts.index,
            autopct="%1.1f%%", startangle=140,
            colors=sns.color_palette(PALETTE, len(outcome_counts)))
axes[1].set_title("Proportion per Outcome")

plt.tight_layout()
savefig("01_target_distribution")

# Imbalance check
print("\n── TARGET CLASS BALANCE ──")
print(outcome_pct.round(2).to_string())
imbalance_ratio = outcome_counts.max() / outcome_counts.min()
print(f"\nMax/Min class ratio: {imbalance_ratio:.2f}  "
      f"({'BALANCED ✓' if imbalance_ratio < 1.5 else 'IMBALANCED ✗'})")


# =============================================================================
# 3. DEMOGRAPHIC FEATURES
# =============================================================================

# ── 3a. Gender ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Gender Distribution", fontsize=14, fontweight="bold")

gender_counts = df["gender"].value_counts()
axes[0].bar(gender_counts.index, gender_counts.values,
            color=sns.color_palette(PALETTE, len(gender_counts)))
axes[0].set_title("Overall Gender Count")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=20)

# Gender vs match_outcome (normalised stacked bar)
gender_outcome = (df.groupby(["gender", TARGET])
                    .size()
                    .unstack(fill_value=0))
gender_outcome_pct = gender_outcome.div(gender_outcome.sum(axis=1), axis=0) * 100
gender_outcome_pct.plot(kind="bar", stacked=True, ax=axes[1],
                        colormap="tab10", legend=True)
axes[1].set_title("Match Outcome by Gender (normalised)")
axes[1].set_ylabel("Proportion (%)")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend(bbox_to_anchor=(1.01, 1), fontsize=8)

plt.tight_layout()
savefig("02_gender_distribution")

# ── 3b. Sexual Orientation ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
orient_counts = df["sexual_orientation"].value_counts()
ax.bar(orient_counts.index, orient_counts.values,
       color=sns.color_palette(PALETTE, len(orient_counts)))
ax.set_title("Sexual Orientation Distribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
savefig("03_sexual_orientation")

# ── 3c. Income Bracket ────────────────────────────────────────────────────────
income_order = ["Very Low", "Low", "Lower-Middle", "Middle",
                "Upper-Middle", "High", "Very High"]
income_order = [i for i in income_order if i in df["income_bracket"].unique()]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Income Bracket Analysis", fontsize=14, fontweight="bold")

income_counts = df["income_bracket"].value_counts().reindex(income_order, fill_value=0)
axes[0].bar(income_counts.index, income_counts.values,
            color=sns.color_palette("Blues_d", len(income_counts)))
axes[0].set_title("Income Bracket Count")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=20)

income_outcome = (df.groupby(["income_bracket", TARGET])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(income_order, fill_value=0))
income_outcome_pct = income_outcome.div(income_outcome.sum(axis=1), axis=0) * 100
income_outcome_pct.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10")
axes[1].set_title("Match Outcome by Income (normalised)")
axes[1].set_ylabel("Proportion (%)")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend(bbox_to_anchor=(1.01, 1), fontsize=8)

plt.tight_layout()
savefig("04_income_distribution")

# ── 3d. Education Level ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
edu_counts = df["education_level"].value_counts()
ax.bar(edu_counts.index, edu_counts.values,
       color=sns.color_palette("Greens_d", len(edu_counts)))
ax.set_title("Education Level Distribution", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
savefig("05_education_distribution")

# ── 3e. Location Type ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
loc_counts = df["location_type"].value_counts()
ax.pie(loc_counts, labels=loc_counts.index, autopct="%1.1f%%",
       startangle=90, colors=sns.color_palette(PALETTE, len(loc_counts)))
ax.set_title("Location Type Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("06_location_type")


# =============================================================================
# 4. BEHAVIOURAL FEATURES
# =============================================================================
NUM_COLS = ["app_usage_time_min", "swipe_right_ratio", "likes_received",
            "mutual_matches", "profile_pics_count", "bio_length",
            "message_sent_count", "emoji_usage_rate", "last_active_hour"]

# ── 4a. App Usage Time ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("App Usage Time Analysis", fontsize=14, fontweight="bold")

axes[0].hist(df["app_usage_time_min"], bins=40,
             color="#4C72B0", edgecolor="white")
axes[0].set_title("Distribution (minutes)")
axes[0].set_xlabel("Minutes")

sns.boxplot(x=TARGET, y="app_usage_time_min", data=df,
            palette=PALETTE, ax=axes[1])
axes[1].set_title("Usage Time by Outcome")
axes[1].tick_params(axis="x", rotation=30)
axes[1].set_xlabel("")

usage_label_counts = df["app_usage_time_label"].value_counts()
axes[2].bar(usage_label_counts.index, usage_label_counts.values,
            color=sns.color_palette(PALETTE, len(usage_label_counts)))
axes[2].set_title("Usage Label Distribution")
axes[2].tick_params(axis="x", rotation=15)

plt.tight_layout()
savefig("07_app_usage_time")

# ── 4b. Swipe Right Ratio ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Swipe Right Ratio Analysis", fontsize=14, fontweight="bold")

axes[0].hist(df["swipe_right_ratio"], bins=40,
             color="#DD8452", edgecolor="white")
axes[0].set_title("Distribution of Swipe Right Ratio")
axes[0].set_xlabel("Ratio (0–1)")

sns.violinplot(x=TARGET, y="swipe_right_ratio", data=df,
               palette=PALETTE, ax=axes[1])
axes[1].set_title("Swipe Right Ratio by Outcome")
axes[1].tick_params(axis="x", rotation=30)
axes[1].set_xlabel("")

plt.tight_layout()
savefig("08_swipe_right_ratio")

# ── 4c. Likes Received & Mutual Matches ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Engagement Metrics", fontsize=14, fontweight="bold")

for ax, col in zip(axes, ["likes_received", "mutual_matches"]):
    ax.hist(df[col], bins=40, edgecolor="white",
            color="#55A868" if col == "likes_received" else "#C44E52")
    ax.set_title(f"Distribution: {col.replace('_', ' ').title()}")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Count")

plt.tight_layout()
savefig("09_engagement_metrics")

# ── 4d. Message Sent Count & Bio Length ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Communication Metrics", fontsize=14, fontweight="bold")

for ax, col, color in zip(axes,
                           ["message_sent_count", "bio_length"],
                           ["#8172B2", "#937860"]):
    ax.hist(df[col], bins=40, color=color, edgecolor="white")
    ax.set_title(f"Distribution: {col.replace('_', ' ').title()}")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Count")

plt.tight_layout()
savefig("10_communication_metrics")

# ── 4e. Swipe Time of Day ─────────────────────────────────────────────────────
tod_order = ["Early Morning", "Morning", "Afternoon", "Evening",
             "Night", "After Midnight"]
tod_order = [t for t in tod_order if t in df["swipe_time_of_day"].unique()]

fig, ax = plt.subplots(figsize=(10, 5))
tod_counts = df["swipe_time_of_day"].value_counts().reindex(tod_order, fill_value=0)
ax.bar(tod_counts.index, tod_counts.values,
       color=sns.color_palette("twilight_shifted", len(tod_counts)))
ax.set_title("Swipe Activity by Time of Day", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
savefig("11_swipe_time_of_day")

# ── 4f. Last Active Hour (heatmap-style) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
hour_counts = df["last_active_hour"].value_counts().sort_index()
ax.bar(hour_counts.index, hour_counts.values,
       color=sns.color_palette("flare", 24))
ax.set_title("User Activity by Hour of Day", fontsize=13, fontweight="bold")
ax.set_xlabel("Hour (0–23)")
ax.set_ylabel("Count")
ax.set_xticks(range(0, 24))
plt.tight_layout()
savefig("12_last_active_hour")


# =============================================================================
# 5. CORRELATION HEATMAP
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 8))
corr = df[NUM_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, linewidths=0.5,
            annot_kws={"size": 9}, ax=ax)
ax.set_title("Correlation Heatmap — Numerical Features",
             fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("13_correlation_heatmap")

print("\n── TOP CORRELATED PAIRS ──")
corr_pairs = (corr.where(mask == False)
                  .stack()
                  .reset_index()
                  .rename(columns={0: "corr",
                                   "level_0": "feature_1",
                                   "level_1": "feature_2"}))
corr_pairs["abs_corr"] = corr_pairs["corr"].abs()
print(corr_pairs.sort_values("abs_corr", ascending=False)
                .head(10)
                .to_string(index=False))


# =============================================================================
# 6. OUTLIER DETECTION — IQR Method
# =============================================================================
print("\n── OUTLIER SUMMARY (IQR Method) ──")
outlier_summary = {}
for col in NUM_COLS:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out  = ((df[col] < lower) | (df[col] > upper)).sum()
    pct    = n_out / len(df) * 100
    outlier_summary[col] = {"lower_fence": round(lower, 2),
                             "upper_fence": round(upper, 2),
                             "n_outliers":  n_out,
                             "outlier_%":   round(pct, 2)}
    print(f"  {col:<25} outliers: {n_out:>5} ({pct:.2f}%)")

# Box-plot grid for all numerical features
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle("Box Plots — Outlier Detection (Numerical Features)",
             fontsize=14, fontweight="bold")
for ax, col in zip(axes.flatten(), NUM_COLS):
    ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6),
               medianprops=dict(color="red", linewidth=2),
               flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.set_title(col.replace("_", " ").title(), fontsize=9)
    ax.set_ylabel("Value")

plt.tight_layout()
savefig("14_boxplots_outliers")


# =============================================================================
# 7. FEATURE vs TARGET — Mean numerical values per outcome
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Mean Feature Value per Match Outcome",
             fontsize=14, fontweight="bold")

for ax, col in zip(axes.flatten(), NUM_COLS):
    means = df.groupby(TARGET)[col].mean().sort_values(ascending=False)
    bars  = ax.barh(means.index, means.values,
                    color=sns.color_palette(PALETTE, len(means)))
    ax.set_title(col.replace("_", " ").title(), fontsize=9)
    ax.set_xlabel("Mean Value")
    ax.axvline(df[col].mean(), color="red", linestyle="--",
               linewidth=1, label="Overall mean")

plt.tight_layout()
savefig("15_feature_by_outcome")


# =============================================================================
# 8. PAIRPLOT — core behavioural features coloured by outcome
# =============================================================================
PAIR_COLS = ["app_usage_time_min", "swipe_right_ratio",
             "likes_received", "mutual_matches", TARGET]
pair_df = df[PAIR_COLS].copy()

g = sns.pairplot(pair_df, hue=TARGET, diag_kind="kde",
                 plot_kws={"alpha": 0.3, "s": 10},
                 palette="tab10")
g.fig.suptitle("Pairplot — Core Behavioural Features by Outcome",
               y=1.02, fontsize=13, fontweight="bold")
savefig("16_pairplot")


print("\n✅  EDA complete — all figures saved.")
