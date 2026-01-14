import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

#loading dataset
df = pd.read_csv("Dataset_task3-4/Task 3 and 4_Loan_Data.csv")
fico = df["fico_score"].values
default = df["default"].values
# Sort by FICO (important)
order = np.argsort(fico)
fico = fico[order]
default = default[order]

#bucket lokehood
def bucket_log_likelihood(y):
    n = len(y)
    if n == 0:
        return 0

    k = np.sum(y)
    p = (k + 0.5) / (n + 1)  # smoothing
    return k*np.log(p) + (n-k)*np.log(1-p)

#precompute likehood
N = len(default)
LL = np.zeros((N, N))
for i in range(N):
    for j in range(i, N):
        LL[i, j] = bucket_log_likelihood(default[i:j+1])

#optimal binning
def optimal_binning(num_buckets):
    dp = np.full((num_buckets+1, N), -np.inf)
    prev = np.zeros((num_buckets+1, N), dtype=int)
    dp[1] = [LL[0, j] for j in range(N)]
    for b in range(2, num_buckets+1):
        for j in range(N):
            best_val = -np.inf
            best_idx = 0
            for i in range(b-1, j+1):
                val = dp[b-1, i-1] + LL[i, j]
                if val > best_val:
                    best_val = val
                    best_idx = i
            dp[b, j] = best_val
            prev[b, j] = best_idx
    bounds = []
    idx = N - 1
    for b in range(num_buckets, 0, -1):
        i = prev[b, idx]
        bounds.append((i, idx))
        idx = i - 1

    bounds.reverse()
    return bounds

#rating map
NUM_BUCKETS = 6
buckets = optimal_binning(NUM_BUCKETS)
rating_map = []
for r, (i, j) in enumerate(buckets, 1):
    low = fico[i]
    high = fico[j]
    n = j - i + 1
    k = np.sum(default[i:j+1])
    pd_val = k / n
    rating_map.append([r, low, high, n, k, round(pd_val,4)])
rating_df = pd.DataFrame(
    rating_map,
    columns=["Rating", "FICO_low", "FICO_high", "Total", "Defaults", "PD"]
)

print("\n========== RATING MAP ==========\n")
print(rating_df)

#assing rating
def assign_rating(score, rating_df):
    for _, row in rating_df.iterrows():
        if row["FICO_low"] <= score <= row["FICO_high"]:
            return row["Rating"]
    return rating_df["Rating"].max()

df["Rating"] = df["fico_score"].apply(lambda x: assign_rating(x, rating_df))

#plotting
# PD by rating
plt.figure(figsize=(7,5))
sns.barplot(data=rating_df, x="Rating", y="PD")
plt.title("Probability of Default by Rating Bucket")
plt.xlabel("Credit Rating (lower = better)")
plt.ylabel("Probability of Default")
plt.grid()
plt.show()

# FICO distribution vs default
plt.figure(figsize=(8,5))
sns.histplot(df, x="fico_score", hue="default", bins=30, kde=True)
plt.title("FICO Distribution by Default Status")
plt.show()

# FICO bucket boundaries
plt.figure(figsize=(9,5))
sns.histplot(df, x="fico_score", bins=40)
for _, row in rating_df.iterrows():
    plt.axvline(row["FICO_low"], color="red", linestyle="--", alpha=0.6)
plt.title("Optimal FICO Buckets")
plt.show()