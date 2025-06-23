import numpy as np
import pandas as pd

from paths import FEATURES_PRQ_PATH


def main():
    # Load feature parquet
    df = pd.read_parquet(FEATURES_PRQ_PATH)

    # Keep only positive class (Yellowhammer)
    df_pos = df[df['label'] == 1]
    if df_pos.empty:
        print("No positive-class (label==1) samples found in the features file.")
        return

    # Prepare containers for per-sample statistics
    per_mean = []
    per_median = []
    per_max = []
    per_min = []
    per_std = []

    for arr in df_pos['features']:
        v = np.asarray(arr).flatten()
        per_mean.append(np.mean(v))
        per_median.append(np.median(v))
        per_max.append(np.max(v))
        per_min.append(np.min(v))
        per_std.append(np.std(v))

    # Count specs where (max - median) > 400
    per_max = np.array(per_max)
    per_median = np.array(per_median)
    per_std = np.array(per_std)
    high_contrast_mask = (per_max - per_median) > 400
    n_high_contrast = np.sum(high_contrast_mask)
    total_pos = len(per_max)

    stats = {
        'mean': np.array(per_mean),
        'median': np.array(per_median),
        'max': np.array(per_max),
        'min': np.array(per_min),
        'std': np.array(per_std),
    }

    # Aggregate and print
    print("Statistical summary over positive-class spectrograms\n")
    for metric_name, values in stats.items():
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        print(
            f"{metric_name:<6}: mean={values.mean():.3f} | median={np.median(values):.3f} | "
            f"max={values.max():.3f} | min={values.min():.3f} | std={values.std():.3f} | "
            f"p25={p25:.3f} | p75={p75:.3f}"
        )

    print("\nHigh-contrast examples (max - median > 400): "
          f"{n_high_contrast}/{total_pos} ({n_high_contrast/total_pos:.2%})")


if __name__ == "__main__":
    main() 