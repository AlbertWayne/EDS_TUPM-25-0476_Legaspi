# =============================================================================
# ROB-03: Kinematic Repeatability Error — Engineering Data Systems Pipeline
# Course: Computer Programming | Academic Year: 2026
# Topic: Kinematic Repeatability Error (Industrial Robot Accuracy)
# =============================================================================

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CLASS 1: DATA INGESTION
# Handles loading the raw dataset and applying the unique filter logic.
# =============================================================================
class DataIngestion:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df = None
        self.filtered_df = None

    def load(self) -> pd.DataFrame:
        try:
            self.raw_df = pd.read_csv(self.filepath)
            print(f"[INGESTION] Dataset loaded: {self.raw_df.shape[0]} rows, "
                  f"{self.raw_df.shape[1]} columns.")
            return self.raw_df
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.filepath}")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            raise

    def apply_unique_filter(self) -> pd.DataFrame:
        """
        Unique Filter Logic — Elevated-Elbow, High-Reach Wrist Configuration:
          - q3 > 1.572  : Elbow joint above median (upper posture region)
          - z  > 0.30   : End-effector in elevated Z zone (above 0.30 m)
          - 3.0 <= q6 <= 5.5 : Wrist rotation in mid-to-high sector
        This isolates a physically distinct arm posture subset of the dataset.
        """
        try:
            self.filtered_df = self.raw_df[
                (self.raw_df['q3'] > 1.572) &
                (self.raw_df['z'] > 0.30) &
                (self.raw_df['q6'] >= 3.0) &
                (self.raw_df['q6'] <= 5.5)
            ].copy().reset_index(drop=True)

            print(f"[INGESTION] Filter applied: {len(self.filtered_df)} rows retained "
                  f"({100 * len(self.filtered_df) / len(self.raw_df):.2f}% of original).")
            return self.filtered_df
        except KeyError as e:
            print(f"[ERROR] Missing expected column during filtering: {e}")
            raise


# =============================================================================
# CLASS 2: DATA CLEANER
# Handles null removal, duplicate removal, type correction, and saving outputs.
# =============================================================================
class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_nulls(self) -> 'DataCleaner':
        try:
            null_count = self.df.isnull().sum().sum()
            if null_count > 0:
                self.df = self.df.dropna()
                print(f"[CLEANER] Removed {null_count} null values.")
            else:
                print("[CLEANER] No null values found.")
        except Exception as e:
            print(f"[ERROR] Null removal failed: {e}")
        return self

    def remove_duplicates(self) -> 'DataCleaner':
        try:
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            removed = before - len(self.df)
            print(f"[CLEANER] Removed {removed} duplicate rows.")
        except Exception as e:
            print(f"[ERROR] Duplicate removal failed: {e}")
        return self

    def correct_dtypes(self) -> 'DataCleaner':
        try:
            cols = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'x', 'y', 'z']
            for col in cols:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df = self.df.dropna()
            print("[CLEANER] Data types verified and corrected.")
        except Exception as e:
            print(f"[ERROR] Type correction failed: {e}")
        return self

    def engineer_features(self) -> 'DataCleaner':
        try:
            self.df['position_error'] = np.sqrt(
                self.df['x']**2 + self.df['y']**2 + self.df['z']**2
            )
            self.df['wrist_angle_sum'] = self.df['q4'] + self.df['q5'] + self.df['q6']
            median_q3 = self.df['q3'].median()
            self.df['elbow_group'] = np.where(
                self.df['q3'] <= median_q3, 'Mid-Elbow', 'High-Elbow'
            )
            print("[CLEANER] Feature engineering complete "
                  f"(position_error, wrist_angle_sum, elbow_group).")
        except Exception as e:
            print(f"[ERROR] Feature engineering failed: {e}")
        return self

    def save(self, original_df: pd.DataFrame, out_dir: str = "data/") -> 'DataCleaner':
        try:
            os.makedirs(out_dir, exist_ok=True)
            original_df.to_csv(os.path.join(out_dir, "dataset_original.csv"), index=False)
            self.df.to_csv(os.path.join(out_dir, "dataset_cleaned.csv"), index=False)
            print(f"[CLEANER] Saved dataset_original.csv and dataset_cleaned.csv to '{out_dir}'.")
        except Exception as e:
            print(f"[ERROR] Saving datasets failed: {e}")
        return self

    def get_clean_df(self) -> pd.DataFrame:
        return self.df


# =============================================================================
# CLASS 3: ANALYZER
# Computes all required statistics using NumPy and SciPy.
# =============================================================================
class Analyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def descriptive_stats(self) -> dict:
        try:
            numeric_cols = ['q1','q2','q3','q4','q5','q6','x','y','z','position_error']
            stats_dict = {}
            for col in numeric_cols:
                arr = self.df[col].values
                stats_dict[col] = {
                    'mean'     : float(np.mean(arr)),
                    'median'   : float(np.median(arr)),
                    'std_dev'  : float(np.std(arr, ddof=1)),
                    'variance' : float(np.var(arr, ddof=1)),
                    'min'      : float(np.min(arr)),
                    'max'      : float(np.max(arr)),
                }
            self.results['descriptive'] = stats_dict

            print("\n[ANALYZER] ── Descriptive Statistics (position_error) ──")
            pe = stats_dict['position_error']
            print(f"  Mean     : {pe['mean']:.6f} m")
            print(f"  Median   : {pe['median']:.6f} m")
            print(f"  Std Dev  : {pe['std_dev']:.6f} m")
            print(f"  Variance : {pe['variance']:.6f} m²")
            print(f"  Min      : {pe['min']:.6f} m")
            print(f"  Max      : {pe['max']:.6f} m")
            return stats_dict
        except Exception as e:
            print(f"[ERROR] Descriptive stats failed: {e}")
            raise

    def distribution_analysis(self) -> dict:
        try:
            arr = self.df['position_error'].values
            skewness  = float(stats.skew(arr))
            kurtosis  = float(stats.kurtosis(arr))
            q1_val    = float(np.percentile(arr, 25))
            q3_val    = float(np.percentile(arr, 75))
            iqr       = q3_val - q1_val
            lower_fence = q1_val - 1.5 * iqr
            upper_fence = q3_val + 1.5 * iqr
            outliers  = self.df[
                (self.df['position_error'] < lower_fence) |
                (self.df['position_error'] > upper_fence)
            ]

            dist_dict = {
                'skewness'    : skewness,
                'kurtosis'    : kurtosis,
                'IQR'         : iqr,
                'lower_fence' : lower_fence,
                'upper_fence' : upper_fence,
                'outlier_count': len(outliers),
            }
            self.results['distribution'] = dist_dict

            print("\n[ANALYZER] ── Distribution Analysis ──")
            print(f"  Skewness      : {skewness:.6f}")
            print(f"  Kurtosis      : {kurtosis:.6f}")
            print(f"  IQR           : {iqr:.6f} m")
            print(f"  Lower Fence   : {lower_fence:.6f} m")
            print(f"  Upper Fence   : {upper_fence:.6f} m")
            print(f"  Outliers Found: {len(outliers)}")
            return dist_dict
        except Exception as e:
            print(f"[ERROR] Distribution analysis failed: {e}")
            raise

    def correlation_analysis(self) -> pd.DataFrame:
        try:
            corr_cols = ['q1','q2','q3','q4','q5','q6','x','y','z','position_error']
            corr_matrix = self.df[corr_cols].corr(method='pearson')
            self.results['correlation'] = corr_matrix

            print("\n[ANALYZER] ── Correlation with position_error ──")
            pe_corr = corr_matrix['position_error'].drop('position_error').sort_values(
                key=abs, ascending=False
            )
            for col, val in pe_corr.items():
                print(f"  {col:20s}: {val:+.4f}")
            return corr_matrix
        except Exception as e:
            print(f"[ERROR] Correlation analysis failed: {e}")
            raise

    def comparative_analysis(self) -> dict:
        try:
            grp_a = self.df[self.df['elbow_group'] == 'Mid-Elbow']['position_error'].values
            grp_b = self.df[self.df['elbow_group'] == 'High-Elbow']['position_error'].values
            t_stat, p_val = stats.ttest_ind(grp_a, grp_b)

            comp_dict = {
                'Group_A_mean'  : float(np.mean(grp_a)),
                'Group_A_std'   : float(np.std(grp_a, ddof=1)),
                'Group_A_n'     : len(grp_a),
                'Group_B_mean'  : float(np.mean(grp_b)),
                'Group_B_std'   : float(np.std(grp_b, ddof=1)),
                'Group_B_n'     : len(grp_b),
                't_statistic'   : float(t_stat),
                'p_value'       : float(p_val),
            }
            self.results['comparative'] = comp_dict

            print("\n[ANALYZER] ── Comparative Analysis (Mid-Elbow vs High-Elbow) ──")
            print(f"  Group A (Mid-Elbow)  | n={len(grp_a)} | "
                  f"Mean={np.mean(grp_a):.6f} m | Std={np.std(grp_a, ddof=1):.6f} m")
            print(f"  Group B (High-Elbow) | n={len(grp_b)} | "
                  f"Mean={np.mean(grp_b):.6f} m | Std={np.std(grp_b, ddof=1):.6f} m")
            print(f"  t-statistic: {t_stat:.4f} | p-value: {p_val:.6f}")
            sig = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
            print(f"  Difference is statistically {sig} at α=0.05")
            return comp_dict
        except Exception as e:
            print(f"[ERROR] Comparative analysis failed: {e}")
            raise

    def run_all(self):
        """Run the complete analysis suite."""
        self.descriptive_stats()
        self.distribution_analysis()
        self.correlation_analysis()
        self.comparative_analysis()
        return self.results


# =============================================================================
# CLASS 4: VISUALIZER
# Generates all required static and animated charts.
# =============================================================================
class Visualizer:
    def __init__(self, df: pd.DataFrame, out_dir: str = "outputs/"):
        self.df = df
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    # ── STATIC CHART 1: Histogram of Position Error ──────────────────────────
    def plot_histogram(self):
        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            arr = self.df['position_error'].values
            ax.hist(arr, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
            ax.axvline(np.mean(arr),   color='red',    lw=2, linestyle='--',
                       label=f'Mean = {np.mean(arr):.4f} m')
            ax.axvline(np.median(arr), color='orange', lw=2, linestyle='-.',
                       label=f'Median = {np.median(arr):.4f} m')
            ax.set_title('Distribution of Kinematic Position Error\n'
                         '(Elevated-Elbow, High-Reach Wrist Configuration)', fontsize=13)
            ax.set_xlabel('Position Error (m)')
            ax.set_ylabel('Frequency')
            ax.legend()
            plt.tight_layout()
            path = os.path.join(self.out_dir, 'static_1_histogram.png')
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[VISUALIZER] Saved: {path}")
        except Exception as e:
            print(f"[ERROR] Histogram failed: {e}")

    # ── STATIC CHART 2: Boxplot — Mid-Elbow vs High-Elbow ────────────────────
    def plot_boxplot(self):
        try:
            grp_a = self.df[self.df['elbow_group'] == 'Mid-Elbow']['position_error'].values
            grp_b = self.df[self.df['elbow_group'] == 'High-Elbow']['position_error'].values
            fig, ax = plt.subplots(figsize=(8, 6))
            bp = ax.boxplot([grp_a, grp_b],
                            labels=['Mid-Elbow (q3 ≤ 1.76)', 'High-Elbow (q3 > 1.76)'],
                            patch_artist=True,
                            medianprops=dict(color='black', lw=2))
            bp['boxes'][0].set_facecolor('#4C9BE8')
            bp['boxes'][1].set_facecolor('#E87B4C')
            ax.set_title('Position Error Comparison\nMid-Elbow vs High-Elbow Configuration',
                         fontsize=13)
            ax.set_ylabel('Position Error (m)')
            ax.set_xlabel('Elbow Joint Group')
            plt.tight_layout()
            path = os.path.join(self.out_dir, 'static_2_boxplot.png')
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[VISUALIZER] Saved: {path}")
        except Exception as e:
            print(f"[ERROR] Boxplot failed: {e}")

    # ── STATIC CHART 3: Scatter — q2 vs Position Error ───────────────────────
    def plot_scatter(self):
        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            sc = ax.scatter(self.df['q2'], self.df['position_error'],
                            c=self.df['z'], cmap='viridis', alpha=0.5, s=12)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('End-Effector Z Height (m)')
            ax.set_title('Shoulder Joint Angle (q2) vs Position Error\n'
                         'Coloured by End-Effector Z Height', fontsize=13)
            ax.set_xlabel('q2 — Shoulder Joint Angle (rad)')
            ax.set_ylabel('Position Error (m)')
            plt.tight_layout()
            path = os.path.join(self.out_dir, 'static_3_scatter.png')
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[VISUALIZER] Saved: {path}")
        except Exception as e:
            print(f"[ERROR] Scatter failed: {e}")

    # ── STATIC CHART 4: Heatmap — Pearson Correlation Matrix ─────────────────
    def plot_heatmap(self, corr_matrix: pd.DataFrame):
        try:
            cols = ['q1','q2','q3','q4','q5','q6','x','y','z','position_error']
            cm = corr_matrix.loc[cols, cols]
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax, label='Pearson r')
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(cols, fontsize=9)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax.text(j, i, f"{cm.values[i, j]:.2f}",
                            ha='center', va='center', fontsize=7,
                            color='white' if abs(cm.values[i, j]) > 0.6 else 'black')
            ax.set_title('Pearson Correlation Heatmap\nRobot Joint Angles & End-Effector Positions',
                         fontsize=13)
            plt.tight_layout()
            path = os.path.join(self.out_dir, 'static_4_heatmap.png')
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"[VISUALIZER] Saved: {path}")
        except Exception as e:
            print(f"[ERROR] Heatmap failed: {e}")

    # ── ANIMATION 1: Position Error Rolling Mean Over Index ──────────────────
    def animate_rolling_error(self):
        try:
            window = 30
            errors = self.df['position_error'].values
            rolling_mean = pd.Series(errors).rolling(window=window, min_periods=1).mean().values

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlim(0, len(errors))
            ax.set_ylim(errors.min() * 0.95, errors.max() * 1.05)
            ax.set_title(f'Animation 1: Position Error Trend with {window}-Sample Rolling Mean',
                         fontsize=12)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Position Error (m)')

            raw_line,  = ax.plot([], [], color='lightsteelblue', lw=0.8,
                                 alpha=0.6, label='Raw Error')
            roll_line, = ax.plot([], [], color='red', lw=2,
                                 label=f'Rolling Mean (w={window})')
            ax.legend(loc='upper right')

            step = 10  
            frames = range(1, len(errors) // step + 1)

            def init():
                raw_line.set_data([], [])
                roll_line.set_data([], [])
                return raw_line, roll_line

            def update(frame):
                idx = frame * step
                x = np.arange(idx)
                raw_line.set_data(x, errors[:idx])
                roll_line.set_data(x, rolling_mean[:idx])
                return raw_line, roll_line

            ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                          blit=True, interval=30)
            path = os.path.join(self.out_dir, 'animation_1_rolling_error.gif')
            ani.save(path, writer='pillow', fps=25)
            plt.close()
            print(f"[VISUALIZER] Saved: {path}")
        except Exception as e:
            print(f"[ERROR] Animation 1 failed: {e}")

    # ── ANIMATION 2: Joint Angle Distribution Shift Across Error Bins ────────
    def animate_joint_distribution_shift(self):
        try:
            joint_cols = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
            colors     = ['#4C9BE8', '#E87B4C', '#5DBB63',
                          '#B05DBB', '#E8C84C', '#E84C6B']

            sorted_df = self.df.sort_values('position_error').reset_index(drop=True)
            n_bins    = 10
            bin_size  = len(sorted_df) // n_bins
            bins      = [sorted_df.iloc[i * bin_size:(i + 1) * bin_size]
                         for i in range(n_bins)]

            global_min = {col: self.df[col].min() for col in joint_cols}
            global_max = {col: self.df[col].max() for col in joint_cols}

            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            axes = axes.flatten()
            fig.suptitle('Animation 2: Joint Angle Distribution Shift\n'
                         'Across Increasing Position Error Bins',
                         fontsize=13, fontweight='bold')

            bar_containers = []
            bin_edges_list = []
            for idx, col in enumerate(joint_cols):
                ax = axes[idx]
                counts, edges = np.histogram(bins[0][col], bins=18,
                                             range=(global_min[col], global_max[col]))
                container = ax.bar(edges[:-1], counts,
                                   width=np.diff(edges),
                                   color=colors[idx], alpha=0.80,
                                   edgecolor='white', linewidth=0.4,
                                   align='edge')
                ax.set_xlim(global_min[col], global_max[col])
                ax.set_ylim(0, bin_size // 2)
                ax.set_title(f'{col} — Joint Angle (rad)', fontsize=10)
                ax.set_xlabel('Angle (rad)', fontsize=8)
                ax.set_ylabel('Frequency',   fontsize=8)
                bar_containers.append(container)
                bin_edges_list.append(edges)

            error_text = fig.text(0.5, 0.01, '', ha='center',
                                  fontsize=10, color='dimgray')
            plt.tight_layout(rect=[0, 0.04, 1, 1])

            def update(frame_idx):
                bin_df = bins[frame_idx]
                pe_min = bin_df['position_error'].min()
                pe_max = bin_df['position_error'].max()
                error_text.set_text(
                    f'Error Bin {frame_idx + 1}/{n_bins}  |  '
                    f'Position Error Range: [{pe_min:.4f} – {pe_max:.4f}] m'
                )
                for idx, col in enumerate(joint_cols):
                    counts, _ = np.histogram(bin_df[col], bins=18,
                                             range=(global_min[col], global_max[col]))
                    for bar, h in zip(bar_containers[idx], counts):
                        bar.set_height(h)
                return [bar for container in bar_containers for bar in container]

            ani = animation.FuncAnimation(
                fig, update,
                frames=n_bins,
                interval=700,   
                blit=False,
                repeat=True
            )

            path = os.path.join(self.out_dir, 'animation_2_joint_distribution_shift.gif')
            ani.save(path, writer='pillow', fps=1.5)
            plt.close()
            print(f"[VISUALIZER] Saved: {path}")
        except Exception as e:
            print(f"[ERROR] Animation 2 failed: {e}")

    def run_all(self, corr_matrix: pd.DataFrame):
        """Run all visualizations in sequence."""
        print("\n[VISUALIZER] ── Generating Static Charts ──")
        self.plot_histogram()
        self.plot_boxplot()
        self.plot_scatter()
        self.plot_heatmap(corr_matrix)
        print("\n[VISUALIZER] ── Generating Animations ──")
        self.animate_rolling_error()
        self.animate_joint_distribution_shift()
        print("[VISUALIZER] All visualizations complete.")


# =============================================================================
# MAIN ENTRY POINT
# Full pipeline: Ingest → Clean → Analyze → Visualize
# =============================================================================
def main():
    print("=" * 65)
    print("  ROB-03: Kinematic Repeatability Error — Data Pipeline")
    print("=" * 65)

    # ── STEP 1: INGESTION ─────────────────────────────────────────
    print("\n[STEP 1] Data Ingestion")
    ingestion = DataIngestion(filepath="data/dataset_original.csv")
    raw_df = ingestion.load()
    filtered_df = ingestion.apply_unique_filter()

    # ── STEP 2: CLEANING ──────────────────────────────────────────
    print("\n[STEP 2] Data Cleaning & Feature Engineering")
    cleaner = DataCleaner(filtered_df)
    cleaner.remove_nulls() \
           .remove_duplicates() \
           .correct_dtypes() \
           .engineer_features() \
           .save(original_df=raw_df, out_dir="data/")
    clean_df = cleaner.get_clean_df()

    # ── STEP 3: ANALYSIS ──────────────────────────────────────────
    print("\n[STEP 3] Statistical Analysis")
    analyzer = Analyzer(clean_df)
    results = analyzer.run_all()
    corr_matrix = results['correlation']

    # ── STEP 4: VISUALIZATION ─────────────────────────────────────
    print("\n[STEP 4] Visualization & Animation")
    visualizer = Visualizer(clean_df, out_dir="outputs/")
    visualizer.run_all(corr_matrix)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE.")
    print("  Outputs saved to: data/  and  outputs/")
    print("=" * 65)


if __name__ == "__main__":
    main()