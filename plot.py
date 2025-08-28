import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter

def plot_csv_time(
    path: str,
    *,
    has_header: bool = True,
    delimiter: str = ",",
    time_col: int = 0,
    mocap_x_col: int = 1,
    mocap_y_col: int = 2,
    mocap_z_col: int = 3,
    site_x_col: int = 18,
    site_y_col: int = 19,
    site_z_col: int = 20,
    mocap_quat_cols: tuple[int, int, int, int] = (4, 5, 6, 7),  # (w, x, y, z)
    site_quat_cols: tuple[int, int, int, int] = (21, 22, 23, 24),  # (w, x, y, z)
    mocap_rot_cols: tuple[int, int, int, int, int, int, int, int, int] = (8, 9, 10, 11, 12, 13, 14, 15, 16),  # 3x3 matrix
    site_rot_cols: tuple[int, int, int, int, int, int, int, int, int] = (25, 26, 27, 28, 29, 30, 31, 32, 33),  # 3x3 matrix
    time_unit: str | None = None,
    title: str | None = None,
    save: str | None = None,
    labels: tuple[str, str] = ("Desired (Mocap)", "Actual (Site)")
) -> None:
    print(f"Reading CSV file: {path}")
    header = 0 if has_header else None
    
    try:
        df = pd.read_csv(path, sep=delimiter, header=header, index_col=None)
        print(f"CSV shape: {df.shape}")
        print(f"CSV columns: {list(df.columns)}")
        print(f"CSV dtypes:\n{df.dtypes}")
        print(f"First few rows:\n{df.head()}")
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Debug column access
    print(f"time_col={time_col}, but using df['time'] for t_raw")
    print(f"df['time'] first few values: {df['time'].head()}")
    print(f"Column name at index {time_col}: '{df.columns[time_col]}'")

    # Select columns by index
    t_raw = df['time']  # Use column name to avoid indexing issues
    mocap_x = pd.to_numeric(df.iloc[:, mocap_x_col], errors="coerce")
    mocap_y = pd.to_numeric(df.iloc[:, mocap_y_col], errors="coerce")
    mocap_z = pd.to_numeric(df.iloc[:, mocap_z_col], errors="coerce")
    site_x = pd.to_numeric(df.iloc[:, site_x_col], errors="coerce")
    site_y = pd.to_numeric(df.iloc[:, site_y_col], errors="coerce")
    site_z = pd.to_numeric(df.iloc[:, site_z_col], errors="coerce")

    # Extract quaternion data
    mocap_quat_w = pd.to_numeric(df.iloc[:, mocap_quat_cols[0]],
                                 errors="coerce")
    mocap_quat_x = pd.to_numeric(df.iloc[:, mocap_quat_cols[1]],
                                 errors="coerce")
    mocap_quat_y = pd.to_numeric(df.iloc[:, mocap_quat_cols[2]],
                                 errors="coerce")
    mocap_quat_z = pd.to_numeric(df.iloc[:, mocap_quat_cols[3]],
                                 errors="coerce")
    
    site_quat_w = pd.to_numeric(df.iloc[:, site_quat_cols[0]],
                                errors="coerce")
    site_quat_x = pd.to_numeric(df.iloc[:, site_quat_cols[1]],
                                errors="coerce")
    site_quat_y = pd.to_numeric(df.iloc[:, site_quat_cols[2]],
                                errors="coerce")
    site_quat_z = pd.to_numeric(df.iloc[:, site_quat_cols[3]],
                                errors="coerce")

    # Extract rotation matrix data
    mocap_rot = []
    site_rot = []
    for i in range(9):
        mocap_rot.append(pd.to_numeric(df.iloc[:, mocap_rot_cols[i]],
                                       errors="coerce"))
        site_rot.append(pd.to_numeric(df.iloc[:, site_rot_cols[i]],
                                      errors="coerce"))

    # Parse time
    is_datetime = False
    print(f"t_raw type: {type(t_raw.iloc[0])}, "
          f"first few: {t_raw.head()}")
    print(f"t_raw values look like time? First few: {t_raw.head().values}")
    print(f"Are t_raw values monotonically increasing? "
          f"{t_raw.is_monotonic_increasing}")
    
    if time_unit == "datetime":
        t = pd.to_datetime(t_raw, errors="coerce")
        is_datetime = True
        print(f"Parsed as datetime, t type: {type(t.iloc[0])}")
    else:
        t_num = pd.to_numeric(t_raw, errors="coerce")
        print(f"t_num type: {type(t_num.iloc[0])}, "
              f"first few: {t_num.head()}")
        
        if t_num.notna().all():
            if time_unit in {"s", "ms", "us", "ns"}:
                t = pd.to_datetime(t_num, unit=time_unit, errors="coerce")
                is_datetime = True
                print(f"Parsed with unit {time_unit}, "
                      f"t type: {type(t.iloc[0])}")
            else:
                t = t_num
                print(f"Using numeric time, t type: {type(t.iloc[0])}")
        else:
            t = pd.to_datetime(t_raw, errors="coerce")
            is_datetime = True
            print(f"Parsed as datetime (fallback), "
                  f"t type: {type(t.iloc[0])}")

    print(f"Final t first few: {t.head()}")
    print(f"is_datetime: {is_datetime}")

    print(f"Data points: {len(t)} time points")
    print("Creating plots with dual trajectory mode")

    # Build plots - create figure with position and quaternion data
    fig, axes = plt.subplots(3, figsize=(8, 12), constrained_layout=True)
    
    # Position plots
    # mocap x(t) and site x(t) in same plot
    axes[0].plot(t, mocap_x, label=labels[0], color='blue')
    axes[0].plot(t, site_x, label=labels[1], color='red')
    axes[0].set_ylabel("x [m]")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title("X Position Tracking")

    # mocap y(t) and site y(t) in same plot
    axes[1].plot(t, mocap_y, label=labels[0], color='blue')
    axes[1].plot(t, site_y, label=labels[1], color='red')
    axes[1].set_ylabel("y [m]")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_title("Y Position Tracking")

    # mocap z(t) and site z(t) in same plot
    axes[2].plot(t, mocap_z, label=labels[0], color='blue')
    axes[2].plot(t, site_z, label=labels[1], color='red')
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("z [m]")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].set_title("Z Position Tracking")

    if title:
        fig.suptitle(title)

    if save:
        fig.savefig(save, dpi=150)
        print(f"Plot saved to: {save}")

    print("Showing plot...")
    plt.show()
    print("Plot display completed.")


plot_csv_time(
    "trajectories.csv",
    labels=("Desired (Mocap)", "Actual (Site)"),
    title="Robot Trajectory Tracking Analysis",
    save="trajectory_analysis.png"
)
