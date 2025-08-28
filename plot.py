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
    site_x_col: int = 9,
    site_y_col: int = 10,
    site_z_col: int = 11,
    mocap_quat_cols: tuple[int, int, int, int] = (4, 5, 6, 7),  # (w, x, y, z)
    site_quat_cols: tuple[int, int, int, int] = (12, 13, 14, 15),
    # (w, x, y, z)
    time_unit: str | None = None,
    title: str | None = None,
    save: str | None = None,
    labels: tuple[str, str] = ("Desired (Mocap)", "Actual (Site)")
) -> None:
    """
    Plot trajectories from CSV file matching the log_trajectories format.
    CSV layout: [time, mocap_x, mocap_y, mocap_z, site_x, site_y, site_z,
                 mocap_quat_w, mocap_quat_x, mocap_quat_y, mocap_quat_z,
                 site_quat_w, site_quat_x, site_quat_y, site_quat_z]
    Plots position trajectories and quaternion components over time.
    """
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
    print(f"time_col={time_col}, df.iloc[:, {time_col}] first few values:")
    print(df.iloc[:5, time_col])
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

    # Parse time
    is_datetime = False
    print(f"t_raw type: {type(t_raw.iloc[0])}, "
          f"first few: {t_raw.head()}")
    
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
    # mocap x(t)
    axes[0].plot(t, mocap_x, label=labels[0], color='blue')
    axes[0].plot(t, site_x, label=labels[1], color='red')
    axes[0].set_ylabel("x [m]")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # # site x(t)
    # axes[0, 1].plot(t, site_x, label=labels[1], color='red')
    # axes[0, 1].set_ylabel("site_x")
    # axes[0, 1].legend()
    # axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # mocap y(t)
    axes[1].plot(t, mocap_y, label=labels[0], color='blue')
    axes[1].plot(t, site_y, label=labels[1], color='red')
    axes[1].set_ylabel("y [m]")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # # site y(t)
    # axes[1, 1].plot(t, site_y, label=labels[1], color='red')
    # axes[1, 1].set_ylabel("site_y")
    # axes[1, 1].legend()
    # axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    # mocap z(t)
    axes[2].plot(t, mocap_z, label=labels[0], color='blue')
    axes[2].plot(t, site_z, label=labels[1], color='red')
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("z [m]")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # # Quaternion plots
    # # mocap quaternion components
    # axes[2, 0].plot(t, mocap_quat_w, label=f"{labels[0]} w", color='blue', linestyle='-')
    # axes[2, 0].plot(t, mocap_quat_x, label=f"{labels[0]} x", color='blue', linestyle='--')
    # axes[2, 0].plot(t, mocap_quat_y, label=f"{labels[0]} y", color='blue', linestyle='-.')
    # axes[2, 0].plot(t, mocap_quat_z, label=f"{labels[0]} z", color='blue', linestyle=':')
    # axes[2, 0].set_ylabel("mocap_quat")
    # axes[2, 0].legend()
    # axes[2, 0].grid(True, linestyle="--", alpha=0.5)

    # # site quaternion components
    # axes[2, 1].plot(t, site_quat_w, label=f"{labels[1]} w", color='red', linestyle='-')
    # axes[2, 1].plot(t, site_quat_x, label=f"{labels[1]} x", color='red', linestyle='--')
    # axes[2, 1].plot(t, site_quat_y, label=f"{labels[1]} y", color='red', linestyle='-.')
    # axes[2, 1].plot(t, site_quat_z, label=f"{labels[1]} z", color='red', linestyle=':')
    # axes[2, 1].set_ylabel("site_quat")
    # axes[2, 1].legend()
    # axes[2, 1].grid(True, linestyle="--", alpha=0.5)

    # # Trajectory plots
    # # mocap y vs x
    # axes[3, 0].plot(mocap_x, mocap_y, label=labels[0], color='blue')
    # axes[3, 0].set_xlabel("mocap_x")
    # axes[3, 0].set_ylabel("mocap_y")
    # axes[3, 0].set_aspect("equal", adjustable="box")
    # axes[3, 0].grid(True, linestyle="--", alpha=0.5)

    # # site y vs x
    # axes[3, 1].plot(site_x, site_y, label=labels[1], color='red')
    # axes[3, 1].set_xlabel("site_x")
    # axes[3, 1].set_ylabel("site_y")
    # axes[3, 1].set_aspect("equal", adjustable="box")
    # axes[3, 1].grid(True, linestyle="--", alpha=0.5)

    # Format time axis if datetime
    if is_datetime:
        locator = AutoDateLocator()
        formatter = DateFormatter("%Y-%m-%d %H:%M:%S")
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]):
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()

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
