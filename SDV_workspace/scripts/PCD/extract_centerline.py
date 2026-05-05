#!/usr/bin/env python3
"""
extract_centerline.py — Extract a smooth centerline path from global_lane_map.txt.
Outputs ordered waypoints as centerline_waypoints.txt and a visualization plot.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def main():
    # 1. Load raw lane points
    data = np.loadtxt('global_lane_map.txt')
    x_raw, y_raw = data[:, 0], data[:, 1]
    print(f"Loaded {len(x_raw):,} raw points")
    print(f"  X range: {x_raw.min():.2f} to {x_raw.max():.2f} m")
    print(f"  Y range: {y_raw.min():.2f} to {y_raw.max():.2f} m")

    # 2. Remove outliers (points too far from the bulk)
    x_med, y_med = np.median(x_raw), np.median(y_raw)
    x_std, y_std = np.std(x_raw), np.std(y_raw)
    inlier = (
        (np.abs(x_raw - x_med) < 3 * x_std) &
        (np.abs(y_raw - y_med) < 3 * y_std)
    )
    x_clean, y_clean = x_raw[inlier], y_raw[inlier]
    print(f"After outlier removal: {len(x_clean):,} points")

    # 3. Bin along the dominant travel direction
    #    Compute cumulative arc-length-like parameterization using X as primary
    #    Since the car drives mostly forward (X), bin by X and take median Y
    x_min, x_max = x_clean.min(), x_clean.max()
    total_dist = x_max - x_min
    bin_size = 0.05  # 5cm bins
    n_bins = max(int(total_dist / bin_size), 10)

    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers_x = []
    bin_centers_y = []

    for i in range(n_bins):
        mask = (x_clean >= bin_edges[i]) & (x_clean < bin_edges[i + 1])
        if np.sum(mask) >= 5:  # need minimum points for reliability
            bin_centers_x.append(np.median(x_clean[mask]))
            bin_centers_y.append(np.median(y_clean[mask]))

    cx = np.array(bin_centers_x)
    cy = np.array(bin_centers_y)
    print(f"Binned into {len(cx)} segments ({bin_size*100:.0f}cm resolution)")

    # 4. Smooth the centerline
    if len(cx) > 10:
        # Adaptive window: ~20cm smoothing
        window = max(int(0.20 / bin_size), 3)
        if window % 2 == 0:
            window += 1
        cy_smooth = uniform_filter1d(cy, size=window)
        cx_smooth = uniform_filter1d(cx, size=window)
    else:
        cx_smooth = cx
        cy_smooth = cy

    print(f"Smoothed with window={window} ({window * bin_size * 100:.0f}cm)")

    # 5. Compute arc-length parameterized waypoints
    dx = np.diff(cx_smooth)
    dy = np.diff(cy_smooth)
    ds = np.sqrt(dx**2 + dy**2)
    arc_length = np.concatenate([[0], np.cumsum(ds)])
    total_path_length = arc_length[-1]
    print(f"Total centerline length: {total_path_length:.2f} m")

    # 6. Resample at uniform spacing (every 5cm)
    spacing = 0.05
    n_waypoints = int(total_path_length / spacing) + 1
    s_uniform = np.linspace(0, total_path_length, n_waypoints)
    wx = np.interp(s_uniform, arc_length, cx_smooth)
    wy = np.interp(s_uniform, arc_length, cy_smooth)
    print(f"Resampled to {len(wx)} waypoints at {spacing*100:.0f}cm spacing")

    # 7. Save waypoints
    waypoints = np.column_stack([wx, wy])
    np.savetxt('centerline_waypoints.txt', waypoints, fmt='%.6f',
               header='X Y (meters, global frame)')
    print(f"Saved: centerline_waypoints.txt")

    # 8. Compute heading at each waypoint
    headings = np.arctan2(np.diff(wy), np.diff(wx))
    headings = np.append(headings, headings[-1])  # duplicate last

    waypoints_with_heading = np.column_stack([wx, wy, headings])
    np.savetxt('centerline_waypoints_heading.txt', waypoints_with_heading,
               fmt='%.6f', header='X Y Heading(rad)')
    print(f"Saved: centerline_waypoints_heading.txt")

    # 9. Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0f0f23')

    # Left: raw points + centerline
    ax1 = axes[0]
    ax1.scatter(x_raw, y_raw, s=0.05, c='cyan', alpha=0.15, label='Raw lane pts')
    ax1.plot(wx, wy, 'r-', linewidth=2, label=f'Centerline ({len(wx)} pts)')
    ax1.plot(wx[0], wy[0], 'g*', markersize=15, label='Start')
    ax1.plot(wx[-1], wy[-1], 'rs', markersize=10, label='End')
    # Direction arrows every 20 waypoints
    step = max(len(wx) // 15, 1)
    for i in range(0, len(wx) - 1, step):
        dx_a = wx[min(i+3, len(wx)-1)] - wx[i]
        dy_a = wy[min(i+3, len(wy)-1)] - wy[i]
        ax1.annotate('', xy=(wx[i] + dx_a*2, wy[i] + dy_a*2),
                     xytext=(wx[i], wy[i]),
                     arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))

    ax1.set_xlabel('X (Forward) [m]', color='white')
    ax1.set_ylabel('Y (Left) [m]', color='white')
    ax1.set_title('Lane Map + Extracted Centerline', color='white', fontsize=13)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=10, loc='lower left')
    ax1.set_facecolor('#1a1a2e')
    ax1.tick_params(colors='white')

    # Right: centerline only with heading arrows
    ax2 = axes[1]
    ax2.plot(wx, wy, 'r-', linewidth=2)
    ax2.plot(wx[0], wy[0], 'g*', markersize=15, label='Start')
    ax2.plot(wx[-1], wy[-1], 'rs', markersize=10, label='End')
    # Heading arrows
    for i in range(0, len(wx), step):
        h = headings[i]
        ax2.arrow(wx[i], wy[i], 0.15*np.cos(h), 0.15*np.sin(h),
                  head_width=0.06, head_length=0.03, fc='yellow', ec='yellow')

    ax2.set_xlabel('X (Forward) [m]', color='white')
    ax2.set_ylabel('Y (Left) [m]', color='white')
    ax2.set_title(f'Centerline Waypoints ({len(wx)} pts, {total_path_length:.1f}m path)',
                  color='white', fontsize=13)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=10)
    ax2.set_facecolor('#1a1a2e')
    ax2.tick_params(colors='white')

    plt.tight_layout()
    out = 'centerline_plot.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {out}")

    print("\n=== Summary ===")
    print(f"  Raw points:      {len(x_raw):,}")
    print(f"  Centerline pts:  {len(wx)}")
    print(f"  Path length:     {total_path_length:.2f} m")
    print(f"  Start: ({wx[0]:.2f}, {wy[0]:.2f})")
    print(f"  End:   ({wx[-1]:.2f}, {wy[-1]:.2f})")


if __name__ == "__main__":
    main()
