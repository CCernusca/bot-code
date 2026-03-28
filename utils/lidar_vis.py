import matplotlib.pyplot as plt
import numpy as np

def visualise(px, py, width, length, angle_f, results, detected_dict, intersections=None, corners=[]):
    angles_deg = np.array([r[0] for r in results])
    distances = np.array([r[1] for r in results])
    raw_rad = np.radians(angles_deg)
    offset_rad = np.radians(angle_f)
    
    # Calculate raw Cartesian for reconstruction
    all_pts_x = distances * np.cos(raw_rad)
    all_pts_y = distances * np.sin(raw_rad)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Prepare point styles
    point_colors = ['black'] * len(distances)
    point_edge_colors = ['none'] * len(distances)
    point_widths = [0] * len(distances)

    colors = plt.cm.get_cmap('tab10')
    inf_val = max(width, length) * 5

    for i, ((slope, first_idx), indices) in enumerate(detected_dict.items()):
        line_color = colors(i % 10)
        
        # 1. Reconstruct Infinite Line
        dx, dy = slope
        x0, y0 = all_pts_x[first_idx], all_pts_y[first_idx]
        t = np.linspace(-inf_val, inf_val, 200)
        lx, ly = x0 + t*dx, y0 + t*dy
        ax.plot(np.arctan2(ly, lx), np.sqrt(lx**2 + ly**2), 
                color=line_color, lw=2, ls='--', alpha=0.6)

        # 2. Update outlines for points belonging to this line
        for idx in indices:
            point_edge_colors[idx] = line_color
            point_widths[idx] = 1.0
    
    # Get original points used for detection
    pts = [(d * np.cos(np.radians(deg)), d * np.sin(np.radians(deg))) 
           for deg, d in results]

    # 3. Plot points with their outlines
    # We plot them one by one to handle individual edgecolors efficiently 
    # or use a scatter with array inputs
    ax.scatter(raw_rad, distances, c='black', 
               edgecolors=point_edge_colors, linewidths=point_widths, 
               s=30, zorder=5, label='Sensor Points')

    # 4. Plot intersections (Passed in from outside)
    if intersections:
        for ix, iy in intersections:
            r_int = np.sqrt(ix**2 + iy**2)
            theta_int = np.arctan2(iy, ix)
            ax.scatter(theta_int, r_int, color='yellow', marker='X', s=100, 
                       edgecolors='black', zorder=10, label='Intersection')
     
    # 5. Plot corners (Passed in from outside)
    if len(corners) > 0:
        for ix, iy in corners:
            r_int = np.sqrt(ix**2 + iy**2)
            theta_int = np.arctan2(iy, ix)
            ax.scatter(theta_int, r_int, color='white', marker='X', s=100, 
                       edgecolors='black', zorder=10, label='Corner')
    
    ax.set_rmax(max(distances) * 1.2)
    plt.show()


class LiveVisualiser:
    """
    Real-time polar plot of lidar scan data.
    Call update(angle_dict) once per completed scan to refresh the display.
    angle_dict maps angle in degrees (int) -> distance in mm (int/float).
    """

    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='polar')
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_title("Lidar scan")
        plt.show(block=False)

    def update(self, angle_dict, corners=None):
        if not angle_dict:
            return
        angles = np.radians(list(angle_dict.keys()))
        distances = np.array(list(angle_dict.values()), dtype=float)

        self.ax.cla()
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_title("Lidar scan")
        self.ax.scatter(angles, distances, s=8, c='black', zorder=5)

        if corners:
            corner_angles = np.radians([a for a, _ in corners])
            corner_dists = np.array([d for _, d in corners], dtype=float)
            self.ax.scatter(corner_angles, corner_dists, s=100, c='red',
                            marker='X', edgecolors='black', zorder=10, label='Corners')

        self.ax.set_rmax(distances.max() * 1.2)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
