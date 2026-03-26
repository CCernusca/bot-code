from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from lidar_utils.lidar_analysis import simple_corners, intersection_corners
import json
import math

# Corners within this distance (metres) of each other are considered "similar".
# If the largest such group is a majority (> half of all detected corners),
# lone outliers outside that group are discarded.
OUTLIER_DIST = 0.15

mb = TelemetryBroker()

_walls = []   # latest walls from broker, used for intersection_corners


def _filter_outliers(corner_xy):
    """
    If more than 2 corners are detected and a majority cluster exists,
    discard corners that do not belong to it.
    """
    n = len(corner_xy)
    if n <= 2:
        return corner_xy

    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(corner_xy[i][0] - corner_xy[j][0],
                          corner_xy[i][1] - corner_xy[j][1]) <= OUTLIER_DIST:
                adj[i].add(j)
                adj[j].add(i)

    visited = [False] * n
    components = []
    for i in range(n):
        if visited[i]:
            continue
        component, queue = [], [i]
        visited[i] = True
        while queue:
            curr = queue.pop()
            component.append(curr)
            for nb in adj[curr]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(component)

    largest = max(components, key=len)
    if len(largest) > n / 2:
        return [corner_xy[i] for i in largest]
    return corner_xy


def on_update(key, value):
    global _walls

    if value is None:
        return

    if key == "lidar":
        try:
            raw = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return
        sorted_angles = sorted(int(k) for k in raw.keys())
        points = [
            (
                (raw[str(a)] / 1000) * math.cos(math.radians(a)),
                (raw[str(a)] / 1000) * math.sin(math.radians(a))
            )
            for a in sorted_angles
        ]
        corner_xy = _filter_outliers(simple_corners(points))
        corners = [
            (int(round(math.degrees(math.atan2(y, x)) % 360)), int(round(math.hypot(x, y) * 1000)))
            for x, y in corner_xy
        ]
        mb.set("depth_corners", json.dumps(corners))

    elif key == "lidar_walls":
        try:
            _walls = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return
        pts = intersection_corners(_walls)
        mb.set("wall_corners", json.dumps([[round(x, 3), round(y, 3)] for x, y in pts]))


if __name__ == "__main__":
    mb.setcallback(["lidar", "lidar_walls"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping corner detection.")
        mb.close()
