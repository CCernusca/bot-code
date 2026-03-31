#!/usr/bin/env python3
"""
Enable production nodes: kill any running individual nodes, then start all
node_prod_* nodes as background processes.
"""

import os
import sys
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Individual nodes that prod nodes replace
DEV_NODES = [
    "node_imu.py",
    "node_lidar.py",
    "node_pos_walls.py",
    "node_pos.py",
    "node_pos_robots.py",
    "node_predict_robots.py",
    "node_predict_ball.py",
    "node_vision.py",
    "node_time.py",
    "node_bus_display.py",
]

# Production nodes to start
PROD_NODES = [
    "node_prod_sensor.py",
    "node_prod_positioning.py",
    "node_prod_prediction.py",
    "node_prod_vision.py",
    "node_prod_communication.py",
]


def kill_nodes(node_files):
    for node in node_files:
        result = subprocess.run(
            ["pkill", "-f", f"python.*{node}"],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"[KILL]  {node}")


def start_nodes(node_files):
    procs = []
    for node in node_files:
        path = os.path.join(ROOT, node)
        if not os.path.exists(path):
            print(f"[WARN]  {node} not found, skipping.")
            continue
        p = subprocess.Popen(
            [sys.executable, path],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[START] {node}  (pid {p.pid})")
        procs.append(p)
    return procs


if __name__ == "__main__":
    print("── Stopping individual nodes ─────────────────────────────────────────")
    kill_nodes(DEV_NODES)
    time.sleep(0.5)

    print("── Starting production nodes ─────────────────────────────────────────")
    start_nodes(PROD_NODES)

    print("── Done ──────────────────────────────────────────────────────────────")
    print("Production nodes are running in the background.")
    print("Run helper/disable_prod.py to switch back to individual nodes.")
