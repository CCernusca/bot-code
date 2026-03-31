#!/usr/bin/env python3
"""
Disable production nodes: kill any running node_prod_* nodes, then start all
individual nodes as background processes.
"""

import os
import sys
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Production nodes to kill
PROD_NODES = [
    "node_prod_sensor.py",
    "node_prod_positioning.py",
    "node_prod_prediction.py",
    "node_prod_vision.py",
    "node_prod_communication.py",
]

# Individual nodes to start
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
    print("── Stopping production nodes ─────────────────────────────────────────")
    kill_nodes(PROD_NODES)
    time.sleep(0.5)

    print("── Starting individual nodes ─────────────────────────────────────────")
    start_nodes(DEV_NODES)

    print("── Done ──────────────────────────────────────────────────────────────")
    print("Individual nodes are running in the background.")
    print("Run helper/enable_prod.py to switch back to production nodes.")
