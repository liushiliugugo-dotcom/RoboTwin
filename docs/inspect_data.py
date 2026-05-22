#!/usr/bin/env python3
"""
RoboTwin 数据采集结果可视化工具。

支持 PKL（轨迹规划数据）和 HDF5（逐帧观测数据）两种文件的查看和导出。

────────────────────────────────────────────────────────────
PKL 文件（_traj_data/episode*.pkl）
────────────────────────────────────────────────────────────
PKL 是 Python pickle 格式，存储运动规划器（cuRobo）生成的关节空间轨迹。
只能用本工具查看，HDFView / H5Web 无法打开。

  查看结构与数值：
      python docs/inspect_data.py pkl data/<task>/<config>/_traj_data/episode0.pkl

  查看结构 + 绘制关节分段轨迹图（标注每段动作名称）：
      python docs/inspect_data.py pkl <path> --plot

  查看结构 + 绘制 EE 3D 空间轨迹图（含高度变化+夹爪状态）：
      python docs/inspect_data.py pkl <path> --plot3d

  全部：
      python docs/inspect_data.py pkl <path> --plot --plot3d

  输出内容：
      - 左/右臂轨迹段数、总步数、仿真时间
      - 每段自动映射到 play_once() 的具体动作（grasp / lift / place）
      - 逐关节的首帧 → 末帧 → 变化量
      - position / velocity 数值范围
      - 段间连续性检查（末帧与下段首帧是否衔接）
      - 自动关联同名 HDF5 读取末端姿态轨迹和夹爪开合
      - (--plot)  关节分段轨迹图 PNG: 每段不同颜色 + 段间虚线 + 动作标签
      - (--plot3d) EE 3D 空间轨迹图 PNG: 3D路径 + 高度/夹爪时间曲线

────────────────────────────────────────────────────────────
HDF5 文件（data/episode*.hdf5）
────────────────────────────────────────────────────────────
HDF5 是二进制科学数据格式，存储逐帧的 RGB 图像（JPEG 编码）、关节角度、末端姿态。
也可用 HDFView (sudo apt install hdfview) 或 VS Code H5Web 插件交互式浏览。

  仅查看结构：
      python docs/inspect_data.py hdf5 data/<task>/<config>/data/episode0.hdf5

  查看结构 + 导出首帧图像（PNG）：
      python docs/inspect_data.py hdf5 data/<task>/<config>/data/episode0.hdf5 --img

  查看结构 + 绘制关节轨迹图（PNG）：
      python docs/inspect_data.py hdf5 data/<task>/<config>/data/episode0.hdf5 --plot

  查看结构 + 导出前 N 帧为图像序列：
      python docs/inspect_data.py hdf5 data/<task>/<config>/data/episode0.hdf5 --video 30

  可同时使用多个选项：
      python docs/inspect_data.py hdf5 <path> --img --plot --video 30

────────────────────────────────────────────────────────────
环境要求
────────────────────────────────────────────────────────────
PKL 模式：  无需任何 conda 环境，base env 即可运行
HDF5 模式： 需先激活 RoboTwin 环境: conda activate RoboTwin

VS Code 快捷操作：右键 .hdf5 文件 → Open With → 搜索 H5Web 可直接内联查看图像
"""

import argparse
import os
import pickle
import numpy as np

_MISSING = "请先激活 RoboTwin 环境: conda activate RoboTwin"


# ── 辅助函数 ──────────────────────────────────────────────

def _inspect_value(value, indent=0):
    prefix = " " * indent
    if isinstance(value, dict):
        print(f"{prefix}dict, keys={list(value.keys())}")
        for k, v in value.items():
            if isinstance(v, np.ndarray):
                print(f"{prefix}  {k}: ndarray shape={v.shape} dtype={v.dtype} "
                      f"range=[{v.min():.4f}, {v.max():.4f}]")
            elif isinstance(v, list):
                print(f"{prefix}  {k}: list len={len(v)}")
                if v and isinstance(v[0], dict):
                    print(f"{prefix}    [0] keys={list(v[0].keys())}")
                elif v:
                    print(f"{prefix}    [0]: {type(v[0]).__name__} = {str(v[0])[:80]}")
            elif isinstance(v, str):
                print(f"{prefix}  {k}: str = '{v[:80]}'")
            else:
                print(f"{prefix}  {k}: {type(v).__name__} = {v}")
    elif isinstance(value, np.ndarray):
        print(f"{prefix}ndarray shape={value.shape} dtype={value.dtype} "
              f"range=[{value.min():.4f}, {value.max():.4f}]")
    elif isinstance(value, list):
        print(f"{prefix}list len={len(value)}")
        if value:
            print(f"{prefix}  element[0] type: {type(value[0]).__name__}")
            if isinstance(value[0], dict):
                print(f"{prefix}  element[0] keys: {list(value[0].keys())}")
    else:
        print(f"{prefix}{type(value).__name__} (scalar)")


# ── PKL ───────────────────────────────────────────────────

def inspect_pkl(filepath, plot_joints=False, plot3d=False):
    print(f"\n{'='*65}")
    print(f"PKL: {filepath}")
    print(f"Size: {os.path.getsize(filepath) / 1024:.1f} KB")
    print(f"{'='*65}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    left = data.get('left_joint_path', [])
    right = data.get('right_joint_path', [])

    # ── 摘要 ──
    print(f"\n  左臂轨迹段数: {len(left)}   右臂轨迹段数: {len(right)}")

    active_arm = 'left' if len(left) > len(right) else 'right'
    segments = left if active_arm == 'left' else right

    total_steps = sum(seg['position'].shape[0] for seg in segments)
    sim_time = total_steps / 250.0  # 1/250 timestep
    print(f"  主要工作臂: {active_arm}   总步数: {total_steps}   仿真时间: {sim_time:.2f}s")

    # ── 轨迹动作映射 ──
    n = len(segments)
    if n == 5:
        action_map = [
            (0, "pre-grasp"),
            (1, "grasp"),
            (2, "lift"),
            (3, "pre-place"),
            (4, "place"),
        ]
    elif n == 2:
        action_map = [(0, "grasp"), (1, "place")]
    elif n == 1:
        action_map = [(0, "single")]
    elif n > 5:
        action_map = None
    else:
        action_map = [(i, f"seg{i}") for i in range(n)]

    # ── 逐段打印 ──
    for i, seg in enumerate(segments):
        pos = seg['position']
        vel = seg['velocity']
        n_steps = pos.shape[0]
        n_joints = pos.shape[1]

        # 动作标签
        label = ""
        if action_map:
            for idx, desc in action_map:
                if i == idx:
                    labels_full = {
                        "pre-grasp": "grasp_actor — pre-grasp",
                        "grasp":     "grasp_actor — final grasp",
                        "lift":      "move_by_displacement / lift",
                        "pre-place": "place_actor — pre-place",
                        "place":     "place_actor — final place",
                    }
                    full = labels_full.get(desc, desc)
                    label = f"  ← {full}"
                    break

        print(f"\n  ┌─ 段 {i}/{n} ─────────────────────────────{label}")
        print(f"  │ 状态: {seg['status']}   |   步数: {n_steps}   |   关节数: {n_joints}")

        # 关节数值表：首帧 → 末帧 → 变化量
        first = pos[0]
        last = pos[-1]
        delta = last - first

        print(f"  │")
        print(f"  │  关节      首帧        →   末帧         变化量")
        for j in range(n_joints):
            print(f"  │  J{j}    {first[j]:+8.4f}    {last[j]:+8.4f}    {delta[j]:+8.4f}")
        print(f"  │")
        print(f"  │  position range: [{pos.min():+.4f}, {pos.max():+.4f}]")
        print(f"  │  velocity range: [{vel.min():+.4f}, {vel.max():+.4f}]")

        # 段间连续性检查
        if i > 0:
            prev_last = segments[i - 1]['position'][-1]
            jump = np.linalg.norm(pos[0] - prev_last)
            status = "✓ 连续" if jump < 0.01 else f"⚠ 跳跃 {jump:.4f}"
            print(f"  │  → 接续上段: {status}")

    # ── 末端姿态轨迹（如果 HDF5 存在则读取） ──
    # 尝试从同名 HDF5 读取末端姿态
    hdf5_path = _find_hdf5(filepath)
    if hdf5_path:
        _print_endpose_summary(hdf5_path, active_arm)
    else:
        print(f"\n  (未找到对应 HDF5，无法显示末端姿态轨迹)")

    # ── 闲置臂 ──
    if active_arm == 'right' and len(left) == 0:
        print(f"\n  ℹ 左臂闲置: 此任务仅使用右臂")
    elif active_arm == 'left' and len(right) == 0:
        print(f"\n  ℹ 右臂闲置: 此任务仅使用左臂")

    # ── 绘图 (可选) ──
    if plot_joints:
        _plot_pkl_joints(filepath, left, right, action_map)
    if plot3d and hdf5_path:
        _plot_pkl_ee3d(hdf5_path, active_arm, action_map, segments)


def _plot_pkl_joints(filepath, left, right, action_map):
    """从 PKL 数据绘制关节轨迹，标注动作分段"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base_dir = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(base_dir, f"_inspect_{base_name}", "pkl_joint_trajectories.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    active_segments = left if len(left) > len(right) else right
    active_label = 'L' if len(left) > len(right) else 'R'
    n_joints = active_segments[0]['position'].shape[1]
    n_segs = len(active_segments)

    # 拼接所有段并记录段边界
    all_pos = []
    boundaries = [0]  # 每段开始的全局步数索引
    for seg in active_segments:
        all_pos.append(seg['position'])
        boundaries.append(boundaries[-1] + seg['position'].shape[0])
    all_pos = np.concatenate(all_pos, axis=0)

    fig, axes = plt.subplots(n_joints, 1, figsize=(14, n_joints * 2.2), sharex=True)
    if n_joints == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_segs))
    time = np.arange(all_pos.shape[0]) / 250.0

    for j in range(n_joints):
        ax = axes[j]
        # 逐段绘制不同颜色
        for s in range(n_segs):
            t0, t1 = boundaries[s], boundaries[s + 1]
            ax.plot(time[t0:t1], all_pos[t0:t1, j], color=colors[s], linewidth=1.0,
                    label=(action_map[s][1][:30] if action_map and s < len(action_map) else f'Seg {s}'))
        # 段边界虚线
        for s in range(1, n_segs):
            ax.axvline(x=time[boundaries[s]], color='gray', linestyle='--', alpha=0.5, linewidth=0.6)
        ax.set_ylabel(f'{active_label} J{j}', fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=7, ncol=min(n_segs, 3), loc='upper right')
    axes[-1].set_xlabel('Time (s)', fontsize=9)
    fig.suptitle(f'PKL Joint Trajectories with Action Segments\n{os.path.basename(filepath)}', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n✅ 关节分段轨迹图: {out_path}")


def _plot_pkl_ee3d(hdf5_path, active_arm, action_map, segments):
    """从配对 HDF5 读取末端姿态，绘制 3D 空间轨迹"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    import h5py
    base_dir = os.path.dirname(hdf5_path)
    base_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    out_path = os.path.join(base_dir, f"_inspect_{base_name}", "pkl_ee_3d.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(hdf5_path, 'r') as f:
        prefix = 'left' if active_arm == 'left' else 'right'
        key = f'endpose/{prefix}_endpose'
        if key not in f:
            print("\n⚠️  HDF5 中无末端姿态数据，跳过 3D 轨迹")
            return
        ee = f[key][:, :3]  # [T, 3] xyz positions
        gripper_key = f'endpose/{prefix}_gripper'
        gripper = f[gripper_key][:] if gripper_key in f else None

    # 根据 save_freq 和每段步数估算每段在 HDF5 帧中的边界
    save_freq = 15
    seg_boundaries_hdf5 = [0]
    for seg in segments:
        hdf5_steps = seg['position'].shape[0] // save_freq + 1
        seg_boundaries_hdf5.append(seg_boundaries_hdf5[-1] + hdf5_steps)
    seg_boundaries_hdf5[-1] = min(seg_boundaries_hdf5[-1], len(ee))

    n_segs = len(segments)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_segs, 1)))

    fig = plt.figure(figsize=(14, 6))

    # ── 子图1: 3D 轨迹 ──
    ax = fig.add_subplot(121, projection='3d')
    for s in range(n_segs):
        t0, t1 = seg_boundaries_hdf5[s], min(seg_boundaries_hdf5[s + 1], len(ee))
        if t1 > t0:
            label = action_map[s][1][:25] if action_map and s < len(action_map) else f'Seg {s}'
            ax.plot(ee[t0:t1, 0], ee[t0:t1, 1], ee[t0:t1, 2],
                    color=colors[s], linewidth=1.5, label=label)
    # 起点/终点标记
    ax.scatter(*ee[0], color='green', s=60, marker='o', label='Start')
    ax.scatter(*ee[-1], color='red', s=60, marker='*', label='End')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('End-Effector 3D Trajectory')
    ax.legend(fontsize=7)

    # ── 子图2: 夹爪 + Z轴投影 ──
    ax2 = fig.add_subplot(122)
    time = np.arange(len(ee)) * (1/250) * 15
    ax2.plot(time, ee[:, 2], color='blue', linewidth=1, label='EE Height (Z)')
    if gripper is not None:
        ax2_dual = ax2.twinx()
        ax2_dual.plot(time, gripper, color='red', linewidth=0.8, alpha=0.6, label='Gripper')
        ax2_dual.set_ylabel('Gripper (0=close, 1=open)', color='red', fontsize=8)
        ax2_dual.set_ylim(-0.05, 1.05)
    for s in range(1, n_segs):
        t0 = seg_boundaries_hdf5[s]
        if t0 < len(time):
            ax2.axvline(x=time[t0], color='gray', linestyle='--', alpha=0.4, linewidth=0.6)
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Height Z (m)')
    ax2.set_title('EE Height + Gripper State')
    ax2.grid(True, alpha=0.25)

    fig.suptitle(f'PKL End-Effector Trajectory\n{base_name}', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ EE 3D 轨迹图: {out_path}")


def _find_hdf5(filepath):
    """从 pkl 路径推导同 episode 的 hdf5 路径"""
    # _traj_data/episodeN.pkl →  ../data/episodeN.hdf5
    pkl_dir = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    ep_name = os.path.splitext(base)[0]  # episode0
    hdf5_path = os.path.join(os.path.dirname(pkl_dir), 'data', f'{ep_name}.hdf5')
    if os.path.exists(hdf5_path):
        return hdf5_path
    return None


def _print_endpose_summary(hdf5_path, active_arm):
    import h5py
    print(f"\n  ── 末端姿态轨迹 (from {os.path.basename(hdf5_path)}) ──")
    with h5py.File(hdf5_path, 'r') as f:
        prefix = 'left' if active_arm == 'left' else 'right'
        key = f'endpose/{prefix}_endpose'
        if key in f:
            ee = f[key][:]
            p_min = ee[:, :3].min(axis=0)
            p_max = ee[:, :3].max(axis=0)
            print(f"  末端位置范围: x[{p_min[0]:+.3f}, {p_max[0]:+.3f}]  "
                  f"y[{p_min[1]:+.3f}, {p_max[1]:+.3f}]  z[{p_min[2]:+.3f}, {p_max[2]:+.3f}]")
            print(f"  起点: {ee[0, :3].round(3)} → 终点: {ee[-1, :3].round(3)}")
        gripper_key = f'endpose/{prefix}_gripper'
        if gripper_key in f:
            g = f[gripper_key][:]
            print(f"  夹爪开度: start={g[0]:.2f} end={g[-1]:.2f}  "
                  f"(0=闭合, 1=全开)")


# ── HDF5 ──────────────────────────────────────────────────

def inspect_hdf5(filepath, export_img=False, plot_joints=False, video_frames=0):
    import h5py

    print(f"\n{'='*60}")
    print(f"HDF5: {filepath}")
    print(f"Size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
    print(f"{'='*60}")

    with h5py.File(filepath, 'r') as f:
        print("\nDataset tree:")
        _print_tree(f)

        rgb_keys = [k for k in _all_keys(f) if k.endswith('/rgb')]
        joint_keys = [k for k in _all_keys(f) if 'joint_action' in k and 'vector' not in k]

        for rgb_key in rgb_keys:
            data = f[rgb_key]
            total_mb = sum(len(d) for d in data[:]) / (1024 * 1024)
            print(f"\n📷 {rgb_key}: {data.shape[0]} frames, {total_mb:.1f} MB")

        for joint_key in joint_keys:
            ds = f[joint_key]
            arr = ds[:]
            print(f"🔧 {joint_key}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")

        if export_img:
            _export_images(f, filepath)
        if plot_joints:
            _plot_joints(f, filepath)
        if video_frames > 0:
            _export_frames(f, filepath, video_frames)


def _print_tree(f, prefix=''):
    h5py = __import__('h5py')
    for key in f.keys():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(f[key], h5py.Group):
            print(f"  📁 {path}/")
            _print_tree(f[key], path)
        else:
            ds = f[key]
            size_mb = ds.size * ds.dtype.itemsize / (1024 * 1024)
            shape_str = ', '.join(map(str, ds.shape))
            print(f"  📄 {path}: ({shape_str})  {ds.dtype}  {size_mb:.2f} MB")


def _all_keys(f, prefix=''):
    import h5py
    keys = []
    for key in f.keys():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(f[key], h5py.Group):
            keys.extend(_all_keys(f[key], path))
        else:
            keys.append(path)
    return keys


def _export_images(f, filepath):
    import cv2

    base_dir = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_dir = os.path.join(base_dir, f"_inspect_{base_name}")
    os.makedirs(out_dir, exist_ok=True)

    for key in _all_keys(f):
        if key.endswith('/rgb'):
            raw = f[key][0]
            try:
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                cam_name = key.replace('/', '_').replace('observation_', '')
                out_path = os.path.join(out_dir, f"{cam_name}_frame0.png")
                cv2.imwrite(out_path, img)
                print(f"\n✅ Exported: {out_path} ({img.shape[1]}x{img.shape[0]})")
            except Exception as e:
                print(f"\n⚠️  Decode {key} failed: {e}")


def _plot_joints(f, filepath):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base_dir = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(base_dir, f"_inspect_{base_name}", "joint_trajectories.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    has_left = 'joint_action/left_arm' in f
    has_right = 'joint_action/right_arm' in f

    n_plots = (7 if has_left else 0) + (7 if has_right else 0)
    if n_plots == 0:
        print("\n⚠️  No joint data found")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, n_plots * 2), sharex=True)
    if n_plots == 1:
        axes = [axes]

    def _plot_arm(axes, joint_data, gripper_data, label, row_offset):
        T = joint_data.shape[0]
        time = np.arange(T) * (1/250) * 15  # save_freq=15, dt=1/250
        colors = plt.cm.viridis(np.linspace(0, 1, joint_data.shape[1]))
        for j in range(joint_data.shape[1]):
            ax = axes[row_offset + j]
            ax.plot(time, joint_data[:, j], color=colors[j], linewidth=0.8)
            ax.set_ylabel(f'{label} J{j}', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        ax = axes[row_offset + joint_data.shape[1]]
        ax.plot(time, gripper_data[:], color='red', linewidth=0.8)
        ax.set_ylabel(f'{label} Gripper', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    if has_left:
        _plot_arm(axes, f['joint_action/left_arm'][:], f['joint_action/left_gripper'][:], 'L', 0)
    if has_right:
        _plot_arm(axes, f['joint_action/right_arm'][:], f['joint_action/right_gripper'][:], 'R', 7 if has_left else 0)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    fig.suptitle(f'Joint Trajectories\n{os.path.basename(filepath)}', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n✅ Exported: {out_path}")


def _export_frames(f, filepath, n_frames):
    import cv2

    base_dir = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_dir = os.path.join(base_dir, f"_inspect_{base_name}", "frames")
    os.makedirs(out_dir, exist_ok=True)

    for key in _all_keys(f):
        if key.endswith('/rgb'):
            cam_name = key.replace('/', '_').replace('observation_', '')
            rgb_data = f[key][:]
            actual_n = min(n_frames, len(rgb_data))
            for i in range(actual_n):
                img = cv2.imdecode(np.frombuffer(rgb_data[i], np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(out_dir, f"{cam_name}_frame{i:04d}.png"), img)
            print(f"✅ Exported {cam_name}: {actual_n} frames → {out_dir}")


# ── CLI ───────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RoboTwin 数据可视化工具')
    parser.add_argument('type', choices=['pkl', 'hdf5'])
    parser.add_argument('path', help='文件路径')
    parser.add_argument('--plot', action='store_true', help='绘制关节分段轨迹图 (PKL/HDF5 均可用)')
    parser.add_argument('--img', action='store_true', help='导出首帧图像 (仅 HDF5)')
    parser.add_argument('--plot3d', action='store_true', help='绘制 EE 3D 空间轨迹 (仅 PKL, 需配对 HDF5)')
    parser.add_argument('--video', type=int, default=0, help='导出前 N 帧 (仅 HDF5)')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"❌ File not found: {args.path}")
        exit(1)

    if args.type == 'pkl':
        inspect_pkl(args.path, plot_joints=args.plot, plot3d=args.plot3d)
    elif args.type == 'hdf5':
        inspect_hdf5(args.path, export_img=args.img, plot_joints=args.plot, video_frames=args.video)
