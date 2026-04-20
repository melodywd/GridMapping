#!/usr/bin/env python
# coding: utf-8
"""
基于KITTI位姿轨迹的栅格地图构建实验
- 使用KITTI Odometry sequence 00的真实车辆轨迹
- 模拟室外环境（建筑、道路、障碍物）
- 对比无贝叶斯与贝叶斯方法
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import os

OUTPUT_DIR = 'results/kitti_map'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

POSES_PATH = 'data/kitti/dataset/poses/01.txt'


# =========================
# 1. KITTI位姿加载
# =========================
def load_kitti_poses(filepath):
    """加载KITTI位姿文件，提取2D轨迹(x, y, yaw)"""
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            T = np.array(vals).reshape(3, 4)
            x = T[0, 3]
            y = T[2, 3]  # 取z轴作为2D的y（KITTI坐标: x前, y左, z上）
            yaw = math.atan2(T[2, 0], T[0, 0])  # 从旋转矩阵提取航向角
            poses.append([x, y, yaw])
    return np.array(poses)


poses_raw = load_kitti_poses(POSES_PATH)
print(f"KITTI Sequence 05: {len(poses_raw)} 帧")
print(f"轨迹范围: x=[{poses_raw[:,0].min():.1f}, {poses_raw[:,0].max():.1f}], "
      f"y=[{poses_raw[:,1].min():.1f}, {poses_raw[:,1].max():.1f}]")


# =========================
# 2. 轨迹离散化到栅格坐标
# =========================
RESOLUTION = 2.0  # 每个栅格2米
x_min, x_max = poses_raw[:, 0].min() - 40, poses_raw[:, 0].max() + 40
y_min, y_max = poses_raw[:, 1].min() - 40, poses_raw[:, 1].max() + 40

M = int((x_max - x_min) / RESOLUTION)
N = int((y_max - y_min) / RESOLUTION)
print(f"栅格地图: {M}x{N}, 分辨率: {RESOLUTION}m/格")


def world_to_grid(x, y):
    """世界坐标转栅格坐标"""
    gi = int((x - x_min) / RESOLUTION)
    gj = int((y - y_min) / RESOLUTION)
    return gi, gj


# 将轨迹转换为栅格坐标
poses_grid = np.zeros((len(poses_raw), 3))
for i in range(len(poses_raw)):
    gi, gj = world_to_grid(poses_raw[i, 0], poses_raw[i, 1])
    poses_grid[i, 0] = gi
    poses_grid[i, 1] = gj
    poses_grid[i, 2] = poses_raw[i, 2]

# 重复行驶：正向 + 反向回环（模拟机器人行驶两次）
poses_grid_reverse = poses_grid[::-1].copy()
# 反向时航向角翻转180度
poses_grid_reverse[:, 2] = poses_grid_reverse[:, 2] + np.pi
poses_grid_double = np.vstack([poses_grid, poses_grid_reverse])
poses_raw_double = np.vstack([poses_raw, poses_raw[::-1]])
print(f"重复行驶: 轨迹长度翻倍为 {len(poses_grid_double)} 帧")


# =========================
# 3. 模拟室外环境
# =========================
def create_outdoor_map(M, N, poses_grid):
    """创建模拟室外环境（建筑、围墙、车辆等障碍物）"""
    true_map = np.zeros((M, N), dtype=np.uint8)

    # 沿轨迹两侧放置建筑物和障碍物，模拟街道环境
    np.random.seed(42)

    for t in range(0, len(poses_grid), 5):
        xi = int(poses_grid[t, 0])
        yi = int(poses_grid[t, 1])
        theta = poses_grid[t, 2]

        # 道路两侧放置建筑
        for side in [-1, 1]:
            # 垂直于行驶方向偏移
            perp_x = -math.sin(theta) * side
            perp_y = math.cos(theta) * side

            for dist in range(8, 18):
                bx = int(xi + perp_x * dist + np.random.randint(-1, 2))
                by = int(yi + perp_y * dist + np.random.randint(-1, 2))
                if 2 <= bx < M - 2 and 2 <= by < N - 2:
                    # 建筑块
                    if np.random.random() < 0.3:
                        size = np.random.randint(2, 6)
                        true_map[bx:bx+size, by:by+size] = 1

        # 偶尔放置小障碍物（模拟停放的车辆等）
        if np.random.random() < 0.1:
            for side in [-1, 1]:
                perp_x = -math.sin(theta) * side
                perp_y = math.cos(theta) * side
                bx = int(xi + perp_x * 5)
                by = int(yi + perp_y * 5)
                if 2 <= bx < M - 2 and 2 <= by < N - 2:
                    true_map[bx:bx+2, by:by+3] = 1

    # 添加几个大型建筑
    buildings = [
        (M//5, N//5, 8, 10),
        (M//3, N//2, 6, 8),
        (2*M//3, N//4, 10, 6),
        (M//2, 3*N//4, 7, 9),
    ]
    for bx, by, bw, bh in buildings:
        if bx + bw < M and by + bh < N:
            true_map[bx:bx+bw, by:by+bh] = 1

    # 清除轨迹附近的障碍物（保持道路畅通）
    for t in range(len(poses_grid)):
        xi = int(poses_grid[t, 0])
        yi = int(poses_grid[t, 1])
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = xi + dx, yi + dy
                if 0 <= nx < M and 0 <= ny < N:
                    true_map[nx, ny] = 0

    return true_map


true_map = create_outdoor_map(M, N, poses_grid_double)

# 统计障碍物比例
obstacle_ratio = true_map.sum() / (M * N)
print(f"障碍物占比: {obstacle_ratio*100:.1f}%")


# =========================
# 4. 核心函数
# =========================
# 噪声参数
NOISE_DIST_STD = 1.5     # 距离测量噪声标准差（格）
FALSE_POS_RATE = 0.02    # 假正例率（空气中检测到障碍）
FALSE_NEG_RATE = 0.05    # 假负例率（漏检真实障碍）
np.random.seed(123)


def get_ranges(true_map, X, meas_phi, rmax, add_noise=True):
    """射线投射（带噪声）"""
    MM, NN = np.shape(true_map)
    x, y, theta = X
    meas_r = rmax * np.ones(meas_phi.shape)
    for i in range(len(meas_phi)):
        for r in range(1, rmax + 1):
            xi = int(round(x + r * math.cos(theta + meas_phi[i])))
            yi = int(round(y + r * math.sin(theta + meas_phi[i])))
            if xi <= 0 or xi >= MM - 1 or yi <= 0 or yi >= NN - 1:
                meas_r[i] = r
                break
            if true_map[xi, yi] == 1:
                # 假负例：有一定概率漏检真实障碍
                if add_noise and np.random.random() < FALSE_NEG_RATE:
                    continue  # 跳过，继续射线传播
                meas_r[i] = r
                break

    if add_noise:
        # 添加距离测量噪声
        noise = np.random.normal(0, NOISE_DIST_STD, meas_r.shape)
        meas_r = meas_r + noise

        # 假正例：在空气中随机检测到障碍
        for i in range(len(meas_r)):
            if meas_r[i] < rmax and meas_r[i] > 2 and np.random.random() < FALSE_POS_RATE:
                # 在当前距离之前随机放置一个假障碍
                fake_r = np.random.randint(1, max(2, int(meas_r[i])))
                meas_r[i] = fake_r

    return meas_r


def inverse_scanner(num_rows, num_cols, x, y, theta, meas_phi, meas_r, rmax, alpha, beta):
    """逆扫描测量模型"""
    m = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            r = math.sqrt((i - x)**2 + (j - y)**2)
            phi = (math.atan2(j - y, i - x) - theta + math.pi) % (2 * math.pi) - math.pi
            k = np.argmin(np.abs(np.subtract(phi, meas_phi)))
            if (r > min(rmax, meas_r[k] + alpha / 2.0)) or (abs(phi - meas_phi[k]) > beta / 2.0):
                m[i, j] = 0.5
            elif (meas_r[k] < rmax) and (abs(r - meas_r[k]) < alpha / 2.0):
                m[i, j] = 0.7
            elif r < meas_r[k]:
                m[i, j] = 0.3
    return m


def evaluate_map(pred_map, true_map, threshold=0.5):
    """评估指标"""
    if pred_map.dtype in [np.float32, np.float64]:
        pred_binary = (pred_map >= threshold).astype(np.uint8).flatten()
    else:
        pred_binary = pred_map.flatten()
    true_binary = true_map.flatten()
    TP = np.sum((pred_binary == 1) & (true_binary == 1))
    TN = np.sum((pred_binary == 0) & (true_binary == 0))
    FP = np.sum((pred_binary == 1) & (true_binary == 0))
    FN = np.sum((pred_binary == 0) & (true_binary == 1))
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'iou': iou, 'f1': f1, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


# =========================
# 5. 传感器参数
# =========================
meas_phi = np.arange(-math.pi, math.pi, 0.08)  # 360度扫描（模拟Velodyne投影到2D）
rmax = 20  # 量程20格 = 40米
alpha = 1.0  # 障碍物宽度
beta = 0.08  # 光束角宽
threshold = 1  # 无贝叶斯计数阈值
BAYES_THRESHOLD = 0.6  # 贝叶斯占用判定阈值

# 采样帧数（每隔N帧取一帧，减少计算量）
FRAME_SKIP = 3
frame_indices = list(range(0, len(poses_grid_double), FRAME_SKIP))
T_MAX = len(frame_indices)
print(f"采样帧数: {T_MAX} (每{FRAME_SKIP}帧采样)")
print(f"测量噪声: 距离std={NOISE_DIST_STD}格, 假正例率={FALSE_POS_RATE*100:.1f}%, 假负例率={FALSE_NEG_RATE*100:.1f}%")


# =========================
# 6. 初始化
# =========================
# 无贝叶斯
count_map = np.zeros((M, N), dtype=np.int32)

# 贝叶斯
m_prior = np.multiply(0.5, np.ones((M, N)))
L0 = np.log(np.divide(m_prior, np.subtract(1, m_prior)))
L = L0.copy()

# 记录
no_bayes_maps = []
bayes_maps = []
no_bayes_metrics = []
bayes_metrics = []

# 仅每N帧记录一次指标（减少内存）
METRIC_SKIP = 5
metric_indices = []


# =========================
# 7. 主循环
# =========================
print("开始运行...")
for step, t in enumerate(frame_indices):
    xi = poses_grid_double[t, 0]
    yi = poses_grid_double[t, 1]
    theta = poses_grid_double[t, 2]
    X = [xi, yi, theta]

    # 激光扫描
    meas_r = get_ranges(true_map, X, meas_phi, rmax)

    # 无贝叶斯：终点累加
    for i in range(len(meas_phi)):
        r = meas_r[i]
        if r >= rmax:
            continue
        epx = int(round(xi + r * math.cos(theta + meas_phi[i])))
        epy = int(round(yi + r * math.sin(theta + meas_phi[i])))
        if 0 <= epx < M and 0 <= epy < N:
            count_map[epx, epy] += 1

    occ_no_bayes = (count_map >= threshold).astype(np.uint8)

    # 贝叶斯：逆扫描模型 + 对数几率更新
    invmod = inverse_scanner(M, N, xi, yi, theta, meas_phi, meas_r, rmax, alpha, beta)
    L = np.log(np.divide(invmod, np.subtract(1, invmod))) + L - L0
    m_bayes = np.exp(L)
    m_bayes = m_bayes / (1 + m_bayes)

    # 记录指标（降采样）
    if step % METRIC_SKIP == 0:
        no_bayes_metrics.append(evaluate_map(occ_no_bayes, true_map))
        bayes_metrics.append(evaluate_map(m_bayes, true_map, threshold=BAYES_THRESHOLD))
        metric_indices.append(step)

    # 记录地图（更低频）
    if step % (METRIC_SKIP * 10) == 0:
        no_bayes_maps.append(occ_no_bayes.copy())
        bayes_maps.append(m_bayes.copy())

    if step % 50 == 0:
        nb_m = no_bayes_metrics[-1] if no_bayes_metrics else None
        b_m = bayes_metrics[-1] if bayes_metrics else None
        print(f"  帧 {step}/{T_MAX} (原始帧{t})", end="")
        if nb_m and b_m:
            print(f"  无贝叶斯IoU={nb_m['iou']:.3f}  贝叶斯IoU={b_m['iou']:.3f}")
        else:
            print()

print("运行完成！")


# =========================
# 8. 结果输出
# =========================
nb_final = no_bayes_metrics[-1]
b_final = bayes_metrics[-1]

print("\n" + "=" * 60)
print("KITTI轨迹栅格地图构建 - 最终结果对比")
print("=" * 60)
print(f"{'指标':<12} {'无贝叶斯':>10} {'贝叶斯':>10} {'提升':>10}")
print("-" * 60)
for key in ['accuracy', 'precision', 'recall', 'iou', 'f1']:
    diff = b_final[key] - nb_final[key]
    sign = '+' if diff > 0 else ''
    print(f"{key:<12} {nb_final[key]:>10.4f} {b_final[key]:>10.4f} {sign}{diff:>9.4f}")
print("=" * 60)
print(f"\n混淆矩阵对比:")
print(f"  无贝叶斯: TP={nb_final['TP']}, TN={nb_final['TN']}, FP={nb_final['FP']}, FN={nb_final['FN']}")
print(f"  贝叶斯:   TP={b_final['TP']}, TN={b_final['TN']}, FP={b_final['FP']}, FN={b_final['FN']}")


# =========================
# 9. 可视化
# =========================

# 9.1 最终地图对比
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(1 - true_map, cmap='gray', origin='lower', vmin=0, vmax=1)
traj_gx = [poses_grid_double[t, 1] for t in frame_indices]
traj_gy = [poses_grid_double[t, 0] for t in frame_indices]
axes[0].plot(traj_gx, traj_gy, 'r-', linewidth=0.5, alpha=0.7)
axes[0].set_title(f'真实地图 ({M}x{N}, {RESOLUTION}m/格)')

axes[1].imshow(1 - no_bayes_maps[-1], cmap='gray', origin='lower', vmin=0, vmax=1)
axes[1].plot(traj_gx, traj_gy, 'r-', linewidth=0.5, alpha=0.7)
axes[1].set_title(f'无贝叶斯 (IoU={nb_final["iou"]:.3f})')

axes[2].imshow(bayes_maps[-1], cmap='gray', origin='lower', vmin=0, vmax=1)
axes[2].plot(traj_gx, traj_gy, 'r-', linewidth=0.5, alpha=0.7)
axes[2].set_title(f'贝叶斯 (IoU={b_final["iou"]:.3f})')

fig.suptitle('KITTI轨迹栅格地图构建对比', fontsize=14)
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/kitti_map_comparison.png', dpi=150)
print(f"地图对比图: {OUTPUT_DIR}/kitti_map_comparison.png")


# 9.2 指标曲线
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metric_steps = [frame_indices[i] for i in metric_indices]

for ax, metric in zip(axes, ['iou', 'accuracy', 'f1']):
    nb_vals = [m[metric] for m in no_bayes_metrics]
    b_vals = [m[metric] for m in bayes_metrics]
    ax.plot(metric_steps, nb_vals, 'r-', label='无贝叶斯', linewidth=1.5)
    ax.plot(metric_steps, b_vals, 'b-', label='贝叶斯', linewidth=1.5)
    ax.set_xlabel('原始帧号')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} 随时间变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle('KITTI轨迹 - 指标随时间变化', fontsize=14)
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/kitti_metrics_comparison.png', dpi=150)
print(f"指标曲线: {OUTPUT_DIR}/kitti_metrics_comparison.png")


# 9.3 KITTI轨迹图
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(poses_raw_double[:, 0], poses_raw_double[:, 1], 'b-', linewidth=0.8, label='KITTI Seq.05 往返轨迹')
ax.scatter(poses_raw_double[0, 0], poses_raw_double[0, 1], c='g', s=100, zorder=5, label='起点')
ax.scatter(poses_raw_double[-1, 0], poses_raw_double[-1, 1], c='r', s=100, zorder=5, label='终点')
ax.axvline(x=poses_raw[-1, 0], color='orange', linestyle='--', alpha=0.5, label='往返分界')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('KITTI Odometry Sequence 05 轨迹（含环路）')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
fig.savefig(f'{OUTPUT_DIR}/kitti_trajectory.png', dpi=150)
print(f"轨迹图: {OUTPUT_DIR}/kitti_trajectory.png")


# 9.4 贝叶斯地图动画（降采样）
print("正在生成动画...")
anim_indices = list(range(0, len(bayes_maps), max(1, len(bayes_maps)//50)))

if len(anim_indices) > 1:
    belief_fig = plt.figure()
    belief_ax = belief_fig.add_subplot(111)

    def belief_update(i):
        belief_ax.clear()
        belief_ax.imshow(bayes_maps[i], cmap='gray', origin='lower', vmin=0, vmax=1)
        # 绘制到当前帧为止的轨迹
        current_step = anim_indices[i] * METRIC_SKIP * 10
        t_end = min(frame_indices[min(current_step, len(frame_indices)-1)], len(poses_raw_double)-1)
        belief_ax.plot(poses_raw_double[:t_end, 0], poses_raw_double[:t_end, 1], 'r-', linewidth=0.5)
        # 标记往返分界线
        if t_end >= len(poses_raw):
            belief_ax.axvline(x=poses_raw[-1, 0], color='orange', linestyle='--', alpha=0.5)
        belief_ax.set_title(f'贝叶斯栅格地图 (帧 {t_end}/{len(poses_raw_double)})')

    belief_anim = anim.FuncAnimation(
        belief_fig, belief_update, frames=len(anim_indices), repeat=False
    )
    belief_anim.save(f'{OUTPUT_DIR}/kitti_bayes_animation.mp4', writer='ffmpeg', fps=10)
    print(f"动画: {OUTPUT_DIR}/kitti_bayes_animation.mp4")
else:
    print("地图帧不足，跳过动画生成")

plt.show()
