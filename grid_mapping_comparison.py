#!/usr/bin/env python
# coding: utf-8
"""
无贝叶斯 vs 贝叶斯 栅格地图构建对比实验
- 无贝叶斯：终点直接累加 + 二值化
- 贝叶斯：逆扫描模型 + 对数几率更新
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import os

OUTPUT_DIR = 'results/comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 公共函数
# =========================
def get_ranges(true_map, X, meas_phi, rmax):
    """射线投射：模拟激光雷达扫描"""
    M, N = np.shape(true_map)
    x, y, theta = X
    meas_r = rmax * np.ones(meas_phi.shape)

    for i in range(len(meas_phi)):
        for r in range(1, rmax + 1):
            xi = int(round(x + r * math.cos(theta + meas_phi[i])))
            yi = int(round(y + r * math.sin(theta + meas_phi[i])))
            if xi <= 0 or xi >= M - 1 or yi <= 0 or yi >= N - 1:
                meas_r[i] = r
                break
            if true_map[xi, yi] == 1:
                meas_r[i] = r
                break
    return meas_r


def inverse_scanner(num_rows, num_cols, x, y, theta, meas_phi, meas_r, rmax, alpha, beta):
    """逆扫描测量模型：返回每个栅格的占用概率"""
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
    """计算评估指标"""
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
# 仿真参数
# =========================
T_MAX = 150
time_steps = np.arange(T_MAX)
x_0 = [30, 30, 0]
u = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
u_i = 1
w = np.multiply(0.3, np.ones(len(time_steps)))

M, N = 50, 60
true_map = np.zeros((M, N), dtype=np.uint8)
true_map[0:10, 0:10] = 1
true_map[30:35, 40:45] = 1
true_map[3:6, 40:60] = 1
true_map[20:30, 25:29] = 1
true_map[40:50, 5:25] = 1

meas_phi = np.arange(-0.4, 0.4, 0.05)
rmax = 30
alpha = 1
beta = 0.05
threshold = 1

# =========================
# 无贝叶斯初始化
# =========================
count_map = np.zeros((M, N), dtype=np.int32)

# =========================
# 贝叶斯初始化
# =========================
m_prior = np.multiply(0.5, np.ones((M, N)))
L0 = np.log(np.divide(m_prior, np.subtract(1, m_prior)))
L = L0.copy()

# =========================
# 状态轨迹
# =========================
x = np.zeros((3, len(time_steps)))
x[:, 0] = x_0

# 记录过程
no_bayes_maps = []
bayes_maps = []
no_bayes_metrics = []
bayes_metrics = []


# =========================
# 初始帧
# =========================
meas_r = get_ranges(true_map, x[:, 0], meas_phi, rmax)

# 无贝叶斯：终点累加
for i in range(len(meas_phi)):
    r = meas_r[i]
    if r >= rmax:
        continue
    xi = int(round(x[0, 0] + r * math.cos(x[2, 0] + meas_phi[i])))
    yi = int(round(x[1, 0] + r * math.sin(x[2, 0] + meas_phi[i])))
    if 0 <= xi < M and 0 <= yi < N:
        count_map[xi, yi] += 1

occ_no_bayes = (count_map >= threshold).astype(np.uint8)
no_bayes_maps.append(occ_no_bayes.copy())

# 贝叶斯：逆扫描模型 + 对数几率更新
invmod = inverse_scanner(M, N, x[0, 0], x[1, 0], x[2, 0], meas_phi, meas_r, rmax, alpha, beta)
L = np.log(np.divide(invmod, np.subtract(1, invmod))) + L - L0
m_bayes = np.exp(L)
m_bayes = m_bayes / (1 + m_bayes)
bayes_maps.append(m_bayes.copy())

no_bayes_metrics.append(evaluate_map(occ_no_bayes, true_map))
bayes_metrics.append(evaluate_map(m_bayes, true_map))


# =========================
# 主循环
# =========================
for t in range(1, len(time_steps)):
    # 机器人运动（两种方法共用同一轨迹）
    move = np.add(x[0:2, t - 1], u[:, u_i])
    if ((move[0] >= M - 1) or (move[1] >= N - 1) or
        (move[0] <= 0) or (move[1] <= 0) or
        true_map[int(round(move[0])), int(round(move[1]))] == 1):
        x[:, t] = x[:, t - 1]
        u_i = (u_i + 1) % 4
    else:
        x[0:2, t] = move
    x[2, t] = (x[2, t - 1] + w[t]) % (2 * math.pi)

    # 激光扫描
    meas_r = get_ranges(true_map, x[:, t], meas_phi, rmax)

    # 无贝叶斯：终点累加
    for i in range(len(meas_phi)):
        r = meas_r[i]
        if r >= rmax:
            continue
        xi = int(round(x[0, t] + r * math.cos(x[2, t] + meas_phi[i])))
        yi = int(round(x[1, t] + r * math.sin(x[2, t] + meas_phi[i])))
        if 0 <= xi < M and 0 <= yi < N:
            count_map[xi, yi] += 1

    occ_no_bayes = (count_map >= threshold).astype(np.uint8)
    no_bayes_maps.append(occ_no_bayes.copy())

    # 贝叶斯：逆扫描模型 + 对数几率更新
    invmod = inverse_scanner(M, N, x[0, t], x[1, t], x[2, t], meas_phi, meas_r, rmax, alpha, beta)
    L = np.log(np.divide(invmod, np.subtract(1, invmod))) + L - L0
    m_bayes = np.exp(L)
    m_bayes = m_bayes / (1 + m_bayes)
    bayes_maps.append(m_bayes.copy())

    # 评估
    no_bayes_metrics.append(evaluate_map(occ_no_bayes, true_map))
    bayes_metrics.append(evaluate_map(m_bayes, true_map))


# =========================
# 最终结果输出
# =========================
nb_final = no_bayes_metrics[-1]
b_final = bayes_metrics[-1]

print("=" * 60)
print(f"{'指标':<12} {'无贝叶斯':>10} {'贝叶斯':>10} {'提升':>10}")
print("=" * 60)
for key in ['accuracy', 'precision', 'recall', 'iou', 'f1']:
    diff = b_final[key] - nb_final[key]
    sign = '+' if diff > 0 else ''
    print(f"{key:<12} {nb_final[key]:>10.4f} {b_final[key]:>10.4f} {sign}{diff:>9.4f}")
print("=" * 60)


# =========================
# 指标随时间变化曲线
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ts = time_steps

for ax, metric in zip(axes, ['iou', 'accuracy', 'f1']):
    nb_vals = [m[metric] for m in no_bayes_metrics]
    b_vals = [m[metric] for m in bayes_metrics]
    ax.plot(ts, nb_vals, 'r-', label='无贝叶斯', linewidth=1.5)
    ax.plot(ts, b_vals, 'b-', label='贝叶斯', linewidth=1.5)
    ax.set_xlabel('时间步')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} 随时间变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle('无贝叶斯 vs 贝叶斯 栅格地图构建对比')
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/metrics_comparison.png', dpi=150)
print(f"指标对比图: {OUTPUT_DIR}/metrics_comparison.png")


# =========================
# 最终地图对比
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(1 - true_map, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[0].set_title('真实地图')
axes[1].imshow(1 - no_bayes_maps[-1], cmap='gray', origin='lower', vmin=0, vmax=1)
axes[1].set_title(f'无贝叶斯 (IoU={nb_final["iou"]:.3f})')
axes[2].imshow(1 - bayes_maps[-1], cmap='gray', origin='lower', vmin=0, vmax=1)
axes[2].set_title(f'贝叶斯 (IoU={b_final["iou"]:.3f})')
for ax in axes:
    ax.plot(x[1], x[0], 'bx-', markersize=1)
fig.suptitle('栅格地图构建结果对比')
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/map_comparison.png', dpi=150)
print(f"地图对比图: {OUTPUT_DIR}/map_comparison.png")


# =========================
# 动画：贝叶斯信念地图
# =========================
belief_fig = plt.figure()
belief_ax = belief_fig.add_subplot(111)
belief_ax.set_xlim(0, N)
belief_ax.set_ylim(0, M)


def belief_update(i):
    belief_ax.clear()
    belief_ax.set_xlim(0, N)
    belief_ax.set_ylim(0, M)
    belief_ax.imshow(bayes_maps[i], cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    belief_ax.plot(x[1, max(0, i-10):i+1], x[0, max(0, i-10):i+1], 'bx-')
    belief_ax.set_title(f'贝叶斯占用栅格 (t={i})')


belief_anim = anim.FuncAnimation(belief_fig, belief_update, frames=len(time_steps), repeat=False)
print("正在保存贝叶斯动画...")
belief_anim.save(f'{OUTPUT_DIR}/bayes_belief_animation.mp4', writer='ffmpeg', fps=30)
print(f"贝叶斯动画: {OUTPUT_DIR}/bayes_belief_animation.mp4")

plt.show()
