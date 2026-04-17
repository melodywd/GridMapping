#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import os

# 输出目录设置
OUTPUT_DIR = 'results/map'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# =========================
# 1. 激光射线投射：生成每个方向的测距值
# =========================
def get_ranges(true_map, X, meas_phi, rmax):
    """
    根据真实地图和机器人状态，模拟激光雷达扫描距离。
    true_map: 真实地图，障碍为1，自由为0
    X: [x, y, theta]
    meas_phi: 相对扫描角数组
    rmax: 最大量程
    """
    M, N = np.shape(true_map)
    x, y, theta = X
    meas_r = rmax * np.ones(meas_phi.shape)

    for i in range(len(meas_phi)):
        for r in range(1, rmax + 1):
            xi = int(round(x + r * math.cos(theta + meas_phi[i])))
            yi = int(round(y + r * math.sin(theta + meas_phi[i])))

            # 超出地图边界
            if xi <= 0 or xi >= M - 1 or yi <= 0 or yi >= N - 1:
                meas_r[i] = r
                break

            # 击中障碍
            if true_map[xi, yi] == 1:
                meas_r[i] = r
                break

    return meas_r


# =========================
# 2. 无贝叶斯直接累加：只统计终点占用
# =========================
def accumulate_endpoints(count_map, X, meas_phi, meas_r, rmax):
    """
    将每条激光束终点投影到栅格地图中，并执行计数累加。
    这里只处理终点占用，不处理空闲空间。
    """
    M, N = count_map.shape
    x, y, theta = X

    endpoints = []

    for i in range(len(meas_phi)):
        r = meas_r[i]

        # 若射线量测达到最大量程且没有明确打到障碍，可选择跳过
        # 这样更符合“只累计真实障碍终点”的思路
        if r >= rmax:
            continue

        xi = int(round(x + r * math.cos(theta + meas_phi[i])))
        yi = int(round(y + r * math.sin(theta + meas_phi[i])))

        if 0 <= xi < M and 0 <= yi < N:
            count_map[xi, yi] += 1
            endpoints.append((xi, yi))

    return count_map, endpoints


# =========================
# 3. 二值化地图
# =========================
def build_binary_map(count_map, threshold=1):
    """
    根据阈值将计数地图转成二值占用图
    threshold越大，噪声越少，但漏检可能越多
    """
    return (count_map >= threshold).astype(np.uint8)


# =========================
# 4. 仿真初始化
# =========================
T_MAX = 150
time_steps = np.arange(T_MAX)

# 机器人初始状态
x_0 = [30, 30, 0]

# 机器人运动序列
u = np.array([
    [3, 0, -3, 0],
    [0, 3, 0, -3]
])
u_i = 1

# 机器人旋转角速度
w = np.multiply(0.3, np.ones(len(time_steps)))

# 真实地图
M = 50
N = 60
true_map = np.zeros((M, N), dtype=np.uint8)
# 设置障碍物
true_map[0:10, 0:10] = 1
true_map[30:35, 40:45] = 1
true_map[3:6, 40:60] = 1
true_map[20:30, 25:29] = 1
true_map[40:50, 5:25] = 1

# 无贝叶斯计数地图
count_map = np.zeros((M, N), dtype=np.int32)

# 阈值化后的二值占用图
threshold = 1
occupancy_map = np.zeros((M, N), dtype=np.uint8)

# 传感器参数
meas_phi = np.arange(-0.4, 0.4, 0.05)
rmax = 30

# 状态轨迹
x = np.zeros((3, len(time_steps)))
x[:, 0] = x_0

# 记录过程数据，用于可视化
meas_rs = []
endpoint_history = []
count_maps = []
occupancy_maps = []


# =========================
# 5. 初始帧扫描
# =========================
meas_r = get_ranges(true_map, x[:, 0], meas_phi, rmax)
meas_rs.append(meas_r)

count_map, endpoints = accumulate_endpoints(count_map, x[:, 0], meas_phi, meas_r, rmax)
endpoint_history.append(endpoints)

occupancy_map = build_binary_map(count_map, threshold)
count_maps.append(count_map.copy())
occupancy_maps.append(occupancy_map.copy())


# =========================
# 6. 主循环：运动 + 扫描 + 直接累加
# =========================
for t in range(1, len(time_steps)):
    # 机器人移动
    move = np.add(x[0:2, t - 1], u[:, u_i])

    # 若越界或碰撞，则停止并改变运动方向
    if ((move[0] >= M - 1) or (move[1] >= N - 1) or
        (move[0] <= 0) or (move[1] <= 0) or
        true_map[int(round(move[0])), int(round(move[1]))] == 1):
        x[:, t] = x[:, t - 1]
        u_i = (u_i + 1) % 4
    else:
        x[0:2, t] = move

    # 更新航向
    x[2, t] = (x[2, t - 1] + w[t]) % (2 * math.pi)

    # 激光扫描
    meas_r = get_ranges(true_map, x[:, t], meas_phi, rmax)
    meas_rs.append(meas_r)

    # 无贝叶斯直接累加
    count_map, endpoints = accumulate_endpoints(count_map, x[:, t], meas_phi, meas_r, rmax)
    endpoint_history.append(endpoints)

    # 二值化地图
    occupancy_map = build_binary_map(count_map, threshold)

    # 保存过程
    count_maps.append(count_map.copy())
    occupancy_maps.append(occupancy_map.copy())


# =========================
# 7. 输出一些检查值
# =========================
final_count = count_maps[-1]
final_occ = occupancy_maps[-1]

print("最终计数地图部分值：")
print(final_count[40, 10])
print(final_count[30, 40])
print(final_count[35, 40])
print(final_count[0, 50])
print(final_count[10, 5])
print(final_count[20, 15])
print(final_count[25, 50])

print("\n最终二值占用地图部分值：")
print(final_occ[40, 10])
print(final_occ[30, 40])
print(final_occ[35, 40])
print(final_occ[0, 50])
print(final_occ[10, 5])
print(final_occ[20, 15])
print(final_occ[25, 50])


# =========================
# 8. 可视化
# =========================
true_fig = plt.figure()
true_ax = true_fig.add_subplot(111)
true_ax.set_xlim(0, N)
true_ax.set_ylim(0, M)

count_fig = plt.figure()
count_ax = count_fig.add_subplot(111)
count_ax.set_xlim(0, N)
count_ax.set_ylim(0, M)

occ_fig = plt.figure()
occ_ax = occ_fig.add_subplot(111)
occ_ax.set_xlim(0, N)
occ_ax.set_ylim(0, M)


def true_map_update(i):
    true_ax.clear()
    true_ax.set_xlim(0, N)
    true_ax.set_ylim(0, M)
    true_ax.imshow(1 - true_map, cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    true_ax.plot(x[1, :i+1], x[0, :i+1], 'bx-')
    true_ax.set_title("真实地图与机器人轨迹")


def count_map_update(i):
    count_ax.clear()
    count_ax.set_xlim(0, N)
    count_ax.set_ylim(0, M)

    # 对数显示更容易观察低计数区域
    vis_map = np.log1p(count_maps[i])
    count_ax.imshow(vis_map, cmap='hot', origin='lower')
    count_ax.plot(x[1, :i+1], x[0, :i+1], 'bx-')
    count_ax.set_title("直接累加计数热图（log(count+1)）")


def occupancy_map_update(i):
    occ_ax.clear()
    occ_ax.set_xlim(0, N)
    occ_ax.set_ylim(0, M)
    occ_ax.imshow(1 - occupancy_maps[i], cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    occ_ax.plot(x[1, max(0, i-10):i+1], x[0, max(0, i-10):i+1], 'bx-')
    occ_ax.set_title(f"二值占用地图（threshold={threshold}）")


true_anim = anim.FuncAnimation(true_fig, true_map_update, frames=len(time_steps), repeat=False)
count_anim = anim.FuncAnimation(count_fig, count_map_update, frames=len(time_steps), repeat=False)
occ_anim = anim.FuncAnimation(occ_fig, occupancy_map_update, frames=len(time_steps), repeat=False)

print("正在保存动画...")
true_anim.save(f'{OUTPUT_DIR}/true_map_animation.mp4', writer='ffmpeg', fps=30)
count_anim.save(f'{OUTPUT_DIR}/count_map_animation.mp4', writer='ffmpeg', fps=30)
occ_anim.save(f'{OUTPUT_DIR}/occupancy_map_animation.mp4', writer='ffmpeg', fps=30)
print("动画已保存为：")
print(f"{OUTPUT_DIR}/true_map_animation.mp4")
print(f"{OUTPUT_DIR}/count_map_animation.mp4")
print(f"{OUTPUT_DIR}/occupancy_map_animation.mp4")

plt.show()