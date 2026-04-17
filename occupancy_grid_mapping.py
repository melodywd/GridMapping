#!/usr/bin/env python
# coding: utf-8

# # 模块2评估
# 欢迎参加模块2的评估：规划中的地图构建。在本次评估中，你将使用移动车辆在未知环境中的激光雷达扫描测量数据来生成占据栅格地图。你将使用课程中开发的逆扫描测量模型将这些测量数据映射为占据概率，然后对占据栅格信念地图执行迭代对数几率更新。当车辆收集到足够的数据后，你的占据栅格应该收敛到真实地图。
#
# 在本次评估中，你将：
# * 使用激光雷达扫描功能收集移动车辆周围的距离测量数据。
# * 使用逆扫描模型从距离测量中提取占据信息。
# * 基于传入的测量数据对占据栅格执行对数几率更新。
# * 从这些对数几率更新中迭代构建概率占据栅格。
#
# 对于大多数练习，我们提供了建议的大纲。如果你认为有更好、更高效的方法来解决问题，我们鼓励你偏离大纲。

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 在本笔记本中，你将基于多次模拟激光雷达扫描生成一个占据栅格。逆扫描模型将通过 `inverse_scanner()` 函数提供给你。该函数根据视频讲座中讨论的激光雷达扫描模型返回一个测量占据概率值矩阵。`get_ranges()` 函数实际上返回给定车辆位置和扫描方位角的扫描距离值。这两个函数在下面给出。请确保你理解它们的作用，因为你稍后需要在笔记本中使用它们。


# 计算激光扫描仪的逆测量模型。
# 该模型识别三个区域。第一个区域是在扫描弧之外，没有可用信息。
# 第二个区域是在扫描弧内的距离测量末端，物体可能存在。
# 第三个区域是在扫描弧内但距离小于距离测量的区域，物体不太可能存在。
# 点云到栅格的投影映射
def inverse_scanner(num_rows, num_cols, x, y, theta, meas_phi, meas_r, rmax, alpha, beta):
    m = np.zeros((M, N))
    for i in range(num_rows):
        for j in range(num_cols):
            # 计算相对于输入状态 (x, y, theta) 的距离和方位角。
            r = math.sqrt((i - x)**2 + (j - y)**2)
            phi = (math.atan2(j - y, i - x) - theta + math.pi) % (2 * math.pi) - math.pi

            # 找到与相对方位角相关联的距离测量值。
            k = np.argmin(np.abs(np.subtract(phi, meas_phi)))

            # 如果距离大于最大传感器范围，或位于距离测量之后，
            # 或超出传感器的视野范围，则没有新信息可用。
            if (r > min(rmax, meas_r[k] + alpha / 2.0)) or (abs(phi - meas_phi[k]) > beta / 2.0):
                m[i, j] = 0.5

            # 如果距离测量值位于该单元格内，则该处可能有物体。
            elif (meas_r[k] < rmax) and (abs(r - meas_r[k]) < alpha / 2.0):
                m[i, j] = 0.7

            # 如果该单元格位于距离测量之前，则该处可能为空。
            elif r < meas_r[k]:
                m[i, j] = 0.3

    return m



# 根据地图、车辆位置和传感器参数生成激光扫描仪的距离测量。
# 射线投射————使用光线追踪算法。
def get_ranges(true_map, X, meas_phi, rmax):
    (M, N) = np.shape(true_map)
    x = X[0]
    y = X[1]
    theta = X[2]
    meas_r = rmax * np.ones(meas_phi.shape)

    # 对每个测量方位角进行迭代。
    for i in range(len(meas_phi)):
        # 对每个单位步长进行迭代，直到并包括 rmax。
        for r in range(1, rmax+1):
            # 确定单元格的坐标。
            xi = int(round(x + r * math.cos(theta + meas_phi[i])))
            yi = int(round(y + r * math.sin(theta + meas_phi[i])))

            # 如果不在地图内，在该处设置测量值并停止继续追踪。
            if (xi <= 0 or xi >= M-1 or yi <= 0 or yi >= N-1):
                meas_r[i] = r
                break
            # 如果在地图内但遇到障碍物，设置测量距离并停止光线追踪。
            elif true_map[int(round(xi)), int(round(yi))] == 1:
                meas_r[i] = r
                break

    return meas_r


# 在下面的代码块中，我们初始化模拟所需的变量。这包括初始状态以及汽车的控制动作集合。我们还设置激光雷达扫描的旋转速率。真实地图中的障碍物用 1 表示，自由空间用 0 表示。信念地图 `m` 中的每个单元格初始化为 0.5 作为我们的占据先验概率，然后根据该信念地图计算对数几率占据栅格 `L`。


# 模拟时间初始化。
T_MAX = 150
time_steps = np.arange(T_MAX)

# 初始化机器人的位置。
x_0 = [30, 30, 0]

# 机器人运动序列。
u = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
u_i = 1

# 机器人传感器旋转命令
w = np.multiply(0.3, np.ones(len(time_steps)))

# 真实地图（注意，地图的列对应 y 轴，行对应 x 轴，
# 因此机器人位置 x = x(1) 和 y = x(2) 在绘制时交换以匹配）
M = 50
N = 60
true_map = np.zeros((M, N))
true_map[0:10, 0:10] = 1
true_map[30:35, 40:45] = 1
true_map[3:6,40:60] = 1
true_map[20:30,25:29] = 1
true_map[40:50,5:25] = 1

# 初始化信念地图。
# 我们假设均匀先验。
m = np.multiply(0.5, np.ones((M, N)))

# 初始化对数几率比。
L0 = np.log(np.divide(m, np.subtract(1, m)))
L = L0

# 传感器模型的参数。
meas_phi = np.arange(-0.4, 0.4, 0.05)
rmax = 30  # 最大光束范围。
alpha = 1  # 障碍物宽度（测量值周围填充的距离）。
beta = 0.05  # 光束的角度宽度。

# 初始化模拟的状态向量。
x = np.zeros((3, len(time_steps)))
x[:, 0] = x_0


# 这里是你需要输入代码的地方。
# 你的任务是完成主模拟循环。
# 在机器人运动的每一步之后，你需要从激光雷达扫描中收集距离数据，然后应用逆扫描模型将这些数据映射到测量的占据信念地图。
# 然后，你将对对数几率占据栅格执行对数几率更新，并相应地更新我们的信念地图。当汽车穿越环境时，占据栅格信念地图应该越来越接近真实地图。
# 在循环结束后的代码块中，代码将输出一些值，用于对你的作业进行评分。请确保将这些值复制下来，当你的可视化看起来正确时，将其保存在 .txt 文件中。祝你好运！


# 初始化图形。
map_fig = plt.figure()
map_ax = map_fig.add_subplot(111)
map_ax.set_xlim(0, N)
map_ax.set_ylim(0, M)

invmod_fig = plt.figure()
invmod_ax = invmod_fig.add_subplot(111)
invmod_ax.set_xlim(0, N)
invmod_ax.set_ylim(0, M)

belief_fig = plt.figure()
belief_ax = belief_fig.add_subplot(111)
belief_ax.set_xlim(0, N)
belief_ax.set_ylim(0, M)

meas_rs = []
meas_r = get_ranges(true_map, x[:, 0], meas_phi, rmax)
meas_rs.append(meas_r)
invmods = []
invmod = inverse_scanner(M, N, x[0, 0], x[1, 0], x[2, 0], meas_phi, meas_r, \
                         rmax, alpha, beta)
invmods.append(invmod)
ms = []
ms.append(m)

# 主模拟循环。
for t in range(1, len(time_steps)):
    # 执行机器人运动。
    move = np.add(x[0:2, t-1], u[:, u_i])
    # 如果碰到地图边界，或会发生碰撞，则保持静止。
    if (move[0] >= M - 1) or (move[1] >= N - 1) or (move[0] <= 0) or (move[1] <= 0) \
        or true_map[int(round(move[0])), int(round(move[1]))] == 1:
        x[:, t] = x[:, t-1]
        u_i = (u_i + 1) % 4
    else:
        x[0:2, t] = move
    x[2, t] = (x[2, t-1] + w[t]) % (2 * math.pi)

    # 收集测量距离数据，我们将使用逆测量模型将其转换为占据概率。
    meas_r = get_ranges(true_map, x[:, t], meas_phi, rmax)
    meas_rs.append(meas_r)

    # 给定我们的距离测量和机器人位置，应用逆扫描模型获取测量的占据概率。
    invmod = inverse_scanner(M, N, x[0, t], x[1, t], x[2, t], meas_phi, meas_r, rmax, alpha, beta)
    invmods.append(invmod)

    # 根据逆模型得到的测量占据概率，计算并更新占据栅格的对数几率。
    L = np.log(np.divide(invmod, np.subtract(1, invmod))) + L - L0

    # 从对数几率计算概率栅格。
    p = np.exp(L)
    m = p / (1+p)
    ms.append(m)



# 用于评分的输出。请勿修改此代码！
m_f = ms[-1]

print("{}".format(m_f[40, 10]))
print("{}".format(m_f[30, 40]))
print("{}".format(m_f[35, 40]))
print("{}".format(m_f[0, 50]))
print("{}".format(m_f[10, 5]))
print("{}".format(m_f[20, 15]))
print("{}".format(m_f[25, 50]))


# 现在你已经完成了主模拟循环的编写，你可以在下面可视化机器人在真实地图中的运动、测量的信念地图和占据栅格信念地图。它们分别显示在第1、第2和第3个视频中。如果你的第3个视频收敛到第1个视频中显示的真实地图，恭喜你！你已经完成了作业。请将上方框中的输出作为 .txt 文件提交给作业评分系统。


def map_update(i):
    map_ax.clear()
    map_ax.set_xlim(0, N)
    map_ax.set_ylim(0, M)
    map_ax.imshow(np.subtract(1, true_map), cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    x_plot = x[1, :i+1]
    y_plot = x[0, :i+1]
    map_ax.plot(x_plot, y_plot, "bx-")

def invmod_update(i):
    invmod_ax.clear()
    invmod_ax.set_xlim(0, N)
    invmod_ax.set_ylim(0, M)
    invmod_ax.imshow(invmods[i], cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    for j in range(len(meas_rs[i])):
        invmod_ax.plot(x[1, i] + meas_rs[i][j] * math.sin(meas_phi[j] + x[2, i]), x[0, i] + meas_rs[i][j] * math.cos(meas_phi[j] + x[2, i]), "ko")
    invmod_ax.plot(x[1, i], x[0, i], 'bx')

def belief_update(i):
    belief_ax.clear()
    belief_ax.set_xlim(0, N)
    belief_ax.set_ylim(0, M)
    belief_ax.imshow(ms[i], cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    belief_ax.plot(x[1, max(0, i-10):i], x[0, max(0, i-10):i], 'bx-')

map_anim = anim.FuncAnimation(map_fig, map_update, frames=len(x[0, :]), repeat=False)
invmod_anim = anim.FuncAnimation(invmod_fig, invmod_update, frames=len(x[0, :]), repeat=False)
belief_anim = anim.FuncAnimation(belief_fig, belief_update, frames=len(x[0, :]), repeat=False)


# 保存动画为文件
print("正在保存动画...")
map_anim.save('map_animation.mp4', writer='ffmpeg', fps=30)
invmod_anim.save('invmod_animation.mp4', writer='ffmpeg', fps=30)
belief_anim.save('belief_animation.mp4', writer='ffmpeg', fps=30)
print("动画已保存为 map_animation.mp4, invmod_animation.mp4, belief_animation.mp4")

# 如果没有安装 ffmpeg，可以使用 GIF 格式
# map_anim.save('map_animation.gif', writer='pillow', fps=10)
# invmod_anim.save('invmod_animation.gif', writer='pillow', fps=10)
# belief_anim.save('belief_animation.gif', writer='pillow', fps=10)

