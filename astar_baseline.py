#!/usr/bin/env python
# coding: utf-8

"""
基于占用栅格地图的 A* 路径规划
作者：GridMapping 项目
功能：在二值占用地图上实现A*路径规划并可视化
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import heapq
import os
from typing import List, Tuple, Optional, Set

# 输出目录设置
OUTPUT_DIR = 'results/a_star'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# A* 路径规划算法
# =========================
class AStar:
    """A*路径规划算法实现"""

    def __init__(self, occupancy_map: np.ndarray):
        """
        初始化A*规划器
        occupancy_map: 二值占用地图，1表示障碍，0表示自由
        """
        self.map = occupancy_map
        self.M, self.N = occupancy_map.shape
        # 8方向移动：上、下、左、右、四个对角
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 四个对角
        ]
        # 移动代价：直线1，对角√2
        self.move_costs = [1, 1, 1, 1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]

        # 记录搜索过程用于可视化
        self.visited_nodes = []
        self.open_nodes = []

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        启发函数：使用欧几里得距离
        """
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """检查位置是否有效（在地图范围内且不是障碍）"""
        x, y = pos
        if x < 0 or x >= self.M or y < 0 or y >= self.N:
            return False
        return self.map[x, y] == 0

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int],
             record_search: bool = False) -> Optional[List[Tuple[int, int]]]:
        """
        A*路径规划主函数
        start: 起点坐标 (x, y)
        goal: 终点坐标 (x, y)
        record_search: 是否记录搜索过程用于可视化
        返回: 路径点列表，如果无法到达则返回None
        """
        if not self.is_valid(start):
            print(f"起点 {start} 无效或位于障碍上")
            return None
        if not self.is_valid(goal):
            print(f"终点 {goal} 无效或位于障碍上")
            return None

        # 清空搜索记录
        self.visited_nodes = []
        self.open_nodes = []

        # 优先队列: (f_score, counter, position)
        # counter用于打破平局
        counter = 0
        open_set = [(self.heuristic(start, goal), counter, start)]

        # 记录每个节点的前驱
        came_from = {}

        # g_score: 从起点到当前节点的实际代价
        g_score = {start: 0}

        # f_score: g_score + heuristic
        f_score = {start: self.heuristic(start, goal)}

        # 已访问的节点
        closed_set: Set[Tuple[int, int]] = set()

        if record_search:
            self.open_nodes.append(start)

        while open_set:
            # 取出f值最小的节点
            _, _, current = heapq.heappop(open_set)

            # 到达目标
            if current == goal:
                return self.reconstruct_path(came_from, current)

            # 已访问过则跳过
            if current in closed_set:
                continue
            closed_set.add(current)

            if record_search:
                self.visited_nodes.append(current)

            # 遍历所有邻居
            for direction, move_cost in zip(self.directions, self.move_costs):
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # 检查邻居是否有效
                if not self.is_valid(neighbor):
                    continue

                # 已访问过则跳过
                if neighbor in closed_set:
                    continue

                # 对角移动时检查是否会被角落阻挡
                if abs(direction[0]) == 1 and abs(direction[1]) == 1:
                    if self.map[current[0] + direction[0], current[1]] == 1 or \
                       self.map[current[0], current[1] + direction[1]] == 1:
                        continue

                # 计算新的g值
                tentative_g = g_score[current] + move_cost

                # 如果找到更优路径
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))

                    if record_search:
                        self.open_nodes.append(neighbor)

        print("无法找到从起点到终点的路径")
        return None

    def reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def smooth_path(path: List[Tuple[int, int]], occupancy_map: np.ndarray) -> List[Tuple[int, int]]:
    """
    路径平滑：尝试通过视线检测来减少路径点
    """
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 0

    while i < len(path) - 1:
        # 尝试跳过中间点，直接连接到更远的点
        j = len(path) - 1
        while j > i + 1:
            if has_line_of_sight(path[i], path[j], occupancy_map):
                break
            j -= 1
        smoothed.append(path[j])
        i = j

    return smoothed


def has_line_of_sight(p1: Tuple[int, int], p2: Tuple[int, int],
                      occupancy_map: np.ndarray) -> bool:
    """
    检查两点之间是否有视线（不经过障碍）
    使用Bresenham直线算法
    """
    x0, y0 = p1
    x1, y1 = p2

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    M, N = occupancy_map.shape

    while True:
        # 检查边界
        if x0 < 0 or x0 >= M or y0 < 0 or y0 >= N:
            return False
        if occupancy_map[x0, y0] == 1:
            return False
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return True


def inflate_obstacles(occupancy_map: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    障碍物膨胀：为安全起见，将障碍物周围一定范围内的自由空间也标记为障碍
    """
    inflated = occupancy_map.copy()
    M, N = occupancy_map.shape

    for r in range(1, radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    shifted = np.roll(np.roll(occupancy_map, dx, axis=0), dy, axis=1)
                    inflated = np.maximum(inflated, shifted)

    return inflated


# =========================
# 路径可视化函数
# =========================
def visualize_inflation(original_map: np.ndarray, inflated_map: np.ndarray,
                        save_path: str = 'obstacle_inflation_visualization.png'):
    """
    可视化障碍物膨胀效果对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 左图：原始占用地图
    ax1 = axes[0]
    ax1.imshow(1 - original_map, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax1.set_title(f'原始占用地图\n障碍栅格数: {np.sum(original_map)}', fontsize=12)
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')
    ax1.grid(True, alpha=0.3)

    # 中图：膨胀后占用地图
    ax2 = axes[1]
    ax2.imshow(1 - inflated_map, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax2.set_title(f'膨胀后占用地图\n障碍栅格数: {np.sum(inflated_map)}', fontsize=12)
    ax2.set_xlabel('Y')
    ax2.set_ylabel('X')
    ax2.grid(True, alpha=0.3)

    # 右图：膨胀区域对比（红色显示新增的膨胀区域）
    ax3 = axes[2]
    # 创建对比地图：原始障碍为黑色，膨胀区域为红色，自由空间为白色
    diff_map = np.zeros((original_map.shape[0], original_map.shape[1], 3))
    # 自由空间 - 白色
    diff_map[inflated_map == 0] = [1, 1, 1]
    # 原始障碍 - 黑色
    diff_map[original_map == 1] = [0, 0, 0]
    # 膨胀新增区域 - 红色
    inflation_area = (inflated_map == 1) & (original_map == 0)
    diff_map[inflation_area] = [1, 0, 0]

    ax3.imshow(diff_map, origin='lower')
    ax3.set_title(f'膨胀区域对比\n红色: 新增膨胀区域 ({np.sum(inflation_area)} 栅格)', fontsize=12)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('X')
    ax3.grid(True, alpha=0.3)

    # 添加膨胀半径信息
    inflation_ratio = np.sum(inflated_map) / max(np.sum(original_map), 1)
    fig.suptitle(f'障碍物膨胀可视化 | 膨胀比例: {inflation_ratio:.2f}x', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"膨胀可视化已保存至: {save_path}")
    plt.show()

    return np.sum(inflation_area)


def visualize_path_planning(occupancy_map: np.ndarray, path: List[Tuple[int, int]],
                           start: Tuple[int, int], goal: Tuple[int, int],
                           robot_trajectory: np.ndarray = None,
                           save_path: str = 'astar_path_visualization.png'):
    """
    可视化路径规划结果
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：占用地图与A*路径
    ax1 = axes[0]
    ax1.imshow(1 - occupancy_map, cmap='gray', origin='lower', vmin=0, vmax=1)

    # 绘制路径
    if path:
        path_array = np.array(path)
        ax1.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2, label='A* 路径')
        ax1.plot(path_array[:, 1], path_array[:, 0], 'ro', markersize=3)

    # 标记起点和终点
    ax1.plot(start[1], start[0], 'go', markersize=12, label=f'起点 {start}')
    ax1.plot(goal[1], goal[0], 'b*', markersize=15, label=f'终点 {goal}')

    ax1.set_title('A* 路径规划结果', fontsize=14)
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, occupancy_map.shape[1])
    ax1.set_ylim(0, occupancy_map.shape[0])
    ax1.grid(True, alpha=0.3)

    # 右图：占用地图与机器人轨迹
    ax2 = axes[1]
    ax2.imshow(1 - occupancy_map, cmap='gray', origin='lower', vmin=0, vmax=1)

    if robot_trajectory is not None:
        ax2.plot(robot_trajectory[1, :], robot_trajectory[0, :], 'c-',
                linewidth=1.5, alpha=0.7, label='机器人扫描轨迹')

    # 绘制路径
    if path:
        path_array = np.array(path)
        ax2.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2, label='A* 路径')

    ax2.plot(start[1], start[0], 'go', markersize=12, label=f'起点 {start}')
    ax2.plot(goal[1], goal[0], 'b*', markersize=15, label=f'终点 {goal}')

    ax2.set_title('占用地图、扫描轨迹与规划路径', fontsize=14)
    ax2.set_xlabel('Y')
    ax2.set_ylabel('X')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, occupancy_map.shape[1])
    ax2.set_ylim(0, occupancy_map.shape[0])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"路径可视化已保存至: {save_path}")
    plt.show()


def visualize_search_process(occupancy_map: np.ndarray, astar: AStar,
                             path: List[Tuple[int, int]],
                             start: Tuple[int, int], goal: Tuple[int, int],
                             save_path: str = 'astar_search_process.png'):
    """
    可视化A*搜索过程
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 显示地图
    ax.imshow(1 - occupancy_map, cmap='gray', origin='lower', vmin=0, vmax=1)

    # 显示已访问节点
    if astar.visited_nodes:
        visited_array = np.array(astar.visited_nodes)
        ax.scatter(visited_array[:, 1], visited_array[:, 0],
                  c='lightblue', s=5, alpha=0.5, label=f'已搜索节点 ({len(astar.visited_nodes)})')

    # 显示路径
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2.5, label='最终路径')
        ax.plot(path_array[:, 1], path_array[:, 0], 'ro', markersize=4)

    # 标记起点终点
    ax.plot(start[1], start[0], 'go', markersize=14, label=f'起点 {start}')
    ax.plot(goal[1], goal[0], 'b*', markersize=18, label=f'终点 {goal}')

    ax.set_title('A* 搜索过程可视化', fontsize=14)
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.legend(loc='upper right')
    ax.set_xlim(0, occupancy_map.shape[1])
    ax.set_ylim(0, occupancy_map.shape[0])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"搜索过程可视化已保存至: {save_path}")
    plt.show()


def create_path_animation(occupancy_map: np.ndarray, path: List[Tuple[int, int]],
                         start: Tuple[int, int], goal: Tuple[int, int],
                         filename: str = 'astar_path_animation.mp4'):
    """
    创建路径动画，展示路径行走过程
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        ax.clear()
        ax.imshow(1 - occupancy_map, cmap='gray', origin='lower', vmin=0, vmax=1)

        # 逐步显示路径
        if frame < len(path):
            visible_path = path[:frame + 1]
            path_array = np.array(visible_path)
            ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
            ax.plot(path_array[:, 1], path_array[:, 0], 'ro', markersize=4)

            # 当前位置
            ax.plot(path_array[-1, 1], path_array[-1, 0], 'yo', markersize=10)

        # 标记起点终点
        ax.plot(start[1], start[0], 'go', markersize=12, label='起点')
        ax.plot(goal[1], goal[0], 'b*', markersize=15, label='终点')

        ax.set_xlim(0, occupancy_map.shape[1])
        ax.set_ylim(0, occupancy_map.shape[0])
        ax.set_title(f'A* 路径规划动画 (步数: {frame + 1}/{len(path)})', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    anim_obj = anim.FuncAnimation(fig, update, frames=len(path), repeat=False)
    anim_obj.save(filename, writer='ffmpeg', fps=15)
    print(f"路径动画已保存至: {filename}")
    plt.close()


def create_search_animation(occupancy_map: np.ndarray, astar: AStar,
                            path: List[Tuple[int, int]],
                            start: Tuple[int, int], goal: Tuple[int, int],
                            filename: str = 'astar_search_animation.mp4'):
    """
    创建A*搜索过程动画
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    visited_nodes = astar.visited_nodes
    total_frames = len(visited_nodes) + len(path)

    def update(frame):
        ax.clear()
        ax.imshow(1 - occupancy_map, cmap='gray', origin='lower', vmin=0, vmax=1)

        if frame < len(visited_nodes):
            # 搜索阶段
            current_visited = visited_nodes[:frame + 1]
            visited_array = np.array(current_visited)
            ax.scatter(visited_array[:, 1], visited_array[:, 0],
                      c='lightblue', s=8, alpha=0.7)
            ax.set_title(f'A* 搜索过程 (已搜索: {frame + 1}/{len(visited_nodes)})', fontsize=14)
        else:
            # 显示路径阶段
            path_frame = frame - len(visited_nodes)
            ax.scatter(np.array(visited_nodes)[:, 1], np.array(visited_nodes)[:, 0],
                      c='lightblue', s=5, alpha=0.3)

            if path_frame < len(path):
                visible_path = path[:path_frame + 1]
                path_array = np.array(visible_path)
                ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2.5)
                ax.plot(path_array[:, 1], path_array[:, 0], 'ro', markersize=4)

            ax.set_title(f'A* 重建路径 (步数: {path_frame + 1}/{len(path)})', fontsize=14)

        # 标记起点终点
        ax.plot(start[1], start[0], 'go', markersize=12, label='起点')
        ax.plot(goal[1], goal[0], 'b*', markersize=15, label='终点')

        ax.set_xlim(0, occupancy_map.shape[1])
        ax.set_ylim(0, occupancy_map.shape[0])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    anim_obj = anim.FuncAnimation(fig, update, frames=total_frames, repeat=False)
    anim_obj.save(filename, writer='ffmpeg', fps=30)
    print(f"搜索动画已保存至: {filename}")
    plt.close()


# =========================
# 主程序：加载占用地图并执行路径规划
# =========================
def load_occupancy_map_from_grid_mapping():
    """
    从grid_maping.py的输出加载或生成占用地图
    这里我们直接复制grid_maping.py的核心代码来生成占用地图
    """
    # 激光射线投射函数
    def get_ranges(true_map, X, meas_phi, rmax):
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

    def accumulate_endpoints(count_map, X, meas_phi, meas_r, rmax):
        M, N = count_map.shape
        x, y, theta = X
        endpoints = []

        for i in range(len(meas_phi)):
            r = meas_r[i]
            if r >= rmax:
                continue

            xi = int(round(x + r * math.cos(theta + meas_phi[i])))
            yi = int(round(y + r * math.sin(theta + meas_phi[i])))

            if 0 <= xi < M and 0 <= yi < N:
                count_map[xi, yi] += 1
                endpoints.append((xi, yi))

        return count_map, endpoints

    def build_binary_map(count_map, threshold=1):
        return (count_map >= threshold).astype(np.uint8)

    # 仿真初始化
    T_MAX = 150
    time_steps = np.arange(T_MAX)
    x_0 = [30, 30, 0]

    u = np.array([
        [3, 0, -3, 0],
        [0, 3, 0, -3]
    ])
    u_i = 1

    w = np.multiply(0.3, np.ones(len(time_steps)))

    M, N = 50, 60
    true_map = np.zeros((M, N), dtype=np.uint8)
    true_map[0:10, 0:10] = 1
    true_map[30:35, 40:45] = 1
    true_map[3:6, 40:60] = 1
    true_map[20:30, 25:29] = 1
    true_map[40:50, 5:25] = 1

    count_map = np.zeros((M, N), dtype=np.int32)
    threshold = 1

    meas_phi = np.arange(-0.4, 0.4, 0.05)
    rmax = 30

    x = np.zeros((3, len(time_steps)))
    x[:, 0] = x_0

    # 初始扫描
    meas_r = get_ranges(true_map, x[:, 0], meas_phi, rmax)
    count_map, _ = accumulate_endpoints(count_map, x[:, 0], meas_phi, meas_r, rmax)
    occupancy_map = build_binary_map(count_map, threshold)

    # 主循环
    for t in range(1, len(time_steps)):
        move = np.add(x[0:2, t - 1], u[:, u_i])

        if ((move[0] >= M - 1) or (move[1] >= N - 1) or
            (move[0] <= 0) or (move[1] <= 0) or
            true_map[int(round(move[0])), int(round(move[1]))] == 1):
            x[:, t] = x[:, t - 1]
            u_i = (u_i + 1) % 4
        else:
            x[0:2, t] = move

        x[2, t] = (x[2, t - 1] + w[t]) % (2 * math.pi)

        meas_r = get_ranges(true_map, x[:, t], meas_phi, rmax)
        count_map, _ = accumulate_endpoints(count_map, x[:, t], meas_phi, meas_r, rmax)
        occupancy_map = build_binary_map(count_map, threshold)

    return occupancy_map, true_map, x, M, N


def find_nearest_free(pos: Tuple[int, int], occ_map: np.ndarray) -> Tuple[int, int]:
    """寻找最近的自由空间点"""
    if occ_map[pos[0], pos[1]] == 0:
        return pos
    for r in range(1, 15):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < occ_map.shape[0] and 0 <= ny < occ_map.shape[1]:
                    if occ_map[nx, ny] == 0:
                        return (nx, ny)
    return pos


if __name__ == "__main__":
    print("=" * 60)
    print("  A* 路径规划 - 基于占用栅格地图")
    print("=" * 60)

    # 1. 加载/生成占用地图
    print("\n[步骤1] 生成占用栅格地图...")
    occupancy_map, true_map, robot_trajectory, M, N = load_occupancy_map_from_grid_mapping()
    print(f"  地图尺寸: {M} x {N}")
    print(f"  检测到的障碍栅格数: {np.sum(occupancy_map)}")

    # 2. 可选：障碍物膨胀
    print("\n[步骤2] 障碍物膨胀处理（安全距离）...")
    inflated_map = inflate_obstacles(occupancy_map, radius=1)
    print(f"  膨胀后障碍栅格数: {np.sum(inflated_map)}")

    # 膨胀可视化
    inflation_area = visualize_inflation(
        occupancy_map, inflated_map,
        save_path=f'{OUTPUT_DIR}/obstacle_inflation_visualization.png'
    )
    print(f"  新增膨胀区域: {inflation_area} 栅格")

    # 3. 定义起点和终点
    print("\n[步骤3] 设置起点和终点...")
    start_pos = (int(robot_trajectory[0, -1]), int(robot_trajectory[1, -1]))  # 扫描结束位置
    goal_pos = (15, 50)  # 目标位置

    # 确保在自由空间
    start_pos = find_nearest_free(start_pos, inflated_map)
    goal_pos = find_nearest_free(goal_pos, inflated_map)
    print(f"  起点: {start_pos}")
    print(f"  终点: {goal_pos}")

    # 4. 执行A*路径规划
    print("\n[步骤4] 执行 A* 路径规划...")
    astar = AStar(inflated_map)
    path = astar.plan(start_pos, goal_pos, record_search=True)

    if path:
        print(f"  ✓ 路径规划成功！")
        print(f"  路径长度: {len(path)} 步")

        # 计算路径总长度
        total_length = sum(
            math.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
            for i in range(len(path) - 1)
        )
        print(f"  路径总距离: {total_length:.2f}")

        # 5. 路径平滑
        print("\n[步骤5] 路径平滑处理...")
        smoothed_path = smooth_path(path, inflated_map)
        print(f"  平滑后路径长度: {len(smoothed_path)} 步")

        # 6. 可视化
        print("\n[步骤6] 生成可视化结果...")

        # 路径规划结果可视化
        visualize_path_planning(
            inflated_map, path, start_pos, goal_pos,
            robot_trajectory=robot_trajectory,
            save_path=f'{OUTPUT_DIR}/astar_path_visualization.png'
        )

        # 搜索过程可视化
        visualize_search_process(
            inflated_map, astar, path, start_pos, goal_pos,
            save_path=f'{OUTPUT_DIR}/astar_search_process.png'
        )

        # 路径行走动画
        create_path_animation(
            inflated_map, path, start_pos, goal_pos,
            filename=f'{OUTPUT_DIR}/astar_path_animation.mp4'
        )

        # 搜索过程动画
        create_search_animation(
            inflated_map, astar, path, start_pos, goal_pos,
            filename=f'{OUTPUT_DIR}/astar_search_animation.mp4'
        )

        # 7. 输出统计信息
        print("\n" + "=" * 60)
        print("  统计信息汇总")
        print("=" * 60)
        print(f"  地图尺寸: {M} x {N}")
        print(f"  真实障碍栅格数: {np.sum(true_map)}")
        print(f"  检测到的障碍栅格数: {np.sum(occupancy_map)}")
        print(f"  检测率: {np.sum(occupancy_map & true_map) / max(np.sum(true_map), 1) * 100:.2f}%")
        print(f"  A* 搜索节点数: {len(astar.visited_nodes)}")
        print(f"  原始路径长度: {len(path)} 步")
        print(f"  平滑路径长度: {len(smoothed_path)} 步")
        print(f"  路径总距离: {total_length:.2f}")

        print("\n" + "=" * 60)
        print("  生成的文件")
        print("=" * 60)
        print(f"  - {OUTPUT_DIR}/obstacle_inflation_visualization.png  (障碍物膨胀对比)")
        print(f"  - {OUTPUT_DIR}/astar_path_visualization.png          (路径规划结果)")
        print(f"  - {OUTPUT_DIR}/astar_search_process.png              (搜索过程)")
        print(f"  - {OUTPUT_DIR}/astar_path_animation.mp4              (路径行走动画)")
        print(f"  - {OUTPUT_DIR}/astar_search_animation.mp4            (搜索过程动画)")

    else:
        print("  ✗ 路径规划失败！")
        print("  可能原因：起点或终点被障碍物包围")

        # 仍然保存占用地图
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(1 - inflated_map, cmap='gray', origin='lower')
        ax.plot(robot_trajectory[1, :], robot_trajectory[0, :], 'c-', label='扫描轨迹')
        ax.plot(start_pos[1], start_pos[0], 'go', markersize=12, label='起点')
        ax.plot(goal_pos[1], goal_pos[0], 'b*', markersize=15, label='终点')
        ax.set_title('占用地图（路径规划失败）')
        ax.legend()
        plt.savefig(f'{OUTPUT_DIR}/occupancy_map_no_path.png', dpi=150)
        plt.show()
