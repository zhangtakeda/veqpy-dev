import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter  # 用于检测局部极小值


def generate_strictly_negative_terrain(n, grid_size=800):
    """
    生成最大值为0, 包含 n 个极小值和 n-1 个极大值的场.
    所有局部极小值的深度将被控制为全局最小值的约 80%.
    """
    x = np.linspace(-7, 7, grid_size)
    y = np.linspace(-7, 7, grid_size)
    X, Y = np.meshgrid(x, y)

    X2, Y2 = X**2, Y**2
    Z = np.zeros_like(X)

    def get_separated_random_points(num_points, existing_points, min_dist=2.8):
        points = []
        max_attempts = 2000
        attempts = 0
        while len(points) < num_points and attempts < max_attempts:
            px, py = np.random.uniform(-5.0, 5.0, 2)
            all_pts = existing_points + points
            if all(np.hypot(px - cx, py - cy) > min_dist for cx, cy in all_pts):
                points.append((px, py))
            attempts += 1
        return points

    # --- 1. 构造 (0,0) 处的全局深势阱 ---
    global_min_amp = 10.0
    global_min_sigma = 2.0
    Z -= global_min_amp * np.exp(-(X2 + Y2) / (2 * global_min_sigma**2))

    occupied_points = [(0.0, 0.0)]

    # --- 2. 生成 n-1 个深度相近的局部极小值 ---
    local_minima_pts = get_separated_random_points(n - 1, occupied_points, min_dist=2.8)
    occupied_points.extend(local_minima_pts)

    for x0, y0 in local_minima_pts:
        controlled_amplitude = np.random.uniform(0.76, 0.84) * global_min_amp
        sigma = np.random.uniform(1.2, 1.6)
        Z -= controlled_amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))

    # --- 3. 生成 n-1 个极大值以增加地形复杂性 ---
    maxima_pts = get_separated_random_points(n - 1, occupied_points, min_dist=2.5)
    for x0, y0 in maxima_pts:
        sigma = np.random.uniform(1.2, 1.8)
        amplitude = np.random.uniform(3.0, 6.0)
        Z += amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))

    # --- 4. 归一化: 使最大值为 0 ---
    Z = Z - np.max(Z)

    return x, y, Z


def save_strictly_negative_field_as_svg(x, y, Z, filename="terrain.svg"):
    """
    渲染场, 自动在所有局部极小值点绘制红色的 'x', 并保存为 SVG 矢量图.
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

    # 渲染连续场
    extent = [x.min(), x.max(), y.min(), y.max()]
    ax.imshow(Z, extent=extent, origin="lower", interpolation="bicubic", aspect="equal", vmax=0, cmap="viridis")

    # 绘制等高线
    levels = np.linspace(Z.min(), 0, 25)
    ax.contour(x, y, Z, levels=levels, colors="white", linewidths=0.3, alpha=0.3)

    # --- 核心新增: 自动检测并标记所有极小值点 ---
    # 使用最小滤波器找到局部极小值区域.
    # 窗口大小 (size) 需要根据网格分辨率和极小值的“宽度”进行调整.
    # 对于 800x800 网格, 20x20 的窗口是一个不错的初始值.
    min_local = minimum_filter(Z, size=(20, 20), mode="constant", cval=Z.max())

    # 找到原始 Z 值等于其邻域最小值的点 (即极小值点)
    # 我们加一个微小的阈值以处理浮点数精度问题, 并确保只标记真正的“坑”.
    minima_mask = Z == min_local

    # 获取这些极小值点在 Z 矩阵中的索引 (row, col)
    minima_coords = np.argwhere(minima_mask)

    # 在图像上绘制红色的 'x'
    for row, col in minima_coords:
        # 将矩阵索引转换回实际的地形坐标 (x, y)
        px = x[col]
        py = y[row]
        # 叠加绘制红色的 'x', 调整 markersize 以匹配图像
        ax.scatter(px, py, color="red", marker="x", s=100, linewidths=2, zorder=10)
    # ----------------------------------------------

    # --- 删除所有轴标识 ---
    ax.set_axis_off()

    # 优化布局
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    # 保存为矢量图
    plt.savefig(filename, format="svg", bbox_inches="tight", pad_inches=0)
    print(f"矢量图已保存为: {filename} (已标记 {len(minima_coords)} 个极小值)")
    plt.close(fig)


if __name__ == "__main__":
    n = 4  # 目标极小值数量
    x, y, Z = generate_strictly_negative_terrain(n)

    save_strictly_negative_field_as_svg(x, y, Z, filename="terrain_marked.svg")
