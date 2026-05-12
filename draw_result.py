import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import matplotlib
import networkx as nx
import pickle
import math

# 设置matplotlib使用支持中文的字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("Note: Could not set Chinese fonts, continuing with default fonts.")




def plot_bidirectional_and_oneway_strip_lines(gdf, flow_field,
                                              forward_field,
                                              backward_field,
                                              alpha_range=(0.2, 0.8),
                                              color_forward='red',
                                              color_backward='blue',
                                              color_oneway='green',
                                              base_width=1,
                                              figsize=(12, 10)):
    """
    绘制双向流量条带和单向流量条带，按照实际几何形状绘制

    参数:
    ----------
    gdf : GeoDataFrame
        包含几何和流量数据的GeoDataFrame
    flow_field : str
        单向边流量字段名
    forward_field : str
        双向边正向流量字段名
    backward_field : str
        双向边反向流量字段名
    alpha_range : tuple
        透明度范围(最小值, 最大值)
    color_forward : str
        双向边正向流量颜色
    color_backward : str
        双向边反向流量颜色
    color_oneway : str
        单向边流量颜色
    base_width : float
        基础宽度系数，根据数据范围调整
    figsize : tuple
        图形大小

    返回:
    ----------
    fig, ax : matplotlib图形和坐标轴对象
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    # 1. 分离双向边和单向边
    mask_bidirectional = gdf[forward_field].notna() & gdf[backward_field].notna()
    data = gdf[mask_bidirectional].copy()

    mask_oneway = gdf[flow_field].notna() & (gdf[forward_field].isna() | gdf[backward_field].isna())
    data_oneway = gdf[mask_oneway].copy()

    # 2. 计算全局流量范围
    all_flows = []

    if len(data) > 0:
        all_flows.extend(data[forward_field].values)
        all_flows.extend(data[backward_field].values)

    if len(data_oneway) > 0:
        all_flows.extend(data_oneway[flow_field].values)

    if not all_flows:
        print("警告：没有找到流量数据")
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    max_flow = max(all_flows)
    min_flow = min(all_flows)

    print(f"全局流量范围: {min_flow:.2f} - {max_flow:.2f}")

    # 3. 计算宽度和透明度的函数
    def calc_width(flow):
        if max_flow == min_flow:
            return base_width
        norm = (flow - min_flow) / (max_flow - min_flow)
        return base_width * (0.3 + 0.7 * norm)

    def calc_alpha(flow):
        if max_flow == min_flow:
            return alpha_range[0]
        norm = (flow - min_flow) / (max_flow - min_flow)
        return alpha_range[0] + norm * (alpha_range[1] - alpha_range[0])

    # 4. 为双向边计算宽度和透明度
    if len(data) > 0:
        data['width_forward'] = data[forward_field].apply(calc_width)
        data['width_backward'] = data[backward_field].apply(calc_width)
        data['alpha_forward'] = data[forward_field].apply(calc_alpha)
        data['alpha_backward'] = data[backward_field].apply(calc_alpha)

    # 5. 为单向边计算宽度和透明度
    if len(data_oneway) > 0:
        data_oneway['width'] = data_oneway[flow_field].apply(calc_width)
        data_oneway['alpha'] = data_oneway[flow_field].apply(calc_alpha)

    # 6. 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 7. 存储条带
    forward_strips = []
    forward_colors = []
    backward_strips = []
    backward_colors = []
    oneway_strips = []
    oneway_colors = []

    # 8. 辅助函数：创建沿线的条带
    def create_strip_along_line(coords, width, offset_direction=0):
        """
        沿着线的坐标点创建条带多边形

        参数:
        ----------
        coords : list of tuples
            线的坐标序列 [(x1,y1), (x2,y2), ...]
        width : float
            条带宽度
        offset_direction : float
            偏移方向：0表示居中，1表示正向偏移，-1表示反向偏移

        返回:
        ----------
        polygon : list of points
            条带多边形的顶点
        """
        if len(coords) < 2:
            return None

        # 计算每个线段的方向和法向量
        upper_points = []  # 上边界
        lower_points = []  # 下边界

        for i in range(len(coords) - 1):
            start = np.array(coords[i])
            end = np.array(coords[i + 1])

            # 计算方向向量
            direction = end - start
            length = np.linalg.norm(direction)

            if length == 0:
                continue

            # 单位方向向量
            dir_unit = direction / length

            # 法向量（垂直方向）
            normal = np.array([-dir_unit[1], dir_unit[0]])

            # 计算偏移
            if offset_direction == 0:  # 居中（单向边）
                offset = width / 2
                upper_offset = normal * offset
                lower_offset = normal * -offset
            elif offset_direction == 1:  # 正向偏移（双向边的正向）
                upper_offset = normal * width
                lower_offset = np.array([0.0, 0.0])  # 零偏移（原始边位置）
            else:  # offset_direction == -1: 反向偏移（双向边的反向）
                upper_offset = np.array([0.0, 0.0])  # 零偏移（原始边位置）
                lower_offset = normal * -width

            # 计算上下边界的点
            if i == 0:  # 第一个点
                upper_points.append(start + upper_offset)
                lower_points.append(start + lower_offset)

            # 中间点和终点
            upper_points.append(end + upper_offset)
            lower_points.append(end + lower_offset)

        # 如果没有计算到点，返回None
        if not upper_points or not lower_points:
            return None

        # 创建多边形：上边界正向，下边界反向
        polygon = upper_points + list(reversed(lower_points))

        # 确保多边形闭合
        if len(polygon) > 0 and not np.array_equal(polygon[0], polygon[-1]):
            polygon.append(polygon[0])

        return polygon

    # 9. 绘制双向边的条带
    for idx, row in data.iterrows():
        if row.geometry.geom_type != 'LineString':
            continue

        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue

        # 获取宽度
        w_forward = row['width_forward']
        w_backward = row['width_backward']

        # 创建正向条带（红色）
        forward_poly = create_strip_along_line(coords, w_forward, offset_direction=1)
        if forward_poly:
            forward_strips.append(forward_poly)
            forward_color = list(mcolors.to_rgb(color_forward)) + [row['alpha_forward']]
            forward_colors.append(forward_color)

        # 创建反向条带（蓝色）
        backward_poly = create_strip_along_line(coords, w_backward, offset_direction=-1)
        if backward_poly:
            backward_strips.append(backward_poly)
            backward_color = list(mcolors.to_rgb(color_backward)) + [row['alpha_backward']]
            backward_colors.append(backward_color)

    # 10. 绘制单向边的条带
    for idx, row in data_oneway.iterrows():
        if row.geometry.geom_type != 'LineString':
            continue

        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue

        # 获取宽度
        w = row['width']

        # 创建单向边条带（绿色，居中）
        oneway_poly = create_strip_along_line(coords, w, offset_direction=0)
        if oneway_poly:
            oneway_strips.append(oneway_poly)
            oneway_color = list(mcolors.to_rgb(color_oneway)) + [row['alpha']]
            oneway_colors.append(oneway_color)

    # 11. 绘制所有条带
    # 先绘制双向边的反向条带（底层）
    if backward_strips:
        backward_collection = PolyCollection(
            backward_strips,
            facecolors=backward_colors,
            edgecolors=[(0, 0, 0, 0.1)],
            linewidths=0.5,
            zorder=1
        )
        ax.add_collection(backward_collection)

    # 绘制单向边条带（中层）
    if oneway_strips:
        oneway_collection = PolyCollection(
            oneway_strips,
            facecolors=oneway_colors,
            edgecolors=[(0, 0, 0, 0.1)],
            linewidths=0.5,
            zorder=2
        )
        ax.add_collection(oneway_collection)

    # 绘制双向边的正向条带（顶层）
    if forward_strips:
        forward_collection = PolyCollection(
            forward_strips,
            facecolors=forward_colors,
            edgecolors=[(0, 0, 0, 0.1)],
            linewidths=0.5,
            zorder=3
        )
        ax.add_collection(forward_collection)

    # 12. 可选：绘制原始几何作为参考
    for idx, row in gdf.iterrows():
        if row['type'] == 'walk':
            # 检查几何类型
            if row.geometry.geom_type == 'LineString':
                # 提取坐标
                coords = list(row.geometry.coords)
                if len(coords) >= 2:
                    # 分离x和y坐标
                    x = [c[0] for c in coords]
                    y = [c[1] for c in coords]
                    ax.plot(x, y, color='grey',linestyle='--')
            elif row.geometry.geom_type == 'MultiLineString':
                # 处理多段线
                for line in row.geometry.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        x = [c[0] for c in coords]
                        y = [c[1] for c in coords]
                        ax.plot(x, y, color='grey',linestyle='--')

    # 13. 设置图形
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_title('Bidirectional and One-way Flow Visualization', fontsize=16, fontweight='bold')

    # 14. 添加图例
    legend_elements = [
        Line2D([0], [0], color=color_forward, lw=3, alpha=alpha_range[1],
               label=f'Bidirectional Forward Flow ({color_forward})'),
        Line2D([0], [0], color=color_backward, lw=3, alpha=alpha_range[1],
               label=f'Bidirectional Backward Flow ({color_backward})'),
        Line2D([0], [0], color=color_oneway, lw=3, alpha=alpha_range[1],
               label=f'One-way Flow ({color_oneway})'),
        Line2D([0], [0], color='gray', lw=1, linestyle='--',
               label='walk only'),
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # 15. 添加信息框
    info_text = (f'Global Flow Range: {min_flow:.0f} - {max_flow:.0f}\n'
                 f'Alpha Range: {alpha_range[0]:.1f} - {alpha_range[1]:.1f}\n'
                 f'Base Width: {base_width}\n'
                 f'Bidirectional Edges: {len(data)}\n'
                 f'One-way Edges: {len(data_oneway)}')

    ax.text(0.02, 0.02, info_text,
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 16. 可选：关闭坐标轴
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_flow_with_arrows(gdf,
                          buildings,
                          flow_field,
                          forward_field,
                          backward_field,
                          cmap='viridis',  # 新增：matplotlib色带
                          width_range=(0.1, 3.0),
                          white_center_width=5,
                          limit=None,
                          arrow_size=0.2,  # 新增：箭头大小
                          figsize=(12, 10)):
    """
    绘制流量条带，用颜色和宽度表示流量大小，用箭头表示方向
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    # 1. 分离双向边和单向边
    mask_bidirectional = gdf[forward_field].notna() & gdf[backward_field].notna()
    data = gdf[mask_bidirectional].copy()

    mask_oneway = gdf[flow_field].notna() & (gdf[forward_field].isna() | gdf[backward_field].isna())
    data_oneway = gdf[mask_oneway].copy()

    print(f"双向边数量: {len(data)}")
    print(f"单向边数量: {len(data_oneway)}")

    # 2. 收集所有流量数据
    all_flows = []

    if len(data) > 0:
        all_flows.extend(data[forward_field].values)
        all_flows.extend(data[backward_field].values)

    if len(data_oneway) > 0:
        all_flows.extend(data_oneway[flow_field].values)

    if not all_flows:
        print("警告：没有找到流量数据")
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    # 3. 确定归一化范围
    data_min = min(all_flows)
    data_max = max(all_flows)

    if limit is None:
        vmin, vmax = data_min, data_max
    else:
        vmin, vmax = limit[0], limit[1]
        if vmin > vmax:
            vmin, vmax = vmax, vmin


    # 4. 获取宽度范围
    width_min, width_max = width_range

    # 5. 创建颜色映射
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    # 6. 计算归一化值和颜色/宽度的函数
    def normalize_flow(flow):
        """将流量值归一化到[0, 1]范围"""
        clipped_flow = np.clip(flow, vmin, vmax)

        if vmax == vmin:
            return 0.5

        return (clipped_flow - vmin) / (vmax - vmin)

    def calc_width(flow):
        """根据归一化值计算宽度"""
        norm = normalize_flow(flow)
        return width_min + norm * (width_max - width_min)

    def calc_color(flow):
        """根据归一化值计算颜色"""
        norm = normalize_flow(flow)
        return cmap(norm)

    # 7. 为双向边计算宽度和颜色
    if len(data) > 0:
        # 双向边：每个方向单独计算
        data['width_forward'] = data[forward_field].apply(calc_width)
        data['width_backward'] = data[backward_field].apply(calc_width)
        data['color_forward'] = data[forward_field].apply(calc_color)
        data['color_backward'] = data[backward_field].apply(calc_color)
        data['norm_forward'] = data[forward_field].apply(normalize_flow)
        data['norm_backward'] = data[backward_field].apply(normalize_flow)

    # 8. 为单向边计算宽度和颜色
    if len(data_oneway) > 0:
        data_oneway['width'] = data_oneway[flow_field].apply(calc_width)
        data_oneway['color'] = data_oneway[flow_field].apply(calc_color)
        data_oneway['norm'] = data_oneway[flow_field].apply(normalize_flow)


    # 9. 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    #绘制建筑
    buildings.plot(ax=ax, color="grey", alpha=0.2, zorder=0)

    # 10. 存储条带和箭头信息
    strips = []  # 所有条带
    colors = []  # 所有颜色
    arrow_positions = []  # 箭头位置信息
    arrow_directions = []  # 箭头方向信息
    arrow_colors = []

    # 11. 辅助函数：创建沿线的条带
    def create_strip_along_line(coords, width, offset_direction=0):
        """
        沿着线的坐标点创建条带多边形
        offset_direction: 0=居中, 1=正向偏移, -1=反向偏移
        """
        if len(coords) < 2:
            return None

        upper_points = []
        lower_points = []

        for i in range(len(coords) - 1):
            start = np.array(coords[i])
            end = np.array(coords[i + 1])

            direction = end - start
            length = np.linalg.norm(direction)

            if length == 0:
                continue

            dir_unit = direction / length
            normal = np.array([-dir_unit[1], dir_unit[0]])

            # 计算偏移
            if offset_direction == 0:  # 居中（单向边）
                offset = width /2
                upper_offset = normal * offset
                lower_offset = normal * -offset
            elif offset_direction == 1:  # 正向偏移
                upper_offset = normal * width
                lower_offset = np.array([0.0, 0.0])
            else:  # 反向偏移
                upper_offset = np.array([0.0, 0.0])
                lower_offset = normal * -width

            if i == 0:
                upper_points.append(start + upper_offset)
                lower_points.append(start + lower_offset)

            upper_points.append(end + upper_offset)
            lower_points.append(end + lower_offset)

        if not upper_points or not lower_points:
            return None

        polygon = upper_points + list(reversed(lower_points))

        if len(polygon) > 0 and not np.array_equal(polygon[0], polygon[-1]):
            polygon.append(polygon[0])

        return polygon

    # 12. 绘制双向边的条带
    for idx, row in data.iterrows():
        if row.geometry.geom_type != 'LineString':
            continue

        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue

        # 获取宽度和颜色
        w_forward = row['width_forward']
        w_backward = row['width_backward']
        color_forward = row['color_forward']
        color_backward = row['color_backward']

        # 创建正向条带
        forward_poly = create_strip_along_line(coords, w_forward+white_center_width/2, offset_direction=1)
        if forward_poly:
            strips.append(forward_poly)
            colors.append(color_forward)

            # 计算正向箭头的法线方向（最后一个线段）
            start_point = np.array(coords[-2])
            end_point = np.array(coords[-1])
            segment_direction = end_point - start_point
            length = np.linalg.norm(segment_direction)
            if length > 0:
                dir_unit = segment_direction / length
                normal = np.array([-dir_unit[1], dir_unit[0]])
                # 箭头位置：终点 + 法线方向 * (w_forward / 2)
                arrow_pos = end_point + normal * (w_forward / 2)
            else:
                arrow_pos = end_point

            # 记录箭头位置和方向（方向为最后一个线段的方向，即从起点指向终点）
            arrow_positions.append(arrow_pos)
            arrow_directions.append(segment_direction)
            arrow_colors.append(color_forward)


        # 创建反向条带
        backward_poly = create_strip_along_line(coords, w_backward+white_center_width/2, offset_direction=-1)
        if backward_poly:
            strips.append(backward_poly)
            colors.append(color_backward)

            # 计算反向箭头的法线方向（第一个线段）
            start_point = np.array(coords[0])
            next_point = np.array(coords[1])
            segment_direction = next_point - start_point  # 第一个线段的方向（从起点到第二个点）
            length = np.linalg.norm(segment_direction)
            if length > 0:
                dir_unit = segment_direction / length
                normal = np.array([-dir_unit[1], dir_unit[0]])
                # 箭头位置：起点 - 法线方向 * (w_backward / 2)
                arrow_pos = start_point - normal * (w_backward / 2)
            else:
                arrow_pos = start_point

            # 记录箭头位置和方向（方向为反向，即从第二个点指向起点）
            arrow_directions.append(start_point - next_point)  # 反向
            arrow_positions.append(arrow_pos)
            arrow_colors.append(color_backward)

        # 中心白色条带
        center_poly = create_strip_along_line(coords, white_center_width, offset_direction=0)
        if center_poly:
            strips.append(center_poly)
            colors.append('white')

    # 13. 绘制单向边的条带
    for idx, row in data_oneway.iterrows():
        if row.geometry.geom_type != 'LineString':
            continue

        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue

        # 获取宽度和颜色
        w = row['width']
        color = row['color']

        # 检查是否包含'reversed'字段
        has_reversed = 'reversed' in row.index

        # 创建条带（居中）
        oneway_poly = create_strip_along_line(coords, w, offset_direction=0)
        if oneway_poly:
            strips.append(oneway_poly)
            colors.append(color)

            # 根据'reversed'字段确定箭头方向
            if has_reversed and row['reversed'] == 1:
                # reversed=1：反向箭头（从终点指向起点）
                start_point = np.array(coords[0])  # 第一个点
                end_point = np.array(coords[1])  # 第二个点
                segment_direction = start_point - end_point  # 反向方向

                # 箭头位置放在起点（第一个点）
                arrow_pos = start_point

            else:
                # reversed=0或不包含该字段：正向箭头（默认方向）
                start_point = np.array(coords[-2])  # 倒数第二个点
                end_point = np.array(coords[-1])  # 最后一个点
                segment_direction = end_point - start_point  # 正向方向

                # 箭头位置放在终点（最后一个点）
                arrow_pos = end_point

                # if has_reversed:
                #     print(f"单向边 {idx}: reversed={row['reversed']}，箭头方向正向")
                # else:
                #     print(f"单向边 {idx}: 无reversed字段，使用默认正向方向")

            # 记录箭头位置和方向
            arrow_positions.append(arrow_pos)
            arrow_directions.append(segment_direction)
            arrow_colors.append(color)

    # 14. 绘制所有条带
    if strips:
        strip_collection = PolyCollection(
            strips,
            facecolors=colors,
            edgecolors=[(0, 0, 0, 0.1)],  # 浅色边框
            linewidths=0.5,
            zorder=2
        )
        ax.add_collection(strip_collection)

    # 15. 绘制箭头
    for pos, dir_vec,arr_color in zip(arrow_positions, arrow_directions,arrow_colors):
        if np.linalg.norm(dir_vec) == 0:
            continue

        # 归一化方向向量
        dir_norm = dir_vec / np.linalg.norm(dir_vec)

        # 计算箭头端点
        arrow_start = pos - dir_norm * arrow_size * 0.5
        arrow_end = pos + dir_norm * arrow_size * 0.5

        # 创建箭头
        arrow = FancyArrowPatch(
            arrow_start, arrow_end,
            arrowstyle='-|>',
            mutation_scale=arrow_size * 20,  # 控制箭头大小
            color=arr_color,
            linewidth=1.5,
            fill=True,
            zorder=3
        )
        ax.add_patch(arrow)

    # 16. 绘制walk类型的边
    for idx, row in gdf.iterrows():
        if row['type'] == 'walk':
            if row.geometry.geom_type == 'LineString':
                coords = list(row.geometry.coords)
                if len(coords) >= 2:
                    x = [c[0] for c in coords]
                    y = [c[1] for c in coords]
                    ax.plot(x, y, color='black', linestyle='--', linewidth=0.8, zorder=1)
            elif row.geometry.geom_type == 'MultiLineString':
                for line in row.geometry.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        x = [c[0] for c in coords]
                        y = [c[1] for c in coords]
                        ax.plot(x, y, color='black', linestyle='--', linewidth=0.8, zorder=1)

    # 17. 设置图形
    ax.autoscale_view()
    ax.set_aspect('equal')


    # 18. 添加颜色条
    # 创建归一化的颜色映射
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # 添加颜色条
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('drive saturation', fontsize=12)

    # 19. 添加图例
    legend_elements = [
        Line2D([0], [0], color='grey', linestyle='--', linewidth=1,
               label='Walking paths')
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # # 20. 添加信息框
    # info_text = (f'Data Range: {data_min:.0f} - {data_max:.0f}\n'
    #              f'Normalization Range: {vmin:.0f} - {vmax:.0f}\n'
    #              f'Width Range: {width_min:.2f} - {width_max:.2f}\n'
    #              f'Colormap: {cmap.name if hasattr(cmap, "name") else str(cmap)}\n'
    #              f'Bidirectional Edges: {len(data)}\n'
    #              f'One-way Edges: {len(data_oneway)}')
    #
    # ax.text(0.02, 0.02, info_text,
    #         transform=ax.transAxes,
    #         fontsize=9,
    #         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))



    return fig, ax

def process_edges(gdf_current, G, fields_to_keep):
    """
    处理gdf中的边，根据图G的信息添加字段

    参数:
    ----------
    gdf_current : GeoDataFrame
        包含边的GeoDataFrame
    G : networkx.Graph 或 DiGraph
        包含边信息的图
    fields_to_keep : list
        需要从G中提取的字段列表

    返回:
    ----------
    GeoDataFrame : 处理后的GeoDataFrame
    """
    # 1. 创建gdf的副本
    gdf = gdf_current.copy()

    def extract_start_point(geom):
        """提取起点坐标"""
        if geom.geom_type == 'LineString':
            return geom.coords[0]
        else:
            return None

    def extract_end_point(geom):
        """提取终点坐标"""
        if geom.geom_type == 'LineString':
            return geom.coords[-1]
        else:
            return None

    gdf['u'] = gdf.geometry.apply(extract_start_point)
    gdf['v'] = gdf.geometry.apply(extract_end_point)

    print("gdf列名:", gdf.columns.tolist())

    # 3. 创建图的查找索引（性能优化）
    edge_data_dict = {}
    for u, v, data in G.edges(data=True):
        # 存储两个方向的键，以便双向查找
        edge_data_dict[(u, v)] = data
        # 如果需要考虑反向，也存储(v, u)
        # 注意：这取决于你的具体需求

    for idx, row in gdf.iterrows():
        u = row['u']
        v = row['v']
        gdf.at[idx, 'type'] = 'drive'
        gdf.at[idx, 'reversed'] = 0

        # 检查正向边
        if (u, v) in edge_data_dict:
            data = edge_data_dict[(u, v)]
            if data['oneway']== 1:
                # 单向边
                for field in fields_to_keep:
                    if field in data:
                        gdf.at[idx, field] = data[field]

            elif data['oneway'] == 0:
                # 双向边
                for field in fields_to_keep:
                    if field in data:
                        # 设置正向字段
                        gdf.at[idx, f'forward_{field}'] = data[field]

                        # 检查反向边并设置反向字段
                        if (v, u) in edge_data_dict:
                            reverse_data = edge_data_dict[(v, u)]
                            if field in reverse_data:
                                gdf.at[idx, f'backward_{field}'] = reverse_data[field]
                                # 标记反向边待删除（避免重复处理）
                                G.remove_edge(v, u)
                        else:
                            print('not real double ways')

        # 检查反向边
        elif (v, u) in edge_data_dict:
            data = edge_data_dict[(v, u)]
            if data['oneway']== 1:
                for field in fields_to_keep:
                    if field in data:
                        gdf.at[idx, field] = data[field]
                        gdf.at[idx, 'reversed'] = 1
        else:
            gdf.at[idx, 'type'] = 'walk'

    return gdf

# 预计算所有需要的常数，避免重复计算
_CAR_SPEED_MPM = 40 * 1000 / 60
_BIKE_SPEED_MPM = 15 * 1000 / 60
_WALK_SPEED_MPM = 5 * 1000 / 60

def get_car_probability_optimized(d_meters):
    """
    高度优化的汽车概率计算函数，所有计算都内联且使用本地变量
    """
    # 计算时间
    walk_time = d_meters / _WALK_SPEED_MPM
    bike_time = d_meters / _BIKE_SPEED_MPM + 1
    car_time = d_meters / _CAR_SPEED_MPM + 3

    # 计算效用
    walk_util = -0.2 - 0.2 * walk_time

    # 自行车效用
    w_asc_bike = 1.0 / (1.0 + math.exp(-0.005 * (d_meters - 200)))
    asc_bike = -1 * w_asc_bike
    penalty_bike = -7.0 / (1 + d_meters / 100)
    bike_util = asc_bike - 0.10 * bike_time + penalty_bike

    # 汽车效用
    w_asc_car = 1.0 / (1.0 + math.exp(-0.005 * (d_meters - 500)))
    asc_car = 0.5 * w_asc_car
    penalty_car = -10.0 / (1 + d_meters / 200)
    car_util = asc_car - 0.08 * car_time + penalty_car

    # 找到最大效用
    if walk_util >= bike_util and walk_util >= car_util:
        max_util = walk_util
    elif bike_util >= car_util:
        max_util = bike_util
    else:
        max_util = car_util

    # 计算概率
    exp_walk = math.exp(walk_util - max_util)
    exp_bike = math.exp(bike_util - max_util)
    exp_car = math.exp(car_util - max_util)

    total = exp_walk + exp_bike + exp_car
    return exp_car / total if total > 0 else 1 / 3

# 预计算概率表（示例，使用100米分辨率）
PROB_TABLE = {d: get_car_probability_optimized(d) for d in range(0, 20001, 100)}

def get_car_probability(d_meters):
    """通过查表获取汽车概率，非常快"""
    key = round(d_meters / 100) * 100
    if key >20000:
        return 1

    return PROB_TABLE.get(key, 0.5)  # 默认为0.5



def build_od_demand_dict(G, K: float = 0.4) -> dict:
    od_demand = {}
    nodes = list(G.nodes())

    prod_strength = {}
    attr_strength = {}

    for n in nodes:
        data = G.nodes[n]
        gen = data.get('generate', 0.0)
        absb = data.get('absorb', 0.0)
        a_val = data.get('a', 0.0)
        b_val = data.get('b', 0.0)

        prod_strength[n] = gen * a_val
        attr_strength[n] = absb * b_val

    for s in nodes:
        ps = prod_strength[s]
        if ps <= 0:
            continue
        for t in nodes:
            if s == t:
                continue
            at = attr_strength[t]
            if at <= 0:
                continue

            flow = K * ps * at
            if flow > 0:
                od_demand[(s, t)] = flow

    return od_demand

connect_node_list=[(1254671.886788083, 5431788.372374507), (1254736.0251271345, 5432218.1802799655), (1254833.7547325857, 5432368.825693187), (1256463.2749907007, 5433084.271463453), (1258466.299413995, 5431609.90917222), (1258489.8076031606, 5430904.972644913), (1257887.0048331157, 5430002.580641085), (1257546.2574919362, 5429423.505131982), (1255775.5322349705, 5429841.297012074), (1255202.0701854478, 5430857.627919382)]
def calculate_traffic(graph, od_demand, pass_count='drive_pass_count',
                      traffic_capacity='drive_traffic_capacity',
                      saturation='drive_saturation') -> int:
    total_demand = sum(od_demand.values())
    distance_factor = 0
    for u, v, data in graph.edges(data=True):
        data[pass_count] = 0.0

    od_by_source = {}
    for (s, t), flow in od_demand.items():
        if s != t and flow > 0:
            if s not in od_by_source:
                od_by_source[s] = []
            od_by_source[s].append((t, flow))

    # if not od_by_source:
    #     return

    try:
        for source, (distances, paths) in nx.all_pairs_dijkstra(graph, weight="edge_length"):
            if source not in od_by_source:
                continue

            for target, base_flow in od_by_source[source]:
                if target not in distances:
                    continue

                distance = distances[target]
                path = paths[target]


                if pass_count != 'drive_pass_count':
                    continue
                else:
                    if (source in connect_node_list) or (target  in connect_node_list):
                        flow=base_flow
                    #按出行方式给分配车流
                    else:
                        flow = base_flow * get_car_probability(distance)
                    distance_factor += (flow / total_demand) * distance

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if graph.has_edge(u, v):
                        graph[u][v][pass_count] += flow

        for u, v, data in graph.edges(data=True):
            if traffic_capacity == 'drive_traffic_capacity':
                l = 1
                data[traffic_capacity] = 1800 * data['lanes'] * 0.8 * l
            elif traffic_capacity == 'walk_traffic_capacity':
                data[traffic_capacity] = (3600 * 1 / 1 * 0.75) * data['ped'] * 0.5
            else:
                print('wrong traffic_capacity')

        for u, v, data in graph.edges(data=True):
            if data[traffic_capacity] != 0:
                data[saturation] = data[pass_count] / data[traffic_capacity]
            else:
                data[saturation] = 0

    except Exception as e:
        print(f"Traffic assignment failed: {e}")
    return distance_factor


# 使用示例
if __name__ == "__main__":
    filepath =r"D:\first_term\Florence\chutu\normal_optimized\normal_digraph_real.pkl"#替换为你需要绘制的图文件路径
    with open(filepath, 'rb') as f:
        G= pickle.load(f)
    #计算优化前交通数据
    od_demand = build_od_demand_dict(G)
    distance_factor=calculate_traffic(G,od_demand)
    filepath = r"D:\first_term\Florence\chutu\normal_optimized\normal_current"#替换为你需要保存的png文件路径
    with open(filepath, 'wb') as f:
            pickle.dump(G, f)

    # gdf_1 = gpd.read_file("D:" + r"\first_term\Florence\traffic_model\completed_street.geojson")
    # gdf_1=gdf_1.to_crs(epsg=3857)
    # buildings = gpd.read_file(r"D:\first_term\Florence\shortest\data\buildings_function.geojson")
    # buildings = buildings.to_crs(epsg=3857)
    # # 2. 转换为GeoDataFrame，保留'flow'字段
    # gdf = process_edges(
    #     gdf_1,
    #     G,
    #     fields_to_keep=['drive_saturation']
    # )
    #
    # jet = plt.cm.jet
    # # 创建一个新的colormap，只取jet的0-0.9
    # new_jet = plt.cm.colors.LinearSegmentedColormap.from_list(
    #     'trunc({n},{a:.2f},{b:.2f})'.format(n=jet.name, a=0, b=0.9),
    #     jet(np.linspace(0, 0.9, 256))
    # )
    #
    # fig,ax= plot_flow_with_arrows(gdf, buildings,'drive_saturation', 'forward_drive_saturation', 'backward_drive_saturation', cmap=new_jet,
    #                               width_range=(10.0, 20.0), limit=[0, 1], arrow_size=0.8, figsize=(15, 15))
    # ax.set_title('current', fontsize=16, fontweight='bold')
    #
    # # 21. 可选：关闭坐标轴
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()


