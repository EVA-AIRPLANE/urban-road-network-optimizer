import networkx as nx
import itertools
import math
import pickle
from typing import Dict, Tuple, Optional, List
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import geopandas as gpd
import momepy as mp



def draw_direction(streets_graph,buildings,column, forward_cmp, reserves_cmp, size,limit=None):
    forward_zero_edges = [(u, v) for u, v, d in streets_graph.edges(data=True) if (d[column] == 0 and d.get('reversed') == 0)]
    forward_nonzero_edges = [(u, v) for u, v, d in streets_graph.edges(data=True) if (d[column] != 0 and d.get('reversed') == 0)]
    reverse_zero_edges = [(u, v) for u, v, d in streets_graph.edges(data=True) if(d[column] == 0 and d.get('reversed') == 1)]
    reverse_nonzero_edges = [(u, v) for u, v, d in streets_graph.edges(data=True) if(d[column] != 0 and d.get('reversed') == 1)]


    # 创建颜色数组
    forward_colors = [streets_graph[u][v][column] for u, v in forward_nonzero_edges]
    reverse_colors = [streets_graph[u][v][column] for u, v in reverse_nonzero_edges]

    # fwd_enhanced= enhance_values(forward_colors)
    # res_enhanced= enhance_values(reverse_colors)

    if limit==None:
        limit=[min(min(forward_colors),min(reverse_colors)),max(max(reverse_colors),max(forward_colors))]
    vmin,vmax = limit[0],limit[1]
    norm=Normalize(vmin=vmin,vmax=vmax)

    # 绘制
    fig,ax = plt.subplots(figsize=size)
    pos = {node: node for node in streets_graph.nodes()}

    #绘制节点
    nx.draw_networkx_nodes(streets_graph, pos,ax=ax, node_color='black', node_size=20)

    #绘制正向非零边（直线）
    nx.draw_networkx_edges(
        streets_graph, pos,
        ax=ax,
        edgelist=forward_nonzero_edges,
        edge_color=forward_colors,
        edge_cmap=forward_cmp,
        edge_vmin=vmin,
        edge_vmax=vmax,
        width=1,
        node_size=20,
        connectionstyle='arc3,rad=0.0',  # 直线
        arrows=True,
        arrowsize=10

    )

    # 绘制反向非零边（弧线）
    nx.draw_networkx_edges(
        streets_graph, pos,
        ax=ax,
        edgelist=reverse_nonzero_edges,
        edge_color=reverse_colors,
        edge_cmap=reserves_cmp,
        edge_vmin=vmin,
        edge_vmax=vmax,
        width=1,
        node_size=20,
        connectionstyle='arc3,rad=0.2',  # 正向弧线
        arrows=True,
        arrowsize=10
    )


    # 绘制正向零边（直线）
    nx.draw_networkx_edges(
        streets_graph, pos,
        ax=ax,
        edgelist=forward_zero_edges,
        edge_color='grey',
        width=1,
        node_size=20,
        connectionstyle='arc3,rad=0.0',  # 直线
        arrows=True,
        arrowsize=10

    )

    # 绘制反向零边（弧线）
    nx.draw_networkx_edges(
        streets_graph, pos,
        ax=ax,
        edgelist=reverse_zero_edges,
        edge_color='grey',
        width=1,
        node_size=20,
        connectionstyle='arc3,rad=0.2',  # 正向弧线
        arrows=True,
        arrowsize=10
    )
    buildings.plot(ax=ax, color="lightblue", alpha=0.5,zorder=0)

    # 5. 添加颜色条（图例）
    sm = ScalarMappable(norm=norm, cmap=forward_cmp)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label(column, fontsize=12)
    if forward_cmp != reserves_cmp:
        sm1 = ScalarMappable(norm=norm, cmap=reserves_cmp)
        sm1.set_array([])
        cbar = plt.colorbar(sm1, ax=plt.gca(), shrink=0.8)
        cbar.set_label(column, fontsize=12)

    plt.title(column)
    plt.axis('off')
    plt.show()

def build_od_demand_dict(
        G,  # 注意：这里可以是 Graph 或 DiGraph，我们只读节点
        K: float = 0.4
) -> dict:
    """
    根据节点属性构建全局 OD 需求字典。

    节点必须包含字段: 'generate', 'absorb', 'a', 'b'
    flow(s -> t) = K * generate[s] * a[s] * absorb[t] * b[t]

    返回:
        od_demand: {(s, t): flow}，仅包含 flow > 0 的 OD 对
    """
    od_demand = {}
    nodes = list(G.nodes())

    # 预计算每个节点的产生/吸引强度（避免重复乘法）
    prod_strength = {}  # s -> generate[s] * a[s]
    attr_strength = {}  # t -> absorb[t] * b[t]

    for n in nodes:
        data = G.nodes[n]
        gen = data.get('generate', 0.0)
        absb = data.get('absorb', 0.0)
        a_val = data.get('a', 0.0)
        b_val = data.get('b', 0.0)

        prod_strength[n] = gen * a_val
        attr_strength[n] = absb * b_val

    # 构建 OD 对
    for s in nodes:
        ps = prod_strength[s]
        if ps <= 0:
            continue  # 无出行产生能力
        for t in nodes:
            if s == t:
                continue
            at = attr_strength[t]
            if at <= 0:
                continue  # 无吸引能力

            flow = K * ps * at
            if flow > 0:
                od_demand[(s, t)] = flow

    return od_demand


def calculate_traffic(
    graph: nx.DiGraph,
    od_demand: Dict[Tuple[int, int], float],
    pass_count='drive_pass_count',
    traffic_capacity='drive_traffic_capacity',
    saturation='drive_saturation',
    walk_distance: float = 500.0
) -> None:

    # 初始化边流量
    for u, v, data in graph.edges(data=True):
        data[pass_count] = 0.0

    # 构建反向 OD 映射：按 source 分组，加速查找
    od_by_source = {}
    for (s, t), flow in od_demand.items():
        if s != t and flow > 0:
            if s not in od_by_source:
                od_by_source[s] = []
            od_by_source[s].append((t, flow))

    if not od_by_source:
        return

    # 流式计算每个 source 的最短路径
    try:
        for source, (distances, paths) in nx.all_pairs_dijkstra(graph, weight="edge_length"):
            if source not in od_by_source:
                continue  # 该起点无出行需求

            for target, base_flow in od_by_source[source]:
                if target not in distances:
                    continue  # 不可达

                distance = distances[target]
                path = paths[target]

                # 判断模式
                if distance <= walk_distance:
                    if pass_count != 'walk_pass_count':
                        continue
                    flow =base_flow
                else:
                    if pass_count != 'drive_pass_count':
                        continue
                    flow =base_flow

                # 分配到路径边
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if graph.has_edge(u, v):
                        graph[u][v][pass_count] += flow

        # 计算每条道路的承载量
        for u, v, data in graph.edges(data=True):
            if traffic_capacity == 'drive_traffic_capacity':
                l = 1
                data[traffic_capacity] = 1800 * data['lanes'] * 0.8 * l
            elif traffic_capacity == 'walk_traffic_capacity':
                data[traffic_capacity] = (3600 * 1 / 1 * 0.75) * data['ped'] * 0.5
            else:
                print('wrong traffic_capacity')

        # 计算饱和度
        for u, v, data in graph.edges(data=True):
            if data[traffic_capacity] != 0:
                data[saturation] = data[pass_count] / data[traffic_capacity]
            else:
                data[saturation] = 0

    except Exception as e:
        print(f"⚠️ Traffic assignment failed: {e}")
        # 可选：保留初始 0 值或抛出异常


def check_connectivity_fast(
        G: nx.DiGraph,
        od_demand: Dict[Tuple[int, int], float],
        max_unreachable_ratio: float = 0.1  # 允许最多 10% 的需求不可达
) -> bool:
    if not od_demand:
        return True  # 无需求，视为连通

    total_demand = sum(od_demand.values())
    if total_demand <= 0:
        return True

    unreachable_demand = 0.0
    node_set = set(G.nodes())

    for (o, d), flow in od_demand.items():
        if flow <= 0:
            continue
        # 快速检查：起点/终点是否存在
        if o not in node_set or d not in node_set:
            unreachable_demand += flow
            continue
        # 检查是否存在路径（不计算最短路，只判断连通性）
        try:
            if not nx.has_path(G, o, d):
                unreachable_demand += flow
        except Exception:
            # 图结构异常（如孤立节点），视为不可达
            unreachable_demand += flow

        # 提前终止：如果已超阈值
        if unreachable_demand / total_demand > max_unreachable_ratio:
            return False

    return (unreachable_demand / total_demand) <= max_unreachable_ratio

def optimize_bidirectional_lanes(q1,q2,l,width,current_n1,current_n2):
    if None in (q1,q2,l,width,current_n1,current_n2):
        print('wrong edge')
        return 0,0,0,0,0,0
    #预设一些固定参数
    C0=1800* 0.8 * l
    lane_width = 3.5
    target_sat = 0.8
    weights = {'sat': 1.0, 'lanes': 0.01, 'change': 0.01}

    N_max = max(0, int(width // lane_width))

    best_score = float('inf')  #分数越小越好
    best_n1, best_n2 = 1, 1
    best_s1, best_s2 = 0.0, 0.0
    ch1, ch2 = 0.0, 0.0

    if N_max<=1:   #宽度只能提供一个车道，两个方向混行
        equiv_C = 0.5 * C0
        s1 = q1 / equiv_C if equiv_C > 0 else (float('inf') if q1 > 0 else 0)
        s2 = q2 / equiv_C if equiv_C > 0 else (float('inf') if q2 > 0 else 0)
        return 0.5, 0.5, s1, s2,ch1,ch2  # 用 (0.5,0.5) 表示混行，需外部特殊处理

    else:
        found=False
        for total in range(2, N_max + 1):  # 总车道数从2到N_max（至少2才能双向）
            for n1 in range(1, total):  # n1 >=1
                n2 = total - n1  # n2 = total - n1 >=1
                if n2 < 1:
                    continue


                s1 = q1 / (n1 * C0) if n1 > 0 else float('inf')
                s2 = q2 / (n2 * C0) if n2 > 0 else float('inf')

                sat_error = abs(s1 - target_sat) + abs(s2 - target_sat)
                total_lanes_used = n1 + n2  # <= N_max
                change = abs(n1 - current_n1) + abs(n2 - current_n2)

                score = (
                        weights['sat'] * sat_error +
                        weights['lanes'] * total_lanes_used +  # 鼓励少用
                        weights['change'] * change
                )

                if score < best_score:
                    best_score = score
                    best_n1, best_n2 = n1, n2
                    best_s1, best_s2 = s1, s2
                    ch1 = n1 - current_n1
                    ch2 = n2 - current_n2
                    found = True

            # 如果没找到（理论上不会，因为 N_max>=2 时至少有 (1,1)）
        if not found:
            # 回退到 (1,1)
            n1, n2 = current_n1, current_n2
            s1 = q1 / (n1 * C0)
            s2 = q2 / (n2 * C0)
            ch1=0.0
            ch2=0.0
            return n1, n2, s1, s2,ch1, ch2

        return best_n1, best_n2, best_s1, best_s2,ch1, ch2


def optimize_directional_lanes(q,l,width,current_n):
    if None in (q,l,width,current_n):
        print('wrong edge')
        return 0,0,0
    C0=1800* 0.8 * l
    lane_width = 3.5
    target_sat = 0.8
    weights = {'sat': 1.0, 'lanes': 0.01, 'change': 0.01}

    N_max = max(0, int(width // lane_width))

    best_score = float('inf')  # 分数越小越好
    best_n= 1
    best_s= 0.0
    ch=0.0
    found=False
    for n in range(1, N_max + 1):
        s=q / (n * C0)
        sat_error = abs(s - target_sat)
        change = abs(n - current_n)
        score = (
                weights['sat'] * sat_error +
                weights['lanes'] * n +  # 鼓励少用
                weights['change'] * change
        )
        if score < best_score:
            best_score = score
            best_n = n
            best_s=s
            ch = n - current_n
            found = True
        if not found:
            n=current_n
            s= q / (n * C0)
            ch=0.0
            return n,s,ch
    return best_n, best_s,ch

def optimize_drive_lanes(streets_graph,pass_count):
    for u, v, data in streets_graph.edges(data=True):
        l = 1
        if data['oneway']==1:
            best_n,best_s,ch=optimize_directional_lanes(data[pass_count],l,data['width'],data['lanes'])
            data['best_lanes_num']=best_n
            data['best_sat_num']=best_s
            data['change_num']=ch
        elif data['oneway']==0:
            if data['reversed']==0:
                q1=data[pass_count]
                q2=streets_graph.edges[v,u][pass_count]
                current_n1=data['lanes']
                current_n2=streets_graph.edges[v,u]['lanes']
                best_n1, best_n2, best_s1, best_s2,ch1, ch2=(
                    optimize_bidirectional_lanes(
                        q1,
                        q2,
                        l,
                        data['width'],
                        current_n1,
                        current_n2
                    )
                )
                data['best_lanes_num']=best_n1
                streets_graph.edges[v,u]['best_lanes_num']=best_n2
                data['best_sat_num']=best_s1
                streets_graph.edges[v, u]['best_sat_num'] = best_s2
                data['change_num'] = ch1
                streets_graph.edges[v, u]['change_num'] = ch2
            else:
                continue


def copy_graph_with_selected_edge_attrs(G, edge_attrs_to_keep):
    """
    复制图 G，但边属性只保留 edge_attrs_to_keep 中指定的字段。
    节点属性全部保留（通常无害）。
    """
    H = nx.DiGraph()  # 或 nx.Graph()，根据你的图类型
    H.add_nodes_from(G.nodes(data=True))  # 保留所有节点及其属性

    # 只复制指定的边属性
    for u, v, data in G.edges(data=True):
        filtered_data = {
            key: data[key]
            for key in edge_attrs_to_keep
            if key in data
        }
        H.add_edge(u, v, **filtered_data)

    return H


def optimize_subgraph_with_simulated_annealing(
        G_full: nx.DiGraph,
        subgraph: nx.Graph,
        max_iterations: int = 1000,  # 模拟退火迭代次数
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.995
) -> Optional[nx.DiGraph]:
    import random
    # 设置权重
    weights = {'sat': 1.0, 'lanes': 0.01}

    # Step 1: 提取子图中的无向边列表（去重）
    undirected_edges = list(subgraph.edges())
    if len(undirected_edges) == 0:
        return None

    # Step 2: 预计算每条边的属性
    edge_properties = []
    for u, v in undirected_edges:
        width = subgraph[u][v].get('width', 7.0)
        lane_width = 3.5
        max_lanes = max(1, int(width // lane_width))
        edge_properties.append((u, v, width, max_lanes))

    # 定义状态空间：每条边4种状态，用列表表示，索引对应边在undirected_edges中的顺序
    # 初始状态可以随机生成，或者使用当前图的状态
    current_state = [random.randint(0, 3) for _ in range(len(undirected_edges))]

    # 计算初始状态的得分
    current_G = create_candidate_graph(G_full, undirected_edges, current_state, edge_properties)
    if not check_connectivity_fast(current_G, od_demand):
        # 如果初始状态不连通，重新生成直到连通（或者使用一个保守的初始状态，比如全部双向）
        # 这里为了简单，先尝试全部双向
        current_state = [3] * len(undirected_edges)
        current_G = create_candidate_graph(G_full, undirected_edges, current_state, edge_properties)
        if not check_connectivity_fast(current_G, od_demand):
            # 如果还是不连通，可能需要调整，这里先返回原图
            return G_full

    current_score = evaluate_graph(current_G, weights)
    best_state = current_state[:]
    best_score = current_score
    best_G = current_G

    # 记录基准得分（原图）
    base_score = evaluate_graph(G_full, weights)

    # Step 3: 模拟退火
    temperature = initial_temperature
    for iteration in range(max_iterations):
        # 生成邻域状态：随机改变一条边的状态
        idx = random.randint(0, len(undirected_edges) - 1)
        new_state = current_state[:]
        # 随机选择一个新状态，但不能和当前状态相同
        possible_states = [0, 1, 2, 3]
        possible_states.remove(current_state[idx])
        new_state[idx] = random.choice(possible_states)

        # 创建新图并检查连通性
        new_G = create_candidate_graph(G_full, undirected_edges, new_state, edge_properties)
        if check_connectivity_fast(new_G, od_demand):
            new_score = evaluate_graph(new_G, weights)

            # 接受新状态的概率
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_state = new_state
                current_score = new_score
                current_G = new_G

                if current_score < best_score:
                    best_state = current_state[:]
                    best_score = current_score
                    best_G = current_G

        # 降温
        temperature *= cooling_rate

        # 可以在这里添加提前终止条件，比如温度降到某个阈值

    # Step 4: 返回结果
    if best_score < base_score:
        # 复制最佳图的必要属性
        best_attrs = {'width', 'reversed', 'oneway', 'drive_pass_count',
                      'drive_saturation', 'best_sat_num', 'best_lanes_num', 'edge_length'}
        return copy_graph_with_selected_edge_attrs(best_G, best_attrs)
    else:
        return G_full


def create_candidate_graph(G_full, undirected_edges, states, edge_properties):
    """根据状态列表创建候选图"""
    # 这里使用之前优化的create_optimized_candidate函数，但需要适应状态列表
    # 注意：edge_properties和undirected_edges顺序一致
    G_candidate = nx.DiGraph()
    G_candidate.add_nodes_from(G_full.nodes(data=True))

    edges_to_skip = set(undirected_edges) | {(v, u) for u, v in undirected_edges}

    for u, v, data in G_full.edges(data=True):
        if (u, v) not in edges_to_skip:
            # 只复制必要的属性，这里简化，实际使用你定义的EDGE_ATTRS_TO_KEEP
            EDGE_ATTRS_TO_KEEP = {'lanes', 'width', 'reversed', 'oneway', 'edge_length'}
            filtered_data = {k: v for k, v in data.items() if k in EDGE_ATTRS_TO_KEEP}
            G_candidate.add_edge(u, v, **filtered_data)

    for (u, v, width, max_lanes), state in zip(edge_properties, states):
        if state == 0:
            continue
        elif state == 1:
            G_candidate.add_edge(u, v, lanes=max_lanes, width=width, oneway=1, reversed=0)
        elif state == 2:
            G_candidate.add_edge(v, u, lanes=max_lanes, width=width, oneway=1, reversed=1)
        elif state == 3:
            n1 = max(1, max_lanes // 2)
            n2 = max(1, max_lanes - n1)
            G_candidate.add_edge(u, v, lanes=n1, width=width, oneway=0, reversed=0)
            G_candidate.add_edge(v, u, lanes=n2, width=width, oneway=0, reversed=1)

    return G_candidate


def evaluate_graph(G, weights):
    """计算图的得分，包括交通分配和车道优化"""
    try:
        calculate_traffic(G, od_demand)
        optimize_drive_lanes(G, 'drive_pass_count')
        sat_error = 0
        lanes_sum = 0
        for u, v, data in G.edges(data=True):
            sat_error += abs(data.get('best_sat_num', 0) - 0.8)
            lanes_sum += data.get('best_lanes_num', 0)
        return weights['sat'] * sat_error + weights['lanes'] * lanes_sum
    except Exception as e:
        # 如果分配失败，返回一个很大的分数（惩罚）
        return float('inf')




def calculate_score_fast(G):
    """使用向量化计算得分（如果边数量大）"""
    import numpy as np

    # 收集所有边的数据
    edges_data = []
    for u, v, data in G.edges(data=True):
        sat = data.get('best_sat_num', 0)
        lanes = data.get('best_lanes_num',0)
        edges_data.append((sat, lanes))

    if not edges_data:
        return 0, 0

    # 转换为numpy数组进行向量化计算
    arr = np.array(edges_data)
    sat_error = np.sum(np.abs(arr[:, 0] - 0.8))
    lanes_sum = np.sum(arr[:, 1])

    return sat_error, lanes_sum


from functools import lru_cache
import hashlib


def hash_graph_state(G, undirected_edges):
    """创建图状态的哈希，用于缓存"""
    # 提取与子图相关的边状态
    state_list = []
    for u, v in undirected_edges:
        if G.has_edge(u, v) and G.has_edge(v, u):
            state_list.append(3)  # 双向
        elif G.has_edge(u, v):
            state_list.append(1)  # u→v
        elif G.has_edge(v, u):
            state_list.append(2)  # v→u
        else:
            state_list.append(0)  # 关闭

    # 创建哈希
    state_str = ''.join(map(str, state_list))
    return hashlib.md5(state_str.encode()).hexdigest()


filepath = 'data/normal_digraph_real.pkl'#替换为你的需要优化的图数据路径
with open(filepath, 'rb') as f:
    streets_graph = pickle.load(f)

buildings=gpd.read_file("D:"+r"\first_term\Florence\shortest\data\buildings_function.geojson")#替换为你的建筑数据路径
buildings=buildings.to_crs(epsg=3857)
sub=gpd.read_file("D:"+r"\first_term\Florence\shortest\data\extreme_subgraphs_k2\low_saturation\low_10_road_260.geojson")#替换为你的问题子图数据路径
sub=sub.to_crs(epsg=3857)
subgraph=mp.gdf_to_nx(sub,'primal',length='edge_length',multigraph=False,directed=False)




od_demand=build_od_demand_dict(streets_graph)
G_best= optimize_subgraph_with_simulated_annealing(streets_graph, subgraph)
# calculate_traffic(streets_graph, od_demand)
draw_direction(G_best,buildings,column='best_sat_num',forward_cmp=plt.cm.jet,reserves_cmp=plt.cm.jet,size=(15,15),limit=[0,1])
