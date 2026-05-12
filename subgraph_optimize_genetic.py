import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from typing import Dict, Tuple, Optional
import networkx as nx
import pickle
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import momepy as mp
import random
from pymoo.core.population import Population
import os
import math
from pymoo.algorithms.moo.nsga2 import NSGA2


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


def check_connectivity_fast(G, od_demand) -> float:
    """计算不可达需求比例"""
    if not od_demand:
        return True

    # total_demand = sum(od_demand.values())
    # if total_demand <= 0:
    #     return True

    unreachable_demand = 0.0
    node_set = set(G.nodes())

    for (o, d), flow in od_demand.items():
        if flow <= 0:
            continue
        if o not in node_set or d not in node_set:
            unreachable_demand += flow
            continue
        try:
            if not nx.has_path(G, o, d):
                unreachable_demand += flow
        except Exception:
            unreachable_demand += flow
    if unreachable_demand-754> 0:
        return unreachable_demand-754
    else:
        return 0


def optimize_bidirectional_lanes(q1, q2, l, width, current_n1, current_n2):
    if None in (q1, q2, l, width, current_n1, current_n2):
        print('wrong edge')
        return 0, 0, 0, 0, 0, 0

    C0 = 1800 * 0.8 * l
    lane_width = 3.5
    target_sat = 0.8
    weights = {'sat': 1.0, 'lanes': 0.01, 'change': 0.01}

    N_max = max(0, int(width // lane_width))

    best_score = float('inf')
    best_n1, best_n2 = 1, 1
    best_s1, best_s2 = 0.0, 0.0
    ch1, ch2 = 0.0, 0.0

    if N_max <= 1:
        equiv_C = 0.5 * C0
        s1 = q1 / equiv_C if equiv_C > 0 else (float('inf') if q1 > 0 else 0)
        s2 = q2 / equiv_C if equiv_C > 0 else (float('inf') if q2 > 0 else 0)
        return 0.5, 0.5, s1, s2, ch1, ch2
    else:
        found = False
        for total in range(2, N_max + 1):
            for n1 in range(1, total):
                n2 = total - n1
                if n2 < 1:
                    continue

                s1 = q1 / (n1 * C0) if n1 > 0 else float('inf')
                s2 = q2 / (n2 * C0) if n2 > 0 else float('inf')

                sat_error = abs(s1 - target_sat) + abs(s2 - target_sat)
                total_lanes_used = n1 + n2
                change = abs(n1 - current_n1) + abs(n2 - current_n2)

                score = (weights['sat'] * sat_error +
                         weights['lanes'] * total_lanes_used +
                         weights['change'] * change)

                if score < best_score:
                    best_score = score
                    best_n1, best_n2 = n1, n2
                    best_s1, best_s2 = s1, s2
                    ch1 = n1 - current_n1
                    ch2 = n2 - current_n2
                    found = True

        if not found:
            n1, n2 = current_n1, current_n2
            s1 = q1 / (n1 * C0)
            s2 = q2 / (n2 * C0)
            ch1 = 0.0
            ch2 = 0.0
            return n1, n2, s1, s2, ch1, ch2

        return best_n1, best_n2, best_s1, best_s2, ch1, ch2


def optimize_directional_lanes(q, l, width, current_n):
    if None in (q, l, width, current_n):
        print('wrong edge')
        return 0, 0, 0

    C0 = 1800 * 0.8 * l
    lane_width = 3.5
    target_sat = 0.8
    weights = {'sat': 1.0, 'lanes': 0.01, 'change': 0.01}

    N_max = max(0, int(width // lane_width))

    best_score = float('inf')
    best_n = 1
    best_s = 0.0
    ch = 0.0
    found = False

    for n in range(1, N_max + 1):
        s = q / (n * C0)
        sat_error = abs(s - target_sat)
        change = abs(n - current_n)
        score = (weights['sat'] * sat_error +
                 weights['lanes'] * n +
                 weights['change'] * change)

        if score < best_score:
            best_score = score
            best_n = n
            best_s = s
            ch = n - current_n
            found = True

    if not found:
        n = current_n
        s = q / (n * C0)
        ch = 0.0
        return n, s, ch

    return best_n, best_s, ch


def optimize_drive_lanes(streets_graph, pass_count):
    for u, v, data in streets_graph.edges(data=True):
        l = 1
        if data['oneway'] == 1:
            best_n, best_s, ch = optimize_directional_lanes(
                data[pass_count], l, data['width'], data['lanes'])
            data['best_lanes_num'] = best_n
            data['best_sat_num'] = best_s
            data['change_num'] = ch
        elif data['oneway'] == 0:
            if data['reversed'] == 0:
                q1 = data[pass_count]
                q2 = streets_graph.edges[v, u][pass_count]
                current_n1 = data['lanes']
                current_n2 = streets_graph.edges[v, u]['lanes']
                best_n1, best_n2, best_s1, best_s2, ch1, ch2 = (
                    optimize_bidirectional_lanes(
                        q1, q2, l, data['width'], current_n1, current_n2))
                data['best_lanes_num'] = best_n1
                streets_graph.edges[v, u]['best_lanes_num'] = best_n2
                data['best_sat_num'] = best_s1
                streets_graph.edges[v, u]['best_sat_num'] = best_s2
                data['change_num'] = ch1
                streets_graph.edges[v, u]['change_num'] = ch2
            else:
                continue


def copy_graph_with_selected_edge_attrs(G, edge_attrs_to_keep):
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        filtered_data = {key: data[key] for key in edge_attrs_to_keep if key in data}
        H.add_edge(u, v, **filtered_data)

    return H


def calculate_score_fast(G):
    edges_data = []

    for u, v, data in G.edges(data=True):
        sat = data.get('best_sat_num', 0)
        lanes = data.get('best_lanes_num', 0)
        edges_data.append((sat, lanes))

    if not edges_data:
        return 0, 0

    arr = np.array(edges_data)
    sat_vals=arr[:,0]
    sat_error = np.sum(np.abs(sat_vals - 0.5))
    mask = sat_vals > 0.7
    if np.any(mask):
        penalty = np.sum((sat_vals[mask] - 0.7)*10)
    else:
        penalty = 0.0
    lanes_sum = np.sum(arr[:, 1])


    return sat_error, lanes_sum,penalty


# 遗传算法优化部分
class RoadNetworkProblem(Problem):
    """路网优化问题类，使用pymoo的遗传算法"""

    def __init__(self, G_full, subgraph, od_demand, n_edges):
        """
        初始化问题

        参数:
        G_full: 完整路网图
        subgraph: 需要优化的子图（无向图）
        od_demand: OD需求矩阵
        n_edges: 子图中边的数量
        """
        self.G_full = G_full
        self.subgraph = subgraph
        self.od_demand = od_demand
        self.undirected_edges = list(subgraph.edges())
        self.n_edges = n_edges
        # 生成多种启发式状态
        self.heuristic_states = self._generate_heuristic_states()

        # 预计算每条边的属性
        self.edge_properties = []
        for u, v in self.undirected_edges:
            width = self.subgraph[u][v]['width']
            lane_width = 3.5
            max_lanes = max(1, int(width // lane_width))
            edge_length=subgraph[u][v]['edge_length']
            geom = subgraph[u][v]['geometry']
            parking=subgraph[u][v]['parking']
            self.edge_properties.append((u, v, width, max_lanes, edge_length,geom,parking))

        # 设置权重
        self.weights = {'sat': 1.0, 'lanes': 0.05,'penalty':2.0,'distance':0.05}
        self.penalty_weight = 5.0

        # 变量维度：每条边有4种状态 (0:关闭, 1:u→v, 2:v→u, 3:双向)
        # 目标函数：1个（总得分）
        # 约束条件：1个（连通性约束）
        super().__init__(
            n_var=n_edges,
            n_obj=1,
            n_constr=0,
            xl=0,  # 变量下界
            xu=3,  # 变量上界
            vtype=int  # 整数变量
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """评估种群中的每个个体"""
        #X代表整个一代中种群中所有个体的所有决策变量
        n_population = X.shape[0]
        F = np.zeros((n_population, 1))  # 目标函数值
        # G = np.zeros((n_population, 1))  # 约束违反值

        for i in range(n_population):
            # 获取当前个体的编码（边的状态列表）
            state_vector = X[i, :]

            # 创建候选图
            candidate_graph = self.create_candidate_graph(state_vector)

            # 检查连通性
            unreachable_demand = check_connectivity_fast(candidate_graph, self.od_demand)

            # 计算交通分配和车道优化
            try:
                distance_factor=calculate_traffic(candidate_graph, self.od_demand)
                optimize_drive_lanes(candidate_graph, 'drive_pass_count')

                # 计算目标函数值
                sat_error, lanes_sum,penalty = calculate_score_fast(candidate_graph)
                score = (self.weights['sat'] * sat_error +
                         self.weights['lanes'] * lanes_sum+self.weights['penalty']* penalty+self.weights['distance']*distance_factor)

                unreach_penalty=self.penalty_weight * unreachable_demand
                #流量车道得分数+惩罚数
                F[i, 0] = score+unreach_penalty

            except Exception as e:
                F[i, 0] = 1e6

        out["F"] = F

    def create_candidate_graph(self, state_vector):
        """根据状态向量创建候选图"""
        G_candidate = nx.DiGraph()
        G_candidate.add_nodes_from(self.G_full.nodes(data=True))

        # 需要跳过的边（在子图中的边）
        edges_to_skip = set(self.undirected_edges) | {(v, u) for u, v in self.undirected_edges}

        # 复制不在子图中的边（保持原状）
        for u, v, data in self.G_full.edges(data=True):
            if (u, v) not in edges_to_skip:
                EDGE_ATTRS_TO_KEEP = {'geometry','lanes', 'width', 'reversed', 'oneway', 'edge_length','parking'}
                filtered_data = {k: v for k, v in data.items() if k in EDGE_ATTRS_TO_KEEP}
                G_candidate.add_edge(u, v, **filtered_data)

        # 根据状态向量添加子图中的边
        for (edge_info, state) in zip(self.edge_properties, state_vector):
            u, v, width, max_lanes,edge_length,geom,parking = edge_info

            if state == 0:
                # 关闭该边
                continue
            elif state == 1:
                # u→v 单向
                G_candidate.add_edge(u, v, geometry=geom,lanes=max_lanes, width=width,
                                     oneway=1, reversed=0, edge_length=edge_length,parking=parking)
            elif state == 2:
                # v→u 单向
                G_candidate.add_edge(v, u,geometry=geom, lanes=max_lanes, width=width,
                                     oneway=1, reversed=1, edge_length=edge_length,parking=parking)
            elif state == 3:
                # 双向
                n1 = max(1, max_lanes // 2)
                n2 = max(1, max_lanes - n1)
                G_candidate.add_edge(u, v,geometry=geom, lanes=n1, width=width,
                                     oneway=0, reversed=0, edge_length=edge_length,parking=parking)
                G_candidate.add_edge(v, u,geometry=geom, lanes=n2, width=width,
                                     oneway=0, reversed=1, edge_length=edge_length,parking=parking)

        return G_candidate

    def _generate_heuristic_states(self):
        """生成多种不同的启发式策略状态"""

        states = []
        n = self.n_edges

        # 1. 全双向（最连通）
        states.append([3] * n)

        # 2. 基于原图状态
        state2 = []
        for u, v in self.undirected_edges:
            if self.G_full.has_edge(u, v) and self.G_full.has_edge(v, u):
                state2.append(3)
            elif self.G_full.has_edge(u, v):
                state2.append(1)
            elif self.G_full.has_edge(v, u):
                state2.append(2)
            else:
                state2.append(0)
        states.append(state2)

        # 5. 随机但有一定模式的配置
        for pattern_name in ['sparse', 'dense', 'mixed']:
            state = []
            for i in range(n):
                if pattern_name == 'sparse':
                    # 稀疏：20%双向，80%关闭
                    state.append(3 if random.random() < 0.2 else 0)
                elif pattern_name == 'dense':
                    # 密集：80%双向，20%单向
                    r = random.random()
                    if r < 0.8:
                        state.append(3)
                    else:
                        state.append(1 if random.random() < 0.5 else 2)
                else:  # mixed
                    # 混合：均等概率
                    state.append(random.randint(0, 3))
            states.append(state)

        return states

    def create_diverse_initial_population(self, pop_size):
        """创建具有多样性的初始种群"""
        population = []

        # 添加所有启发式状态（去重）
        seen = set()
        for state in self.heuristic_states:
            state_tuple = tuple(state)
            if state_tuple not in seen:
                seen.add(state_tuple)
                population.append(state)
                if len(population) >= pop_size // 2:  # 启发式占一半
                    break

        # 使用拉丁超立方采样生成其余个体
        remaining = pop_size - len(population)
        if remaining > 0:
            # 使用均匀分布的随机整数
            random_states = np.random.randint(0, 4, size=(remaining, self.n_var))
            for state in random_states:
                population.append(state.tolist())

        return np.array(population)


def optimize_subgraph_with_genetic_algorithm(
        G_full: nx.DiGraph,
        subgraph: nx.Graph,
        od_demand: Dict[Tuple[int, int], float],
        population_size: int = 50,
        n_generations: int = 100,
        save_interval: int = 20,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        seed: int = 42
) -> list[nx.DiGraph]:
    """
    使用遗传算法优化子图

    参数:
    G_full: 完整路网图
    subgraph: 需要优化的子图（无向图）
    od_demand: OD需求矩阵
    population_size: 种群大小
    n_generations: 进化代数
    crossover_prob: 交叉概率
    mutation_prob: 变异概率
    seed: 随机种子

    返回:
    优化后的图
    """
    # 获取子图边数
    undirected_edges = list(subgraph.edges())
    n_edges = len(undirected_edges)

    if n_edges == 0:
        print("子图没有边，无需优化")
        return G_full

    print(f"优化问题：{n_edges}条边，4种状态（0-3）")

    # 创建优化问题实例
    problem = RoadNetworkProblem(G_full, subgraph, od_demand, n_edges)

    # 创建多样化的初始种群
    initial_X = problem.create_diverse_initial_population(population_size)
    print(f"初始种群多样性：{len(set([tuple(x) for x in initial_X]))} 个不同个体")
    initial_population = Population.new("X", initial_X)

    # 设置遗传算法参数
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=initial_population,  # 使用自定义的初始种群
        crossover=SBX(prob=crossover_prob, eta=15, vtype=int),
        mutation=PM(prob=mutation_prob, eta=20, vtype=int),
        eliminate_duplicates=True
    )

    # 设置终止条件
    termination = get_termination("n_gen", n_generations)

    # 运行优化
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
        save_history=True,
        parallelization = ('processes', 8)
    )
    base_graph = G_full.copy()
    base_distance_factor=calculate_traffic(base_graph, od_demand)
    optimize_drive_lanes(base_graph, 'drive_pass_count')
    base_sat_error, base_lanes_sum, base_penalty = calculate_score_fast(base_graph)
    unreachable = check_connectivity_fast(base_graph, od_demand)
    base_score = (problem.weights['sat'] * base_sat_error +
             problem.weights['lanes'] * base_lanes_sum + problem.weights['penalty'] * base_penalty+ problem.weights[
                 'distance'] * base_distance_factor+problem.penalty_weight*unreachable)

    print(f"原始图得分: {base_score}")

    # 收集每save_interval代的最优解图
    best_graphs = []

    for gen in range(save_interval, n_generations + 1, save_interval):
        if gen > len(res.history):
            break

        if gen == 0:
            population = initial_population
        else:
            algorithm_state = res.history[gen - 1]
            population = algorithm_state.pop

        # 获取最优解
        F_values = population.get("F")
        X_values = population.get("X")

        if F_values.shape[1] == 1:
            best_idx = np.argmin(F_values.flatten())
        else:
            best_idx = 0

        best_solution = X_values[best_idx]

        # 创建图
        cand_graph = problem.create_candidate_graph(best_solution.astype(int))
        distance_factor=calculate_traffic(cand_graph, od_demand)
        optimize_drive_lanes(cand_graph, 'drive_pass_count')

        # 添加到结果列表
        # cand_graph.graph['generation'] = gen
        score = float(F_values[best_idx][0]) if F_values.shape[1] == 1 else F_values[best_idx]
        print(f"第{gen}代最优解得分: {score}")
        if score < base_score:
            best_graphs.append(cand_graph)
        else:
            print(f"第{gen}代未找到更优解")
            best_graphs.append(cand_graph)

    return best_graphs

    # 获取前五名解
    if len(res.X) > 0:
        # 获取最后一代种群
        last_pop = res.algorithm.pop
        # 根据目标函数值排序（因为是最小化问题，所以值越小越好）
        sorted_indices = np.argsort(last_pop.get("F").flatten())
        top5_indices = sorted_indices[:5]
        top5_solutions = last_pop.get("X")[top5_indices]
        top5_scores = last_pop.get("F")[top5_indices]

        # 计算基准图
        base_graph = G_full.copy()
        calculate_traffic(base_graph, od_demand)
        optimize_drive_lanes(base_graph, 'drive_pass_count')
        base_sat_error, base_lanes_sum, base_penalty = calculate_score_fast(base_graph)
        base_score = problem.weights['sat'] * base_sat_error + problem.weights['lanes'] * base_lanes_sum+problem.weights['penalty']*base_penalty

        print(f"原始图得分: {base_score}")

        top5_graphs = []
        for i, (state, score) in enumerate(zip(top5_solutions, top5_scores)):
            state = state.astype(int)
            print(f"第{i+1}名解得分: {score}")

            # 创建候选图
            cand_graph = problem.create_candidate_graph(state)
            calculate_traffic(cand_graph, od_demand)
            optimize_drive_lanes(cand_graph, 'drive_pass_count')

            # 如果优化后的图更好，则使用优化后的图，否则使用基准图
            if score < base_score:
                best_attrs = {'geometry','width', 'reversed', 'oneway', 'drive_pass_count',
                              'drive_saturation', 'best_sat_num', 'best_lanes_num', 'edge_length', 'lanes','parking','change_num'}
                graph = copy_graph_with_selected_edge_attrs(cand_graph, best_attrs)
            else:
                print(f"第{i+1}名未找到更优解，返回原图")
                graph = base_graph.copy()

            top5_graphs.append(graph)

        # 返回前五名图
        return top5_graphs

    # 获取最优解
    # if len(res.X) > 0:
    #     best_state = res.X.astype(int)
    #     print(f"最优解得分: {res.F[0]}")
    #
    #     # 创建最优图
    #     best_graph = problem.create_candidate_graph(best_state)
    #
    #     # 计算最优图的交通分配和车道优化
    #     calculate_traffic(best_graph, od_demand)
    #     optimize_drive_lanes(best_graph, 'drive_pass_count')
    #
    #     # 计算基准得分（原始图）
    #     base_graph = G_full.copy()
    #     calculate_traffic(base_graph, od_demand)
    #     optimize_drive_lanes(base_graph, 'drive_pass_count')
    #     base_sat_error, base_lanes_sum,base_penalty = calculate_score_fast(base_graph)
    #     base_score = problem.weights['sat'] * base_sat_error + problem.weights['lanes'] * base_lanes_sum+problem.weights['penalty']*base_penalty
    #
    #     print(f"原始图得分: {base_score}")
    #     print(f"优化提升: {base_score - res.F[0]:.2f}")
    #
    #     # 如果优化后的图更好，返回优化后的图
    #     if res.F[0] < base_score:
    #         # 复制最佳图的必要属性
    #         best_attrs = {'geometry','width', 'reversed', 'oneway', 'drive_pass_count',
    #                       'drive_saturation', 'best_sat_num', 'best_lanes_num', 'edge_length', 'lanes','parking','change_num'}
    #         return copy_graph_with_selected_edge_attrs(best_graph, best_attrs)
    #     else:
    #         print("优化未能提升性能，返回原始图")
    #         return base_graph
    # else:
    #     print("未找到可行解，返回原始图")
    #     return G_full


# 主程序
if __name__ == "__main__":
    filepath = r"D:\first_term\Florence\shortest\data\before_competition_digraph_real.pkl"#替换为你的图数据路径
    with open(filepath, 'rb') as f:
        streets_graph = pickle.load(f)

    buildings = gpd.read_file(r"D:\first_term\Florence\shortest\data\buildings_function.geojson")#替换为你的建筑数据路径
    buildings = buildings.to_crs(epsg=3857)

    # 加载子图数据
    sub = gpd.read_file(
        r"D:\first_term\Florence\shortest\data\before_competition_optimized_1\merged_all_subgraphs301.geojson")#替换为你的问题子图数据路径
    sub = sub.to_crs(epsg=3857)
    subgraph = mp.gdf_to_nx(sub, 'primal', length='edge_length', multigraph=False, directed=False)


    # 构建OD需求
    od_demand = build_od_demand_dict(streets_graph)

    n=100
    save_interval=10
    # 使用遗传算法优化
    best_graphs = optimize_subgraph_with_genetic_algorithm(
        streets_graph,
        subgraph,
        od_demand,
        save_interval=save_interval,
        population_size=64,
        n_generations=n
    )
    print("优化完成!")


    for i in range(int(n/save_interval)):
        filepath = fr"D:\first_term\Florence\shortest\data\before_competition_optimized_1\before_competition_optimized_{i}.pkl"#替换为你想要保存的文件路径
        with open(filepath, 'wb') as f:
            pickle.dump(best_graphs[i], f)

    print(f"✓ 优化后图已保存")

    # 绘制结果
    draw_direction(best_graphs[int(n/save_interval)-1], buildings, column='best_sat_num',
                   forward_cmp=plt.cm.jet, reserves_cmp=plt.cm.jet,
                   size=(15,15), limit=[0,1])

