import networkx as nx
import matplotlib.pyplot as plt
import momepy as mp
import numpy as np
import geopandas as gpd
import os
import pickle
def calculate_trip_generation(buildings):
    """
    根据建筑功能类型和面积计算产生量和吸引量
    """
    # 定义各功能类型的产生率和吸引率（单位：人次/平方米）
    # 这些值需要根据实际情况调整
    # Florence_population = 85000
    # total_building_area = buildings['total a'].sum()
    # p_rate = (Florence_population / total_building_area)

    generation_rates = {
        'residence': {'generate': 0.002, 'absorb': 0.002},  # 居住
        'office': {'generate': 0.007, 'absorb': 0.007},  # 办公
        'commerce': {'generate': 0.015, 'absorb': 0.015},  # 商业
        'railway station': {'generate': 0.025, 'absorb': 0.035},  # 火车站
        'public transport': {'generate': 0.025, 'absorb': 0.025},  # 公交枢纽
        'tourism': {'generate': 0.025, 'absorb': 0.025},  # 文旅
        'school': {'generate': 0.012, 'absorb': 0.002},  # 教育
        'stadium': {'generate': 0.0001, 'absorb': 0.0001},  # 体育馆
        'park': {'generate': 0.002, 'absorb': 0.002},  # 公园
        'accessory': {'generate': 0.001, 'absorb': 0.001},  # 附属
        'public service': {'generate': 0.015, 'absorb': 0.015},  # 公共服务
        'parking': {'generate': 0.002, 'absorb': 0.002},  # 停车场
        'hotel': {'generate': 0.004, 'absorb': 0.004}  # 酒店
    }

    # 初始化产生量和吸引量列
    buildings['generate'] = 0.0
    buildings['absorb'] = 0.0


    for idx, building in buildings.iterrows():
        # 给建筑设置人口

        # 获取建筑功能类型
        function_type = building['function']  # 假设你的建筑数据中有'function'列
        rate = generation_rates[function_type]

        if function_type in generation_rates:
            # 根据功能类型计算有效面积
            if function_type == 'commerce':
                # 商业：使用首层面积
                gen = building['a'] * rate['generate'] + (building['total a']-building['a'])*generation_rates['residence']['generate']
                abb = building['a']* rate['absorb']+(building['total a']-building['a'])*generation_rates['residence']['absorb']
            elif function_type in ['office', 'public service']:
                if building['a'] < 1000:  # 底面积小于1000平方米
                    # 只算首层，二层以上算住宅
                    gen = building['a'] * rate['generate'] + (building['total a'] - building['a']) * \
                          generation_rates['residence']['generate']
                    abb = building['a'] * rate['absorb'] + (building['total a'] - building['a']) * \
                          generation_rates['residence']['absorb']
                else:
                    # 整个面积都算办公/公共服务
                    gen = building['total a'] * rate['generate']
                    abb = building['total a'] * rate['absorb']
            else:
                # 其他功能类型：使用总面积
                gen = building['total a'] * rate['generate']
                abb = building['total a'] * rate['absorb']


            # 计算产生量和吸引量

            buildings.loc[idx, 'generate']= gen*0.5
            buildings.loc[idx, 'absorb']  = abb*1.5

    return buildings


def calculate_population(buildings, streets,streets_graph):
    buildings = buildings.to_crs(epsg=3857)
    buildings=calculate_trip_generation(buildings)

    # 生成tessellation
    limit = mp.buffered_limit(buildings)
    tessellation = mp.morphological_tessellation(buildings, clip=limit, shrink=1, segment=0.5)
    # 生成地块
    streets = streets.reset_index(drop=True)
    buildings = buildings.reset_index(drop=True)
    snapped = mp.extend_lines(
        streets, tolerance=200, target=tessellation, barrier=buildings
    )
    blocks, tessellation_id = mp.generate_blocks(
        tessellation, streets, buildings
    )

    buildings = buildings.reset_index(drop=True)
    buildings["bID"] = tessellation_id.values
    tessellation["bID"] = tessellation_id.values

    # 建筑连接到最近的街道和点
    # graph = mp.gdf_to_nx(streets)
    nodes, edges = mp.nx_to_gdf(streets_graph)
    if nodes.crs is None and buildings.crs is not None:
        nodes.set_crs(buildings.crs, inplace=True)
        edges.set_crs(buildings.crs, inplace=True)
    #建立序号和坐标的对应关系
    index_to_coord = {idx: (row.geometry.x, row.geometry.y) for idx, row in nodes.iterrows()}

    buildings["edge_index"] = mp.get_nearest_street(buildings, edges)
    buildings["node_index"] = mp.get_nearest_node(buildings, nodes, edges, nearest_edge=buildings["edge_index"])

    node_buildings=buildings.groupby('node_index').agg(total_node_generate=('generate', "sum"),total_node_absorb=('absorb', "sum"))
    #通过映射表找到最对应坐标，建立坐标和面积的字典
    coord_generate_dict = {}
    coord_absorb_dict = {}
    for idx, generate in node_buildings['total_node_generate'].items():
        coord = index_to_coord[idx]
        coord_generate_dict[coord] = generate
    for idx, absorb in node_buildings['total_node_absorb'].items():
        coord = index_to_coord[idx]
        coord_absorb_dict[coord] = absorb

    #把字典用于增加streets_graph的一个字段
    nx.set_node_attributes(streets_graph, coord_generate_dict,'generate')
    nx.set_node_attributes(streets_graph, coord_absorb_dict, 'absorb')
    for node in streets_graph.nodes:
        if 'generate' not in streets_graph.nodes[node]:
            streets_graph.nodes[node]['generate'] = 0
        if 'absorb' not in streets_graph.nodes[node]:
            streets_graph.nodes[node]['absorb'] = 0

    return streets_graph

def set_connect_points(graph):
    # 找到特殊连接点(需要手动设置流量的特殊点的id)，并根据实际调研数据设置产生量和吸引量
    node_id = 0
    for node, data in graph.nodes(data=True):
        data['id'] = node_id
        node_id += 1
    check_list = [293, 3, 6, 15, 296, 255, 166, 183, 131, 292]#研究范围与外界的连接点
    check_entrance_list=[60,59,74,79,62]#球场入口（场地中的重要目的地）
    connect_list = []
    entrance_list = []
    for i in check_list:
        for node, data in graph.nodes(data=True):
            if data['id'] == i:
                connect_list.append(node)
    for i in check_entrance_list:
        for node, data in graph.nodes(data=True):
            if data['id'] == i:
                entrance_list.append(node)
    # 平日
    # graph.nodes[connect_list[0]]['generate'] = 3600
    # graph.nodes[connect_list[1]]['generate'] = 1800
    # graph.nodes[connect_list[2]]['generate'] = 0
    # graph.nodes[connect_list[3]]['generate'] = 900
    # graph.nodes[connect_list[4]]['generate'] = 36
    # graph.nodes[connect_list[5]]['generate'] = 360
    # graph.nodes[connect_list[6]]['generate'] = 120
    # graph.nodes[connect_list[7]]['generate'] = 1440
    # graph.nodes[connect_list[8]]['generate'] = 360
    # graph.nodes[connect_list[9]]['generate'] = 0
    #
    # graph.nodes[connect_list[0]]['absorb'] =3600
    # graph.nodes[connect_list[1]]['absorb'] =0
    # graph.nodes[connect_list[2]]['absorb'] =1800
    # graph.nodes[connect_list[3]]['absorb'] =900
    # graph.nodes[connect_list[4]]['absorb'] =36
    # graph.nodes[connect_list[5]]['absorb'] =1800
    # graph.nodes[connect_list[6]]['absorb'] =120
    # graph.nodes[connect_list[7]]['absorb'] =1440
    # graph.nodes[connect_list[8]]['absorb'] =360
    # graph.nodes[connect_list[9]]['absorb'] =0

    #赛后
    graph.nodes[connect_list[0]]['generate'] = 3600/50
    graph.nodes[connect_list[1]]['generate'] = 1800/50
    graph.nodes[connect_list[2]]['generate'] = 0/50
    graph.nodes[connect_list[3]]['generate'] = 900/50
    graph.nodes[connect_list[4]]['generate'] = 36/50
    graph.nodes[connect_list[5]]['generate'] = 360/50
    graph.nodes[connect_list[6]]['generate'] = 120/50
    graph.nodes[connect_list[7]]['generate'] = 1440/50
    graph.nodes[connect_list[8]]['generate'] = 360/50
    graph.nodes[connect_list[9]]['generate'] = 0

    graph.nodes[connect_list[0]]['absorb'] = 10800
    graph.nodes[connect_list[1]]['absorb'] = 8100
    graph.nodes[connect_list[2]]['absorb'] = 4050
    graph.nodes[connect_list[3]]['absorb'] = 4050
    graph.nodes[connect_list[4]]['absorb'] = 162
    graph.nodes[connect_list[5]]['absorb'] = 1620
    graph.nodes[connect_list[6]]['absorb'] = 3540
    graph.nodes[connect_list[7]]['absorb'] = 6480
    graph.nodes[connect_list[8]]['absorb'] = 7620
    graph.nodes[connect_list[9]]['absorb'] = 0
    #赛后球场
    for i in range(0,len(entrance_list)):
        graph.nodes[entrance_list[i]]['generate'] = 10000
        graph.nodes[entrance_list[i]]['absorb'] = 0


    # #赛前
    # graph.nodes[connect_list[0]]['generate'] = 3600*2.0
    # graph.nodes[connect_list[1]]['generate'] = 1800*2.0
    # graph.nodes[connect_list[2]]['generate'] = 900*2.0
    # graph.nodes[connect_list[3]]['generate'] = 900*2.0
    # graph.nodes[connect_list[4]]['generate'] = 36*2.0
    # graph.nodes[connect_list[5]]['generate'] = 360*2.0
    # graph.nodes[connect_list[6]]['generate'] = 120*2.0
    # graph.nodes[connect_list[7]]['generate'] = 1440*2.0
    # graph.nodes[connect_list[8]]['generate'] = 1440*2.0
    # graph.nodes[connect_list[9]]['generate'] = 0
    #
    # graph.nodes[connect_list[0]]['absorb'] =3600/50
    # graph.nodes[connect_list[1]]['absorb'] =0/50
    # graph.nodes[connect_list[2]]['absorb'] =1800/50
    # graph.nodes[connect_list[3]]['absorb'] =900/50
    # graph.nodes[connect_list[4]]['absorb'] =36/50
    # graph.nodes[connect_list[5]]['absorb'] =1800/50
    # graph.nodes[connect_list[6]]['absorb'] =120/50
    # graph.nodes[connect_list[7]]['absorb'] =1440/50
    # graph.nodes[connect_list[8]]['absorb'] =360/50
    # graph.nodes[connect_list[9]]['absorb'] =0
    #
    # # 赛前球场
    # for i in range(0,len(entrance_list)):
    #     graph.nodes[entrance_list[i]]['generate'] = 0
    #     graph.nodes[entrance_list[i]]['absorb'] = 5000


    return graph



def furness_with_factors(P, A,  max_iter=500, tol=1e-12):
    #调整P A总量相等
    P = P.astype(float)
    A = A.astype(float)
    total_P = P.sum()
    total_A = A.sum()

    if total_P == 0 and total_A == 0:
        raise ValueError("所有 P 和 A 都为 0！")
    elif total_P == 0:
        A[:] = 0.0
    elif total_A == 0:
        P[:] = 0.0
    else:
        # 缩放到平均值，保持相对结构
        avg = (total_P + total_A) / 2
        P *= avg / total_P
        A *= avg / total_A

    OD = np.outer(P, A)   # 初始OD矩阵
    a = np.ones_like(P)  # 累积行平衡因子
    b = np.ones_like(A)  # 累积列平衡因子

    print("开始迭代...\n")

    for itr in range(max_iter):
        # 第1步：行平衡
        row_sum = OD.sum(axis=1)
        row_factor = np.divide(P, row_sum, out=np.ones_like(P), where=row_sum>0)  # 行平衡系数
        OD *= row_factor[:, np.newaxis]  # 调整OD矩阵
        a *= row_factor  # 累积到a_i

        # 第2步：列平衡
        col_sum = OD.sum(axis=0)
        col_factor = np.divide(A,col_sum,out=np.ones_like(A),where=col_sum>0)  # 列平衡系数
        OD *= col_factor[np.newaxis, :]  # 调整OD矩阵
        b *= col_factor  # 累积到b_j

        # 检查误差
        row_error = np.max(np.abs(OD.sum(axis=1) - P))
        col_error = np.max(np.abs(OD.sum(axis=0) - A))

        if itr % 10 == 0:
            print(f"迭代{itr+1}: 行误差={row_error:.2e}, 列误差={col_error:.2e}")
            # print(f"        a={a}, b={b}")

        if max(row_error, col_error) < tol:
            print(f"\n✓ 收敛！迭代{itr+1}次")
            break
    else:
        print(f"\n✗ {max_iter}次未收敛")

    return OD, a, b



gdf = gpd.read_file("D:"+r"\first_term\Florence\traffic_model\completed_street.geojson")#替换为你下载并简化后的边数据路径
buildings=gpd.read_file(r"D:\first_term\Florence\shortest\data\buildings_function.geojson")#替换为你下载的建筑数据路径
gdf=gdf.to_crs(epsg=3857)
buildings=buildings.to_crs(epsg=3857)

#保存车行的图
#把gdf转成有向图
gdf['reversed'] = gdf['reversed'].astype(int)
gdf['oneway'] = gdf['oneway'].astype(int)
streets_graph = nx.DiGraph()

for idx, row in gdf.iterrows():
    geom = row.geometry
    start_node = geom.coords[0]
    end_node = geom.coords[-1]
    edge_attrs = row.drop('geometry').to_dict()
    l = edge_attrs['lanes']
    # 根据oneway字段判断
    if row.get('oneway') == 1:
        if row.get('reversed') == 1:
            streets_graph.add_edge(end_node, start_node,geometry=geom, **edge_attrs)
        elif row.get('reversed') == 0:
            streets_graph.add_edge(start_node, end_node,geometry=geom, **edge_attrs)
    elif row.get('oneway') == 0:
        # 双向道路，添加两条边
        streets_graph.add_edge(start_node, end_node,geometry=geom, **edge_attrs|{'lanes':l/2,'reversed':0})
        streets_graph.add_edge(end_node, start_node,geometry=geom, **edge_attrs|{'lanes':edge_attrs['lanes']-l/2,'reversed':1})



streets_graph=calculate_population(buildings,gdf,streets_graph)
streets_graph=set_connect_points(streets_graph)
# 数据
# nodes = sorted(streets_graph.nodes(data=True),key=lambda n: streets_graph.nodes[n]['nodeID'])
P = np.array([data['generate'] for _,  data in streets_graph.nodes(data=True)])
A = np.array([data['absorb'] for _, data in streets_graph.nodes(data=True)])


# 运行
OD, a, b = furness_with_factors(P, A)
n=0
for node,data in streets_graph.nodes(data=True):
    data['a']=a[n]
    data['b']=b[n]
    n+=1

os.makedirs('data', exist_ok=True)
filepath = 'data/after_competition_digraph_real.pkl'#替换为你想要保存的文件路径

with open(filepath, 'wb') as f:
    pickle.dump(streets_graph, f)

print(f"✓ 有向图已保存至 {filepath}")
print(f"  节点数: {len(streets_graph.nodes())}, 边数: {len(streets_graph.edges())}")

# #保存人行的图
# #把gdf转成无向图
# walk_graph = mp.gdf_to_nx(gdf,approach='primal',length='edge_length',multigraph=False,directed=False)
# for u, v, data in walk_graph.edges(data=True):
#     data['ped']=3
#
# walk_graph=calculate_population(buildings,gdf,walk_graph)
# walk_graph=set_connect_points(walk_graph)
# P = np.array([data['generate'] for _,  data in walk_graph.nodes(data=True)])
# A = np.array([data['absorb'] for _, data in walk_graph.nodes(data=True)])
#
# OD, a, b = furness_with_factors(P, A)
# n=0
# for node,data in walk_graph.nodes(data=True):
#     data['a']=a[n]
#     data['b']=b[n]
#     n+=1
#
# os.makedirs('data', exist_ok=True)
# filepath = 'data/normal_graph_real.pkl'
#
# with open(filepath, 'wb') as f:
#     pickle.dump(walk_graph, f)
#
# print(f"✓ 有向图已保存至 {filepath}")
# print(f"  节点数: {len(walk_graph.nodes())}, 边数: {len(walk_graph.edges())}")