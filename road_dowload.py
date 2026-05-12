import networkx as nx
import osmnx as ox
import geojson
import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd

# #下载路网
P=32.07329, 118.79280
G=ox.graph_from_point(P, dist=3000,network_type="walk")
ox.plot_graph(G)
G=ox.project_graph(G)
G=ox.project_graph(G)

pos = {node: node for node in G.nodes()}
nx.draw(G, pos=pos)
plt.show()

# 2. 转换为 GeoDataFrame（边和节点分别处理）
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
gdf_nodes=gdf_nodes.to_crs(epsg=3857)
gdf_edges=gdf_edges.to_crs(epsg=3857)

# 3. 保存为 GeoJSON
output_dir = "YOUR_OUTPUT_DIRECTORY"  # 替换为你想要保存的目录
gdf_nodes.to_file(f"{output_dir}/nodes.geojson", driver="GeoJSON")
gdf_edges.to_file(f"{output_dir}/edges.geojson", driver="GeoJSON")


#下载建筑
P=43.77894, 11.28472
gdf_buildings=ox.features_from_point(P, dist=1800,tags={'building':True})
if isinstance(gdf_buildings.index, pd.MultiIndex):
    buildings = gdf_buildings.reset_index()

gdf_buildings=gdf_buildings.to_crs(epsg=3857)
gdf_buildings.to_file("YOUR_BUILDING_DIRECTORY/buildings.geojson",driver="GeoJSON")# 替换为你想要保存的目录

buildings = gpd.read_file("YOUR_BUILDING_DIRECTORY/buildings.geojson")
buildings = buildings.to_crs(epsg=3857)
fig,ax = plt.subplots(figsize=(15,15))
buildings.plot(ax=ax,color='grey')
plt.axis('off')
plt.show()