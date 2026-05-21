from common.superpotential_parser import *
from common.inconsistents_parser import *
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it


dynkin_diagram = nx.MultiGraph()
algebra = serialize_theory_name(input('Lie algebra to draw: '))

build_dynkin_diagram(dynkin_diagram, algebra[0], algebra[1])

graph_data = from_networkx(dynkin_diagram, group_node_attrs=['short', 'mark', 'comark'])
print('Graph nodes data: ')
print(graph_data.x)
print('Graph edges data: ')
print(graph_data.edge_index)

fig, ax = plt.subplots()
node_pos = nx.planar_layout(dynkin_diagram)
nx.draw_networkx_nodes(
    dynkin_diagram,
    node_pos,
    nodelist=[x for x, y in dynkin_diagram.nodes(data=True) if y['short'] == 0],
    node_color = '#dddddd',
    ax=ax
)
nx.draw_networkx_nodes(
    dynkin_diagram,
    node_pos,
    nodelist=[x for x, y in dynkin_diagram.nodes(data=True) if y['short'] == 1],
    node_color = '#333333',
    ax=ax
)
nx.draw_networkx_edges(
    dynkin_diagram,
    node_pos,
    connectionstyle=[f"arc3,rad={r}" for r in it.accumulate([0.07] * len(dynkin_diagram.edges))],
    ax=ax
)


plt.show()