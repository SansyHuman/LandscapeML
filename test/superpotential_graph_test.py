from common.superpotential_parser import *
from common.inconsistents_parser import *
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it


w_obj = Superpotential()
prev_theory = None

while True:
    theory = input('Theory name: ')
    if theory == 'exit':
        break
    w = input('Superpotential in list format: ')

    if prev_theory != theory:
        w_obj.set_theory(serialize_theory_name(theory))
    w_obj.set_superpotential(w[1:-1].split(','))
    prev_theory = theory

    dynkin_diagram_data = w_obj.get_theory_data()
    print('Dynkin diagram nodes data: ')
    print(dynkin_diagram_data.x)
    print('Dynkin diagram edges data: ')
    print(dynkin_diagram_data.edge_index)
    print()

    superpotential_graph_data = w_obj.get_superpotential_data()
    print('Superpotential graph nodes data: ')
    print(superpotential_graph_data.x)
    print('Superpotential graph edges data: ')
    print(superpotential_graph_data.edge_index)

    fig, ax = plt.subplots(1, 2, squeeze=True)

    node_pos = nx.planar_layout(w_obj.dynkin_diagram)
    nx.draw_networkx_nodes(
        w_obj.dynkin_diagram,
        node_pos,
        nodelist=[x for x, y in w_obj.dynkin_diagram.nodes(data=True) if y['short'] == 0],
        node_color='#dddddd',
        ax=ax[0]
    )
    nx.draw_networkx_nodes(
        w_obj.dynkin_diagram,
        node_pos,
        nodelist=[x for x, y in w_obj.dynkin_diagram.nodes(data=True) if y['short'] == 1],
        node_color='#333333',
        ax=ax[0]
    )
    nx.draw_networkx_edges(
        w_obj.dynkin_diagram,
        node_pos,
        connectionstyle=[f"arc3,rad={r}" for r in it.accumulate([0.07] * len(w_obj.dynkin_diagram.edges))],
        ax=ax[0]
    )

    nx.draw(w_obj.superpotential_graph, with_labels=True, font_weight='bold',
        pos=nx.arf_layout(w_obj.superpotential_graph),
        connectionstyle=[f"arc3,rad={r}" for r in
                         it.accumulate([0.03] * len(w_obj.superpotential_graph.edges))],
        ax=ax[1]
    )

    plt.show()
