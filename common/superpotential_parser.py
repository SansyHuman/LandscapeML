from typing import Union

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

import networkx as nx
import numpy as np

from common.inconsistents_parser import serialize_theory_name
from common.utils import find_number


def serialize_w_terms(w: str):
    """
    Serialize a superpotential string.
    :param w: Raw superpotential list string.
    :return: Serialized superpotential.
    """
    w_terms = w[1:-1].split(',')
    return serialize_w_term_list(w_terms)


def serialize_w_term_list(w: list[str]):
    """
    Serialize a list of superpotential string.
    :param w: List of superpotential term strings.
    :return: Serialized superpotential.
    """
    w_serial = []
    for term in w:
        if term.strip() == '':
            continue

        term_data = []
        ops = term.strip().split('*')
        for op in ops:
            op_exp = op.strip().split('^')
            index_start, index_len = find_number(op_exp[0])

            op_letter = op_exp[0][:index_start].strip()
            index = int(op_exp[0][index_start:index_start + index_len])
            exponent = 1
            if len(op_exp) == 2:
                exponent = int(op_exp[1].strip())

            term_data.append((op_letter, index, exponent))

        w_serial.append(term_data)

    return w_serial


def get_ade_class(theory) -> tuple[int, int]:
    ade_class = theory[0]
    rank = theory[1]

    # equivalent algebra check
    if (ade_class == 2 and rank == 1) or (ade_class == 3 and rank == 1):
        # A1=B1=C1
        ade_class = 1
        rank = 1
    elif ade_class == 3 and rank == 2:
        # B2=C2
        ade_class = 2
        rank = 2
    elif ade_class == 4 and rank == 3:
        # A3=D3
        ade_class = 1
        rank = 3

    return ade_class, rank


def is_same_gauge_group(theory1: str, theory2: str) -> bool:
    t1 = serialize_theory_name(theory1)
    t2 = serialize_theory_name(theory2)

    ade1, rank1 = get_ade_class(t1)
    ade2, rank2 = get_ade_class(t2)

    return ade1 == ade2 and rank1 == rank2


def is_equivalent_matter_contents(theory1: str, theory2: str) -> bool:
    # includes the case where one matter contents contains other matter contents
    if not is_same_gauge_group(theory1, theory2):
        return False

    t1 = serialize_theory_name(theory1)
    t2 = serialize_theory_name(theory2)

    dim1 = calculate_rep_dimensions(t1)
    dim2 = calculate_rep_dimensions(t2)
    max_dim = max(max(dim1), max(dim2))

    matter1 = np.zeros(max_dim)
    matter2 = np.zeros(max_dim)
    for i in range(len(dim1)):
        matter1[dim1[i] - 1] += t1[i + 2]
        matter2[dim2[i] - 1] += t2[i + 2]
    matter_diff = np.sign(matter1 - matter2)

    equivalent = True
    equiv_sign = 0

    for i in range(len(matter_diff)):
        if matter_diff[i] != 0:
            if equiv_sign == 0:
                equiv_sign = matter_diff[i]
                continue
            if equiv_sign != matter_diff[i]:
                equivalent = False
                break

    return equivalent


def build_dynkin_diagram(graph: nx.MultiGraph, ade_class: int, rank: int) -> None:
    """
    Build a dynkin diagram.
    :param graph: Graph to build dynkin diagram. The graph is cleared before build.
    :param ade_class: ADE class of the algebra. A=1, B=2, C=3, D=4, E=5, F=6, G=7.
    :param rank: Rank of the algebra.
    :return:
    """
    # Dynkin diagram attributes
    # [simple_root, short, mark, comark]
    # simple_root: index of the simple root
    # short: 0 if long root, 1 if short root
    # mark, comark: mark and comark of the simple root

    # equivalent algebra check
    if (ade_class == 2 and rank == 1) or (ade_class == 3 and rank == 1):
        # A1=B1=C1
        ade_class = 1
        rank = 1
    elif ade_class == 3 and rank == 2:
        # B2=C2
        ade_class = 2
        rank = 2
    elif ade_class == 4 and rank == 3:
        # A3=D3
        ade_class = 1
        rank = 3

    if ade_class == 1:  # An
        graph.add_nodes_from(
            [(i, {"simple_root": i, "short": 0, "mark": 1, "comark": 1})
             for i in range(1, rank + 1)]
        )
        graph.add_edges_from(
            [(i, i + 1) for i in range(1, rank)]
        )
    elif ade_class == 2:  # Bn
        assert rank >= 2
        if rank == 2:
            graph.add_nodes_from(
                [
                    (1, {"simple_root": 1, "short": 0, "mark": 1, "comark": 1}),
                    (2, {"simple_root": 2, "short": 1, "mark": 2, "comark": 1})
                ]
            )
            graph.add_edge(1, 2)
            graph.add_edge(1, 2)
        else:
            graph.add_nodes_from(
                [(1, {"simple_root": 1, "short": 0, "mark": 1, "comark": 1})]
                + [
                    (i, {"simple_root": i, "short": 0, "mark": 2, "comark": 2})
                    for i in range(2, rank)
                ]
                + [(rank, {"simple_root": rank, "short": 1, "mark": 2, "comark": 1})]
            )
            graph.add_edges_from(
                [(i, i + 1) for i in range(1, rank)]
            )
            graph.add_edge(rank - 1, rank)
    elif ade_class == 3:  # Cn
        assert rank >= 3
        graph.add_nodes_from(
            [
                (i, {"simple_root": i, "short": 1, "mark": 2, "comark": 1})
                for i in range(1, rank)
            ]
            + [(rank, {"simple_root": rank, "short": 0, "mark": 1, "comark": 1})]
        )
        graph.add_edges_from(
            [(i, i + 1) for i in range(1, rank - 1)]
        )
        graph.add_edge(rank - 1, rank)
        graph.add_edge(rank - 1, rank)
    elif ade_class == 4:  # Dn
        assert rank == 2 or rank >= 4
        if rank == 2:  # D2=A1 x A1
            graph.add_nodes_from(
                [(i, {"simple_root": i, "short": 0, "mark": 1, "comark": 1})
                 for i in range(1, 3)]
            )
        else:
            graph.add_nodes_from(
                [(1, {"simple_root": 1, "short": 0, "mark": 1, "comark": 1})]
                + [
                    (i, {"simple_root": i, "short": 0, "mark": 2, "comark": 2})
                    for i in range(2, rank - 1)
                ]
                + [
                    (rank - 1, {"simple_root": rank - 1, "short": 0, "mark": 1, "comark": 1}),
                    (rank, {"simple_root": rank, "short": 0, "mark": 1, "comark": 1})
                ]
            )
            graph.add_edges_from(
                [(i, i + 1) for i in range(1, rank - 1)]
            )
            graph.add_edge(rank - 2, rank)
    elif ade_class == 5:  # En
        assert 6 <= rank <= 8
        if rank == 6:
            graph.add_nodes_from(
                [
                    (1, {"simple_root": 1, "short": 0, "mark": 1, "comark": 1}),
                    (2, {"simple_root": 2, "short": 0, "mark": 2, "comark": 2}),
                    (3, {"simple_root": 3, "short": 0, "mark": 3, "comark": 3}),
                    (4, {"simple_root": 4, "short": 0, "mark": 2, "comark": 2}),
                    (5, {"simple_root": 5, "short": 0, "mark": 1, "comark": 1}),
                    (6, {"simple_root": 6, "short": 0, "mark": 2, "comark": 2})
                ]
            )
            graph.add_edges_from(
                [(1, 2), (2, 3), (3, 4), (4, 5), (3, 6)]
            )
        elif rank == 7:
            graph.add_nodes_from(
                [
                    (1, {"simple_root": 1, "short": 0, "mark": 2, "comark": 2}),
                    (2, {"simple_root": 2, "short": 0, "mark": 3, "comark": 3}),
                    (3, {"simple_root": 3, "short": 0, "mark": 4, "comark": 4}),
                    (4, {"simple_root": 4, "short": 0, "mark": 3, "comark": 3}),
                    (5, {"simple_root": 5, "short": 0, "mark": 2, "comark": 2}),
                    (6, {"simple_root": 6, "short": 0, "mark": 1, "comark": 1}),
                    (7, {"simple_root": 7, "short": 0, "mark": 2, "comark": 2})
                ]
            )
            graph.add_edges_from(
                [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7)]
            )
        else:
            graph.add_nodes_from(
                [
                    (1, {"simple_root": 1, "short": 0, "mark": 2, "comark": 2}),
                    (2, {"simple_root": 2, "short": 0, "mark": 3, "comark": 3}),
                    (3, {"simple_root": 3, "short": 0, "mark": 4, "comark": 4}),
                    (4, {"simple_root": 4, "short": 0, "mark": 5, "comark": 5}),
                    (5, {"simple_root": 5, "short": 0, "mark": 6, "comark": 6}),
                    (6, {"simple_root": 6, "short": 0, "mark": 4, "comark": 4}),
                    (7, {"simple_root": 7, "short": 0, "mark": 2, "comark": 2}),
                    (8, {"simple_root": 8, "short": 0, "mark": 3, "comark": 3})
                ]
            )
            graph.add_edges_from(
                [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (5, 8)]
            )
    elif ade_class == 6:  # Fn
        assert rank == 4
        graph.add_nodes_from(
            [
                (1, {"simple_root": 1, "short": 0, "mark": 2, "comark": 2}),
                (2, {"simple_root": 2, "short": 0, "mark": 3, "comark": 3}),
                (3, {"simple_root": 3, "short": 1, "mark": 4, "comark": 2}),
                (4, {"simple_root": 4, "short": 1, "mark": 2, "comark": 1})
            ]
        )
        graph.add_edges_from(
            [(1, 2), (2, 3), (2, 3), (3, 4)]
        )
    elif ade_class == 7:  # Gn
        assert rank == 2
        graph.add_nodes_from(
            [
                (1, {"simple_root": 1, "short": 0, "mark": 2, "comark": 2}),
                (2, {"simple_root": 2, "short": 1, "mark": 3, "comark": 1})
            ]
        )
        graph.add_edges_from(
            [(1, 2), (1, 2), (1, 2)]
        )
    else:
        assert False


def calculate_rep_dimensions(theory: tuple[int, ...]) -> tuple[int, ...]:
    """
    Calculates the dimensions of each representations of the theory.
    :param theory: Serialized theory name.
    :return: Tuple of dimension of fundamental, antifundamental, adjoint, rank-2 symmetric tensor,
    conjugate of rank-2 symmetric tensor, rank-2 antisymmetric tensor, and conjugate of rank-2 antisymmetric tensor.
    """
    ade_class = theory[0]
    rank = theory[1]

    fund = 0
    adj = 0
    symm = 0
    antisymm = 0

    if ade_class == 1:
        # SU(n + 1)=An, n + 1=N
        # fundamental = N
        # adjoint = N^2 - 1
        # rank-2 symmetric = N(N + 1) / 2
        # rank-2 antisymmetric = N(N - 1) / 2
        N = rank + 1
        fund = N
        adj = N**2 - 1
        symm = N * (N + 1) // 2
        antisymm = N * (N - 1) // 2
    elif ade_class == 2:
        # SO(2n + 1)=Bn, 2n + 1=N
        # fundamental = N
        # rank-2 symmetric = N(N + 1) / 2 - 1
        # rank-2 antisymmetric = adjoint = N(N - 1) / 2
        N = 2 * rank + 1
        fund = N
        symm = N * (N + 1) // 2 - 1
        antisymm = N * (N - 1) // 2
        adj = antisymm
    elif ade_class == 3:
        # Sp(n)=Usp(2n)=Cn, N=n
        # fundamental = 2N
        # rank-2 symmetric = adjoint = 2N(2N + 1) / 2
        # rank-2 antisymmetric = 2N(2N - 1) / 2 - 1
        N = rank
        fund = 2 * N
        symm = 2 * N * (2 * N + 1) // 2
        antisymm = 2 * N * (2 * N - 1) // 2 - 1
        adj = symm
    elif ade_class == 4:
        # SO(2n)=Dn, N=2n
        # fundamental = N
        # rank-2 symmetric = N(N + 1) / 2 - 1
        # rank-2 antisymmetric = adjoint = N(N - 1) / 2
        N = 2 * rank
        fund = N
        symm = N * (N + 1) // 2 - 1
        antisymm = N * (N - 1) // 2
        adj = antisymm
    elif ade_class == 7: # G2
        fund = 7
        adj = 14
        symm = 27
        antisymm = 14
    return fund, fund, adj, symm, symm, antisymm, antisymm


matter_fields = ['q', 'qb', 'phi', 'S', 'Sb', 'A', 'Ab']


class Superpotential:
    def __init__(self, theory: str=None, superpotential: str=None):
        """
        Create superpotential object.
        :param theory: Theory of the superpotential.
        :param superpotential: Raw superpotential string.
        """
        self.theory = None if theory is None else serialize_theory_name(theory)
        self.superpotential = None if superpotential is None else serialize_w_terms(superpotential)

        if theory is not None and superpotential is not None:
            self.__build_graph()
        else:
            self.dynkin_diagram = nx.MultiGraph()
            self.superpotential_graph = nx.MultiGraph()

    def __build_graph(self) -> None:
        self.dynkin_diagram = nx.MultiGraph()
        self.superpotential_graph = nx.MultiGraph()

        ade_class = self.theory[0]
        rank = self.theory[1]
        build_dynkin_diagram(self.dynkin_diagram, ade_class, rank)
        self.__build_superpotential_graph()

    def __build_superpotential_graph(self) -> None:
        # superpotential attributes
        # [node_type, conj, dim, index]
        # node_type
        # 0: peripheral node connected to central nodes
        # 1: central node which represents matter fields
        # 2: central node which represents flipping fields
        # 3: central node which represents terms in superpotential
        # conj
        # for peripheral nodes it is always 0
        # for matter fields 1 if it is a conjugate representation(antifund, symm_conj, antisymm_conj), else 0.
        # for flipping fields 0: M, 1: X
        # for terms it is always 0
        # dim is the dimension of the representation for peripheral nodes, otherwise 0.
        # index is the index of the fields of peripheral nodes connected to central field nodes starting from 1
        # for non-peripheral nodes and peripheral terms index is always 0

        M_num = 0
        X_num = 0
        for term in self.superpotential:
            for op, index, _ in term:
                if op == 'M' and index > M_num:
                    M_num = index
                elif op == 'X' and index > X_num:
                    X_num = index

        term_num = len(self.superpotential)

        rep_dims = calculate_rep_dimensions(self.theory)
        for i in range(len(matter_fields)):
            field = matter_fields[i]
            field_num = self.theory[i + 2]
            if field_num > 0:
                # For SU2(A1), fundamental is identical to antifundamental, so put all antifundamental nodes to fundamental central node.
                if i == 1 and self.theory[0] == 1 and self.theory[1] == 1:
                    self.superpotential_graph.add_nodes_from(
                        [
                            (f'qb{j}', {'node_type': 0, 'conj': 0, 'dim': rep_dims[0], 'index': j}) for j in range(1, field_num + 1)
                        ]
                    )
                    self.superpotential_graph.add_edges_from(
                        [
                            ('q', f'qb{j}') for j in range(1, field_num + 1)
                        ]
                    )
                else:
                    conj = 1 if i == 1 or i == 4 or i == 6 else 0
                    self.superpotential_graph.add_node(field, node_type=1, conj=conj, dim=0, index=0)
                    self.superpotential_graph.add_nodes_from(
                        [
                            (f'{field}{j}', {'node_type': 0, 'conj': 0, 'dim': rep_dims[i], 'index': j}) for j in range(1, field_num + 1)
                        ]
                    )
                    self.superpotential_graph.add_edges_from(
                        [
                            (field, f'{field}{j}') for j in range(1, field_num + 1)
                        ]
                    )

        if M_num > 0:
            self.superpotential_graph.add_node('M', node_type=2, conj=0, dim=0, index=0)
            self.superpotential_graph.add_nodes_from(
                [
                    (f'M{i}', {'node_type': 0, 'conj': 0, 'dim': 1, 'index': i}) for i in range(1, M_num + 1)
                ]
            )
            self.superpotential_graph.add_edges_from(
                [
                    ('M', f'M{i}') for i in range(1, M_num + 1)
                ]
            )

        if X_num > 0:
            self.superpotential_graph.add_node('X', node_type=2, conj=1, dim=0, index=0)
            self.superpotential_graph.add_nodes_from(
                [
                    (f'X{i}', {'node_type': 0, 'conj': 0, 'dim': 1, 'index': i}) for i in range(1, X_num + 1)
                ]
            )
            self.superpotential_graph.add_edges_from(
                [
                    ('X', f'X{i}') for i in range(1, X_num + 1)
                ]
            )

        if term_num > 0:
            self.superpotential_graph.add_node('Term', node_type=3, conj=0, dim=0, index=0)
            self.superpotential_graph.add_nodes_from(
                [
                    (f'Term{i}', {'node_type': 0, 'conj': 0, 'dim': 0, 'index': 0}) for i in range(1, term_num + 1)
                ]
            )
            self.superpotential_graph.add_edges_from(
                [
                    ('Term', f'Term{i}') for i in range(1, term_num + 1)
                ]
            )

        for i in range(len(self.superpotential)):
            term = self.superpotential[i]
            term_node = f'Term{i + 1}'

            for op, index, exponent in term:
                op_node = f'{op}{index}'
                self.superpotential_graph.add_edges_from(
                    [
                        (term_node, op_node) for _ in range(exponent)
                    ]
                )

    def set_theory(self, theory: Union[str, tuple[int, ...]]) -> None:
        """
        Sets the theory of the superpotential.
        :param theory: Theory of the superpotential in string or serialized tuple.
        """
        self.theory = serialize_theory_name(theory) if isinstance(theory, str) else theory
        self.dynkin_diagram.clear()

        ade_class = self.theory[0]
        rank = self.theory[1]
        build_dynkin_diagram(self.dynkin_diagram, ade_class, rank)

    def set_superpotential(self, superpotential: Union[str, list[str]]) -> None:
        """
        Sets the superpotential. Note that if the superpotential and theory does not match,
        it will cause error.
        :param superpotential: Raw superpotential string or list of term strings.
        """
        self.superpotential = serialize_w_terms(superpotential) if isinstance(superpotential, str) else serialize_w_term_list(superpotential)
        self.superpotential_graph.clear()

        self.__build_superpotential_graph()

    def get_theory_data(self) -> Data:
        """
        Gets the data of the theory by dynkin diagram graph with node attribute
        [short, mark, comark].
        :return: PyG graph data of the dynkin diagram of the theory.
        """
        return from_networkx(self.dynkin_diagram, group_node_attrs=['short', 'mark', 'comark'])

    def get_superpotential_data(self) -> Data:
        """
        Gets the data of the superpotential by graph with node attribute
        [node_type, conj, dim].
        :return: PyG graph data of the superpotential graph.
        """
        return from_networkx(self.superpotential_graph, group_node_attrs=['node_type', 'conj', 'dim'])
