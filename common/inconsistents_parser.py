import csv
import numpy as np

from common.utils import *
import ast
import random


def lie_to_ade(lie_alg: str) -> tuple[str, int]:
    """
    Convert the lie algebra to ADE classification form.
    :param lie_alg: Compact Lie algebras, SU, SO, Sp, E, F, and G.
    :return: ADE classification form of the algebra. A_n: su_(n+1), B_n: so_(2n+1),
    C_n: sp_n, D_n: so_2n, E_6, E_7, E_8. F_4, G_2. If none of these, return '', -1.
    """
    if lie_alg.find('SU') == 0:
        start, length = find_number(lie_alg, 2)
        if start != 2:
            return '', -1
        return 'A', int(lie_alg[start:start + length]) - 1
    elif lie_alg.find('SO') == 0:
        start, length = find_number(lie_alg, 2)
        if start != 2:
            return '', -1
        size = int(lie_alg[start:start + length])
        if size % 2 == 0:
            return 'D', size // 2
        else:
            return 'B', size // 2
    elif lie_alg.find('Sp') == 0:
        start, length = find_number(lie_alg, 2)
        if start != 2:
            return '', -1
        return 'C', int(lie_alg[start:start + length])
    elif lie_alg.find('E') == 0:
        start, length = find_number(lie_alg, 1)
        if start != 1:
            return '', -1
        return 'E', int(lie_alg[start:start + length])
    elif lie_alg.find('F') == 0:
        start, length = find_number(lie_alg, 1)
        if start != 1:
            return '', -1
        return 'F', int(lie_alg[start:start + length])
    elif lie_alg.find('G') == 0:
        start, length = find_number(lie_alg, 1)
        if start != 1:
            return '', -1
        return 'G', int(lie_alg[start:start + length])
    else:
        return '', -1


def ade_to_lie(ade_class: str, size: int) -> str:
    """
    Convert the ADE classification form to lie algebra.
    :param ade_class: ADE class of the algebra, one of A, B, C, D, E, F, and G.
    :param size: size of the algebra.
    :return: Lie algebra of the ADE classification. If none of these, return ''.
    """
    if ade_class == 'A':
        return f'SU{size + 1}'
    elif ade_class == 'B':
        return f'SO{size * 2 + 1}'
    elif ade_class == 'C':
        return f'Sp{size}'
    elif ade_class == 'D':
        return f'SO{size * 2}'
    elif ade_class == 'E':
        return f'E{size}'
    elif ade_class == 'F':
        return f'F{size}'
    elif ade_class == 'G':
        return f'G{size}'
    else:
        return ''


def serialize_theory_name(theory_name: str) -> tuple[int, ...]:
    """
    Convert the theory name to a tuple of integer data.
    :param theory_name: Theory name.
    :return: (GaugeGroup, GroupSize, fund, antifund, adj, symm, symm_conj, antisymm, antisymm_conj)
    GaugeGroup: A=1, B=2, C=3, D=4, E=5, F=6, G=7
    GroupSize is the index of ADE classification.
    matter contents are the number of such fields.
    """
    ade_to_index = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    num_start, num_len = find_number(theory_name)
    gauge_group = theory_name[:num_start + num_len]
    ade_class, alg_size = lie_to_ade(gauge_group)
    if ade_class == '':
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    matter_contents = theory_name[num_start + num_len:]
    fund, antifund, adj, symm, symm_conj, antisymm, antisymm_conj = 0, 0, 0, 0, 0, 0, 0

    # nf has different convention amongst different lie algebras.
    # For SU(A), it means a pair of fundamental and antifundamental to remove gauge anomaly.
    # For Sp(C), it means two fundamental to remove Witten anomaly.
    # For SO(B, D) and G2, there is no anomaly constraint so it is just the number of fundamental.
    nf_start = matter_contents.find('nf')
    if nf_start != -1:
        nf_cnt_start, nf_cnt_len = find_number(matter_contents, nf_start + 2)
        nf_cnt = int(matter_contents[nf_cnt_start:nf_cnt_start + nf_cnt_len])

        if ade_class == 'A':
            fund += nf_cnt
            antifund += nf_cnt
        elif ade_class == 'C':
            fund += 2 * nf_cnt
        else:
            fund += nf_cnt

        matter_contents = remove_str_between(matter_contents, nf_start, nf_cnt_start + nf_cnt_len)

    adj_start = matter_contents.find('adj')
    if adj_start != -1:
        adj_cnt_start, adj_cnt_len = find_number(matter_contents, adj_start + 3)
        adj_cnt = int(matter_contents[adj_cnt_start:adj_cnt_start + adj_cnt_len])

        adj += adj_cnt

        matter_contents = remove_str_between(matter_contents, adj_start, adj_cnt_start + adj_cnt_len)

    fund_start = matter_contents.find('f')
    if fund_start != -1:
        fund_cnt_start, fund_cnt_len = find_number(matter_contents, fund_start + 1)
        fund_cnt = int(matter_contents[fund_cnt_start:fund_cnt_start + fund_cnt_len])

        fund += fund_cnt

    antifund_start = matter_contents.find('F')
    if antifund_start != -1:
        antifund_cnt_start, antifund_cnt_len = find_number(matter_contents, antifund_start + 1)
        antifund_cnt = int(matter_contents[antifund_cnt_start:antifund_cnt_start + antifund_cnt_len])

        antifund += antifund_cnt

    symm_start = matter_contents.find('s')
    if symm_start != -1:
        symm_cnt_start, symm_cnt_len = find_number(matter_contents, symm_start + 1)
        symm_cnt = int(matter_contents[symm_cnt_start:symm_cnt_start + symm_cnt_len])

        symm += symm_cnt

    symm_conj_start = matter_contents.find('S')
    if symm_conj_start != -1:
        symm_conj_cnt_start, symm_conj_cnt_len = find_number(matter_contents, symm_conj_start + 1)
        symm_conj_cnt = int(matter_contents[symm_conj_cnt_start:symm_conj_cnt_start + symm_conj_cnt_len])

        # 14-dim rep in SO5 is rank-2 symmetric tensor, but it is not in Sp2.
        # So 14-dim field is noted by capital S, and rank-2 symmetric tensor of SO5 also does.
        # Also, SO and Sp representations are (pseudo)real so there is no conjugate, so s and S are mixed.
        if ade_class == 'B' and alg_size == 2:
            symm += symm_conj_cnt
        elif ade_class == 'C' and alg_size == 2:
            symm += symm_conj_cnt
        else:
            symm_conj += symm_conj_cnt

    antisymm_start = matter_contents.find('a')
    if antisymm_start != -1:
        antisymm_cnt_start, antisymm_cnt_len = find_number(matter_contents, antisymm_start + 1)
        antisymm_cnt = int(matter_contents[antisymm_cnt_start:antisymm_cnt_start + antisymm_cnt_len])

        antisymm += antisymm_cnt

    antisymm_conj_start = matter_contents.find('A')
    if antisymm_conj_start != -1:
        antisymm_conj_cnt_start, antisymm_conj_cnt_len = find_number(matter_contents, antisymm_conj_start + 1)
        antisymm_conj_cnt = int(matter_contents[antisymm_conj_cnt_start:antisymm_conj_cnt_start + antisymm_conj_cnt_len])

        antisymm_conj += antisymm_conj_cnt

    return tuple([ade_to_index[ade_class], alg_size, fund, antifund, adj, symm, symm_conj, antisymm, antisymm_conj])


def get_inconsistent_paths(inconsistents_path: str) -> dict[tuple[int, ...], list[str]]:
    """
    Gets all inconsistent theory paths.
    :param inconsistents_path: Folder of inconsistent theories.
    :return: dictionary of serialized theory name and corresponding paths to inconsistent theories.
    """
    group_fields_dir: dict[tuple[int, ...], list[str]] = dict()

    for group in os.listdir(inconsistents_path):
        group_path = os.path.join(inconsistents_path, group)
        if not os.path.isdir(group_path):
            continue
        for theory in os.listdir(group_path):
            theory_path = os.path.join(group_path, theory)
            if not os.path.isdir(theory_path):
                continue
            log_path = os.path.join(theory_path, f'{theory}_log.txt')
            if not os.path.isfile(log_path):
                continue
            vector = serialize_theory_name(theory)
            if vector[0] == 0:
                continue
            if vector in group_fields_dir:
                group_fields_dir[vector].append(log_path)
            else:
                group_fields_dir[vector] = [log_path]

    return group_fields_dir


def inconsistents_parser(filename: str, inconsistents_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse consistent theories and inconsistent theories
    :param filename: File of consistent theories
    :param inconsistents_path: Folder of inconsistent theories
    :return:
    """
    csv.field_size_limit(np.iinfo(np.int32).max)

    data = None
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    field_content_index, a_index, c_index = -1, -1, -1
    for i in range(len(data[0])):
        if data[0][i] == "Name":
            field_content_index = i
        elif data[0][i] == "CentralChargeA":
            a_index = i
        elif data[0][i] == "CentralChargeC":
            c_index = i

    field_contents_index = dict()
    field_contents = []
    a_charges = []
    c_charges = []

    for i in range(1, len(data)):
        field_content = data[i][field_content_index]
        a, c = float(data[i][a_index]), float(data[i][c_index])

        if field_content not in field_contents_index:
            field_contents_index[field_content] = len(field_contents_index)
        field_contents.append(field_contents_index[field_content])
        a_charges.append(a)
        c_charges.append(c)

    # [GaugeGroup, GroupSize, fund, antifund, adj, symm, symm_conj, antisymm, antisymm_conj]
    # GaugeGroup: A=1, B=2, C=3, D=4, E=5, F=6, G=7
    # GroupSize is the index of ADE classification.
    # matter contents are the number of such fields.
    index_group_fields = dict()

    for content, index in field_contents_index.items():
        vector = serialize_theory_name(content)
        if vector[0] == 0:
            continue

        index_group_fields[index] = vector

    group_fields_dir = get_inconsistent_paths(inconsistents_path)

    input_data = []
    output_data = []

    for i in range(len(field_contents)):
        if field_contents[i] not in index_group_fields:
            continue
        input_data.append(list(index_group_fields[field_contents[i]]) + [a_charges[i], c_charges[i]])
        output_data.append([0]) # 0: consistent, 1: inconsistent

    for content, index in field_contents_index.items():
        if index not in index_group_fields:
            continue
        vector = index_group_fields[index]
        if vector not in group_fields_dir:
            print(f'Warning: the theory {content} does not have the log file.')
            continue

        inc_paths = group_fields_dir[vector]

        for inc_path in inc_paths:
            with open(inc_path) as inc_file:
                data = inc_file.readlines()
                for log in data:
                    log = ast.literal_eval(log.strip())
                    if log == '':
                        continue
                    if log['consistency'] == 'inconsistent':
                        input_data.append(list(index_group_fields[index]) + [log['a'], log['c']])
                        output_data.append([1])

    input_data = np.array(input_data)
    output_data = np.array(output_data)

    perm = np.random.permutation(len(input_data))
    return input_data[perm], output_data[perm]


def inconsistents_graph_parser(filename: str, inconsistents_path: str) -> list[PairData]:
    """
    Parses consistent theories and inconsistent theories.
    :param filename: File of consistent theories
    :param inconsistents_path: Folder of inconsistent theories
    :return: List of PairData which contains dynkin diagram and superpotential graph and consistency mark as a target y.
    """
    csv.field_size_limit(np.iinfo(np.int32).max)

    data = None
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    field_content_index, w_index = -1, -1
    for i in range(len(data[0])):
        if data[0][i] == "Name":
            field_content_index = i
        elif data[0][i] == "Superpotentials":
            w_index = i

    field_contents_index = dict()
    field_contents = []
    superpotentials = []

    for i in range(1, len(data)):
        field_content = data[i][field_content_index]
        w = data[i][w_index]

        if field_content not in field_contents_index:
            field_contents_index[field_content] = len(field_contents_index)
        field_contents.append(field_contents_index[field_content])
        superpotentials.append(w)

    index_group_fields = dict()

    for content, index in field_contents_index.items():
        vector = serialize_theory_name(content)
        if vector[0] == 0:
            continue

        index_group_fields[index] = vector

    group_fields_dir = get_inconsistent_paths(inconsistents_path)

    from common.superpotential_parser import Superpotential
    w_obj = Superpotential()
    data_list: list[PairData] = []
    prev_theory = None

    for i in range(len(field_contents)):
        if field_contents[i] not in index_group_fields:
            continue
        theory = index_group_fields[field_contents[i]]
        w = superpotentials[i]

        if prev_theory != theory:
            w_obj.set_theory(theory)
        w_obj.set_superpotential(w)

        dynkin_diagram = w_obj.get_theory_data()
        superpotential_graph = w_obj.get_superpotential_data()

        data = PairData(x_1=dynkin_diagram.x, x_2=superpotential_graph.x,
                        edge_index_1=dynkin_diagram.edge_index, edge_index_2=superpotential_graph.edge_index,
                        y=torch.tensor([0])) # 0: consistent, 1: inconsistent
        data_list.append(data)
        prev_theory = theory

    for content, index in field_contents_index.items():
        if index not in index_group_fields:
            continue
        vector = index_group_fields[index]
        if vector not in group_fields_dir:
            print(f'Warning: the theory {content} does not have the log file.')
            continue

        w_obj.set_theory(content)
        inc_paths = group_fields_dir[vector]

        for inc_path in inc_paths:
            with open(inc_path) as inc_file:
                data = inc_file.readlines()
                for log in data:
                    log = ast.literal_eval(log.strip())
                    if log == '':
                        continue
                    if log['consistency'] == 'inconsistent':
                        w_obj.set_superpotential(log['w'])

                        dynkin_diagram = w_obj.get_theory_data()
                        superpotential_graph = w_obj.get_superpotential_data()

                        data = PairData(x_1=dynkin_diagram.x, x_2=superpotential_graph.x,
                                        edge_index_1=dynkin_diagram.edge_index,
                                        edge_index_2=superpotential_graph.edge_index,
                                        y=torch.tensor([1]))
                        data_list.append(data)

    random.shuffle(data_list)
    return data_list
