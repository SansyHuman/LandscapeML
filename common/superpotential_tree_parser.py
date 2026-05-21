import csv
import ast

import numpy as np


class TheoryData:
    def __init__(self, id, w, a, c, r):
        self.id = id
        self.w = w
        self.a = a
        self.c = c
        self.r = r


class SuperpotentialTreeNode:
    def __init__(self, theory_data: TheoryData, added_term=''):
        self.theory_data: TheoryData = theory_data
        self.added_term: str = added_term
        self.children: list[SuperpotentialTreeNode] = []


def refine_r_charge_string(raw_r: str) -> str:
    r = '{' + raw_r[1:-1] + '}'

    i = 0
    while True:
        if r[i] == ':':
            for j in range(i - 1, -1, -1):
                if r[j] == ' ' or r[j] == '{':
                    r = r[:j + 1] + '\"' + r[j + 1:i] + '\"' + r[i:]
                    i += 2
                    break

        i += 1
        if i >= len(r):
            break

    return r


def build_superpotential_tree(filename: str) -> tuple[str, SuperpotentialTreeNode]:
    """
    Builds a superpotential tree from the given file.
    :param filename: csv file that contains landscape data.
    :return: Superpotential tree of selected theory where children is the superpotential which adds one
    relevant gauge-invariant term from parent.
    """
    csv.field_size_limit(np.iinfo(np.int32).max)

    data = None
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    id_index, theory_index, w_index, a_index, c_index, r_index = -1, -1, -1, -1, -1, -1
    for i in range(len(data[0])):
        if data[0][i] == "id":
            id_index = i
        elif data[0][i] == "Name":
            theory_index = i
        elif data[0][i] == "Superpotentials":
            w_index = i
        elif data[0][i] == "CentralChargeA":
            a_index = i
        elif data[0][i] == "CentralChargeC":
            c_index = i
        elif data[0][i] == "Rcharges":
            r_index = i

    theory_num = dict()
    for i in range(1, len(data)):
        theory = data[i][theory_index]
        if theory not in theory_num:
            theory_num[theory] = 0
        theory_num[theory] += 1

    theory_num_list = list(theory_num.items())
    theory_num_list.sort(key=lambda x: x[1], reverse=True)

    print('Theory statistics in the file')
    print('=================================')
    for theory, num in theory_num_list:
        print(f'{theory}: {num}')

    theory_name = input('Choose a theory to build a tree: ')
    assert theory_name in theory_num

    theory_data = []

    for i in range(1, len(data)):
        if data[i][theory_index] != theory_name:
            continue

        id = int(data[i][id_index])
        w = data[i][w_index]
        w = w[1:-1].split(',')
        w_terms = []
        for j in range(len(w)):
            w_term = w[j].strip()
            if w_term != '':
                w_terms.append(w_term)

        a = float(data[i][a_index])
        c = float(data[i][c_index])

        r = refine_r_charge_string(data[i][r_index])
        r_charges = ast.literal_eval(r)

        theory_data.append(TheoryData(id, w_terms, a, c, r_charges))

    theory_by_length = dict()
    max_length = -1
    for theory in theory_data:
        w_length = len(theory.w)
        if w_length > max_length:
            max_length = w_length
        if w_length not in theory_by_length:
            theory_by_length[w_length] = []
        theory_by_length[w_length].append(theory)

    superpotential_tree = SuperpotentialTreeNode(theory_by_length[0][0])

    for theory in theory_by_length[1]:
        superpotential_tree.children.append(SuperpotentialTreeNode(theory, theory.w[0]))

    for length in range(2, max_length + 1):
        if length not in theory_by_length:
            continue
        for theory in theory_by_length[length]:
            parent_candidate = superpotential_tree

            found_candidate = False
            for i in range(length - 1):
                found_candidate = False

                for candidate in parent_candidate.children:
                    if candidate.added_term == theory.w[i]:
                        parent_candidate = candidate
                        found_candidate = True
                        break

                if not found_candidate:
                    break

            if not found_candidate:
                continue
            parent_candidate.children.append(SuperpotentialTreeNode(theory, theory.w[length - 1]))

    return theory_name, superpotential_tree
