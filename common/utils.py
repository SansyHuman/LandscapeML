import math
import json
import os
from typing import Union, Any

from pyspark.ml.feature import Bucketizer
from pyspark.sql import functions as F
from pyspark.sql import Window
import pyspark.sql.dataframe as ps
from sklearn.cluster import KMeans, BisectingKMeans
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import numpy as np


def prime_numbers(n: int) -> list[int]:
    """
    Gets a list of prime number.
    :param n: the number of prime numbers.
    :return: a list of prime numbers.
    """
    if n <= 0:
        return []
    primes = [2]
    for i in range(1, n):
        prime = primes[-1] + 1
        while True:
            is_prime = True
            for p in primes:
                if p > math.sqrt(prime):
                    break
                if prime % p == 0:
                    is_prime = False
                    break

            if is_prime:
                primes.append(prime)
                break

            prime += 1

    return primes


def median_sorted(data):
    """
    Gets median from a sorted list.
    :param data: sorted data
    :return: median of the data
    """
    n = len(data)
    if n % 2 == 0:
        return (data[n // 2 - 1] + data[n // 2]) / 2
    else:
        return data[(n - 1) // 2]


class Monomial:
    """
    A class of monomials.
    """
    def __init__(self, indet: str, expo: float):
        """
        Construct a monomial.
        :param indet: indeterminate letter.
        :param expo: exponent of the monomial.
        """
        self.indeterminate = indet
        self.exponential = expo

    def __str__(self):
        if self.exponential == 1:
            return self.indeterminate

        return f'{self.indeterminate}^{self.exponential}'

    def __eq__(self, other):
        return self.indeterminate == other.indeterminate and self.exponential == other.exponential


class Term:
    """
    A class of terms.
    """
    def __init__(self, coeff: float, *args: Monomial):
        """
        Construct a term.
        :param coeff: Coefficient of the term.
        :param args: Monomials of the term.
        """
        self.coefficient = coeff
        self.monomials: dict[str, Monomial] = dict()
        for arg in args:
            self.__add_monomial(arg)

    def __add_monomial(self, monomial: Monomial) -> None:
        if monomial.indeterminate in self.monomials:
            self.monomials[monomial.indeterminate].exponential += monomial.exponential
        else:
            self.monomials[monomial.indeterminate] = monomial

    def indeterminates(self) -> set[str]:
        """
        Returns a set of indeterminates.
        :return: a set of indeterminates.
        """
        return set(self.monomials.keys())

    def exponent(self, indeterminate: str) -> Union[float, None]:
        """
        Returns the exponent of the indeterminate.
        :param indeterminate: indeterminate.
        :return: exponent. If the indeterminate does not exist, return None.
        """
        if indeterminate in self.monomials:
            return self.monomials[indeterminate].exponential

        return None

    def __str__(self) -> str:
        indeterminates = list(self.monomials.keys())
        indeterminates.sort()
        term = ""
        if self.coefficient == -1:
            term += '-'
        elif self.coefficient != 1:
            term += f'{self.coefficient}*'
        else:
            if len(self.monomials) == 0:
                term += '1'
                return term

        for letter in indeterminates:
            mon = self.monomials[letter]
            term += f'{mon}*'

        return term[:-1]

    def __contains__(self, item: Monomial) -> bool:
        if item.indeterminate in self.monomials:
            return item == self.monomials[item.indeterminate]

        return False


class Polynomial:
    """
    A class of polynomials.
    """
    def __init__(self, *args: Term):
        """
        Construct a polynomial.
        :param args: list of terms.
        """
        self.terms = list(args)

    def add_term(self, term: Term) -> None:
        """
        Adds a term to the polynomial.
        :param term: term to add.
        """
        self.terms.append(term)

    def find_with(self, mono: Monomial) -> list[Term]:
        """
        Finds all terms with the given monomial.
        :param mono: monomial to find.
        :return: list of terms with the given monomial.
        """
        result = []
        for term in self.terms:
            if mono.exponential == 0:
                if not (mono.indeterminate in term.indeterminates()):
                    result.append(term)
            elif mono in term:
                result.append(term)

        return result

    def __str__(self) -> str:
        poly = ""
        for term in self.terms:
            if term.coefficient < 0 and len(poly) != 0:
                poly = poly[:-1]
            poly += f'{term}+'
        if len(self.terms) != 0:
            return poly[:-1]

        return poly


def is_number(n: str) -> bool:
    """
    Checks if a string is a number.
    :param n: string to check.
    :return: True if the string is a number. Else, false.
    """
    try:
        float(n)
        return True
    except ValueError:
        return False


def find_number(s: str, start: int=0) -> tuple[int, int]:
    """
    Finds the index of the first number.
    :param s: string to be checked.
    :param start: starting index to search. If not specified, 0.
    :return: start index and length of the first number. If it does not exist, return (-1, 0).
    """
    start_index, length = -1, 0
    for i in range(start, len(s)):
        if s[i].isdigit():
            start_index = i
            break

    if start_index == -1:
        return -1, 0

    for i in range(start_index, len(s)):
        if s[i].isdigit():
            length += 1
        else:
            break

    return start_index, length


def remove_str_between(s: str, start: int, end: int) -> str:
    """
    Removes the substring between the start index and the end index.
    :param s: string to remove substring.
    :param start: start index to remove.
    :param end: next index of the last character of substring to remove
    :return: removed string.
    """
    if start >= end:
        return s

    return s[:start] + s[end:]


def to_poly(expr: str) -> Polynomial:
    """
    Converts the expression string to polynomial.
    :param expr: string of the expression.
    :return: polynomial.
    """

    if expr[0] != '-':
        expr = f'+{expr}'

    poly = Polynomial()

    while len(expr) > 0:
        next_expr_plus = expr.find('+', 1)
        next_expr_minus = expr.find('-', 1)
        next_expr = -1
        term_str = ''

        if next_expr_plus != -1:
            next_expr = next_expr_plus
        if next_expr_minus != -1:
            if next_expr != -1:
                next_expr = min(next_expr, next_expr_minus)
            else:
                next_expr = next_expr_minus

        if next_expr == -1:
            term_str = expr[:]
            expr = ''
        else:
            term_str = expr[:next_expr]
            expr = expr[next_expr:]

        fractions = term_str.split('/')
        numerator = fractions[0].strip()
        numerator = numerator.replace('(', '')
        numerator = numerator.replace(')', '')

        denominator = None
        if len(fractions) > 1:
            denominator = fractions[1].strip()
            if denominator[0] == '(' and denominator[-1] == ')':
                denominator = denominator[1:-1]
            if denominator[0] != '-':
                denominator = f'+{denominator}'

        coeff, monomials = __make_monomials(numerator)
        if denominator is not None:
            _, denominators = __make_monomials(denominator)
            for denom in denominators:
                denom.exponential = -denom.exponential
            monomials += denominators

        poly.add_term(Term(coeff, *monomials))

    return poly


def __make_monomials(term: str) -> tuple[float, list[Monomial]]:
    """
    Returns the coefficient and list of monomials
    """
    multiples = [s.strip() for s in term.split('*')]
    coefficient = 1
    monomials = []

    if is_number(multiples[0]):
        coefficient = float(multiples[0])
        multiples = multiples[1:]
    else:
        if multiples[0][0] == '-':
            coefficient = -1
        multiples[0] = multiples[0][1:]

    for i in range(len(multiples)):
        multiple = multiples[i]
        indet_expo = [s.strip() for s in multiple.split('^')]
        if len(indet_expo) == 1:
            monomials.append(Monomial(indet_expo[0], 1))
        else:
            monomials.append(Monomial(indet_expo[0], float(indet_expo[1])))

    return coefficient, monomials


class GenericDataset(Dataset):
    """
    General dataset class.
    """
    def __init__(self, input_data, output_data):
        """
        Create a new dataset.
        :param input_data: input data
        :param output_data: output data
        """
        self.x_data = torch.tensor(input_data)
        self.y_data = torch.tensor(output_data)

    def __getitem__(self, index):
        return self.x_data[index].float(), self.y_data[index].float()

    def __len__(self):
        return self.x_data.shape[0]


class FullyConnectedNetwork(nn.Module):
    """
    General fully connected network class.
    """
    def __init__(self, input_size: int, output_size: int, *args: tuple[int, nn.Module]):
        """
        Create a new fully connected network.
        :param input_size: dimension of input vector
        :param output_size: dimension of output vector
        :param args: tuple of dimensions of hidden layers and activation function
        """
        super(FullyConnectedNetwork, self).__init__()
        dims = [input_size] + [args[i][0] for i in range(len(args))] + [output_size]
        self.layers = nn.Sequential()

        for i in range(len(dims) - 2):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.layers.append(args[i][1])
        self.layers.append(nn.Linear(dims[-2], dims[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


def save_cluster_data(data, data_name: list[str], cluster_obj: Union[KMeans, BisectingKMeans], file_name: str, test_name: str, path: str) -> None:
    """
    Save data of clustering
    :param data: data used for clustering
    :param data_name: names of data
    :param cluster_obj: KMeans or BisectingKMeans object
    :param file_name: name of the data file
    :param test_name: name of the test
    :param path: path to save data
    """
    os.makedirs(path, exist_ok=True)

    n_clusters = cluster_obj.n_clusters
    n_data = len(data_name)
    clustered_data = [[[] for _ in range(n_data)] for _ in range(n_clusters)]

    for i in range(len(data)):
        cluster = cluster_obj.labels_[i]
        for j in range(n_data):
            clustered_data[cluster][j].append(data[i][j])

    json_data = dict()
    json_data['data_name'] = data_name
    json_data['clusters'] = [dict() for _ in range(n_clusters)]

    for i in range(n_clusters):
        json_data['clusters'][i]['num_data'] = len(clustered_data[i][0])
        json_data['clusters'][i]['center'] = list(cluster_obj.cluster_centers_[i])
        for j in range(n_data):
            clustered_data[i][j].sort()
        json_data['clusters'][i]['min'] = [clustered_data[i][k][0] for k in range(n_data)]
        json_data['clusters'][i]['max'] = [clustered_data[i][k][-1] for k in range(n_data)]
        json_data['clusters'][i]['average'] = [np.mean(clustered_data[i][k]) for k in range(n_data)]
        json_data['clusters'][i]['median'] = [median_sorted(clustered_data[i][k]) for k in range(n_data)]

    with open(f'{path}/{file_name}_{test_name}.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    data_list = ['center', 'min', 'max', 'average', 'median']
    with open(f'{path}/{file_name}_{test_name}.csv', 'w') as csv_file:
        for data_kind in data_list:
            csv_file.write(f'{data_kind}, ')
            for i in range(n_data):
                csv_file.write(f'{data_name[i]}, ')
            csv_file.write('\n')

            for i in range(n_clusters):
                csv_file.write(f'{i}, ')
                for j in range(n_data):
                    csv_file.write(f'{json_data['clusters'][i][data_kind][j]}, ')
                csv_file.write('\n')

            csv_file.write('\n')


class PairData(Data):
    """
    Class of a pair of graph data. The data key name should be the following;
    x_1: node features of first graph
    x_2: node features of second graph
    edge_index_1: graph connectivity matrix of first graph
    edge_index_2: graph connectivity matrix of second graph
    y: target data of the pair of input graphs
    When you make dataloader of PairData, you should put follow batch
    ['x_1', 'x_2'].
    """
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'edge_index_1':
            return self.x_1.size(0)
        if key == 'edge_index_2':
            return self.x_2.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def balanced_sample_theories(df: ps.DataFrame, theory_col: str, a_col: str, c_col: str,
                    min_a: float, max_a: float, min_c: float, max_c: float,
                    n_per_theory: int) -> ps.DataFrame:
    """
    Pick data evenly from different theories within a numerical central charge range,
    excluding categories with fewer than n_per_theory rows.
    """

    # Filter rows within the label range
    filtered = df.filter(
        (F.col(a_col) >= min_a) & (F.col(a_col) < max_a) &
        (F.col(c_col) >= min_c) & (F.col(c_col) < max_c)
    )

    # Add a random number column for shuffling
    shuffled = filtered.withColumn("rand", F.rand())

    # Count rows per category
    counts = shuffled.groupBy(theory_col).count()

    # Extract valid categories as a Series
    valid_categories = counts.filter(F.col("count") >= n_per_theory).select(theory_col)

    # Keep only rows from valid categories
    shuffled = shuffled.join(valid_categories, on=theory_col, how="inner")

    # Rank rows within each category by random number
    window = Window.partitionBy(theory_col).orderBy("rand")
    ranked = shuffled.withColumn("rank", F.row_number().over(window))

    # Keep only the first n_per_category per group
    sampled = ranked.filter(F.col("rank") <= n_per_theory)

    return sampled.drop("rand", "rank")


def manual_sample_theories(df: ps.DataFrame, theory_col: str, a_col: str, c_col: str,
                    theories: list[str], n_per_theory: int) -> ps.DataFrame:
    """
    Pick data evenly from specified theories,
    excluding categories with fewer than n_per_theory rows.
    """

    # Filter rows with specified theories
    filtered = df.filter(F.col(theory_col).isin(theories))

    # Add a random number column for shuffling
    shuffled = filtered.withColumn("rand", F.rand())

    # Count rows per category
    counts = shuffled.groupBy(theory_col).count()

    # Extract valid categories as a Series
    valid_categories = counts.filter(F.col("count") >= n_per_theory).select(theory_col)

    # Keep only rows from valid categories
    shuffled = shuffled.join(valid_categories, on=theory_col, how="inner")

    # Rank rows within each category by random number
    window = Window.partitionBy(theory_col).orderBy("rand")
    ranked = shuffled.withColumn("rank", F.row_number().over(window))

    # Keep only the first n_per_category per group
    sampled = ranked.filter(F.col("rank") <= n_per_theory)

    return sampled.drop("rand", "rank")


def balanced_sample_bins(df: ps.DataFrame, charge_col: str, min_charge: float, max_charge: float, n_bins: int, n_per_bins: int):
    """
    Pick data evenly from specified range of charges.
    """

    # Filter rows with specified charge range
    filtered = df.filter((F.col(charge_col) >= min_charge) & (F.col(charge_col) < max_charge))

    # Discretize into n_bins bins
    bin_step = (max_charge - min_charge) / n_bins
    splits = [min_charge + i * bin_step for i in range(n_bins)] + [float("inf")]
    discretizer = Bucketizer(
        splits=splits,
        inputCol=charge_col,
        outputCol="bucket"
    )
    bucketed = discretizer.transform(filtered)

    # Assign random row numbers within each bucket
    window = Window.partitionBy("bucket").orderBy(F.rand())
    ranked = bucketed.withColumn("rank", F.row_number().over(window))

    # keep only up to n_per_bins rows per bucket
    sampled = ranked.filter(F.col("rank") <= n_per_bins)

    return sampled.drop("rank", "bucket")
