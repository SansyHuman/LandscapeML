from common.utils import *

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np

marginal = Monomial('t', 6)
spectral = Monomial('y', 0)
fermionic = Monomial('y', 1)
bosonic = Monomial('y', 2)


class SuperConformalIndex:
    """
    Class which contains the information of a superconformal index.
    """
    def __init__(self, index: str) -> None:
        """
        Converts the index string into a polynomial object and parses information.
        :param index: index string.
        """
        self.index = to_poly(index)
        # terms with exponent of t less than 6
        self.short_index = Polynomial(
            *filter(lambda term: term.exponent('t') < 6,
                    filter(lambda term2: term2.exponent('t') is not None, self.index.terms))
        )
        # list of [coefficient, t exponent, y exponent]
        self.terms_list = []
        # # of marginal operators - rank of IR flavor symmetry
        self.num_dim3_minus_f = 0
        marginal_term = self.index.find_with(marginal)
        if len(marginal_term) > 0:
            self.num_dim3_minus_f = round(self.index.find_with(marginal)[0].coefficient)

        # dimensions of scalar operators in the order of increasing
        self.dims: list[float] = []
        # the scalar operator's dimensions and coefficient of the sci terms
        self.spectrum: dict[float, int] = dict()
        # dimensions of relevant scalar operators in the order of increasing
        self.relevant_dims: list[float] = []
        # the number of relevant scalar operators with each dimensions
        self.relevant_spectrum: dict[float, int] = dict()
        # total number of relevant operators
        self.num_relevant_ops = 0

        # dimensions of fermionic operators in the order of increasing
        self.fermion_dims: list[float] = []
        # the fermionic operator's dimensions and minus coefficient(since (-1)^F is -1) of the sci terms
        self.fermion_spectrum: dict[float, int] = dict()

        # dimensions of bosonic operators in the order of increasing
        self.boson_dims: list[float] = []
        # the bosonic operator's dimensions and coefficient of the sci terms
        self.boson_spectrum: dict[float, int] = dict()

        for term in self.index.terms:
            coeff = term.coefficient
            t_exp = term.exponent('t')
            y_exp = term.exponent('y')
            if t_exp is None:
                t_exp = 0
            if y_exp is None:
                y_exp = 0

            self.terms_list.append([coeff, t_exp, y_exp])

        terms_spectrum = self.index.find_with(spectral)
        tmp_dims = set()
        for term in terms_spectrum:
            dim = term.exponent('t')
            if dim is not None:
                dim = dim / 2.0 # t^3R, dim = 3R/2
                cnt = round(term.coefficient)

                if dim in self.spectrum:
                    self.spectrum[dim] += cnt
                else:
                    self.spectrum[dim] = cnt

                if dim < 3.0: # relevant
                    if dim in self.relevant_spectrum:
                        self.relevant_spectrum[dim] += cnt
                    else:
                        self.relevant_dims.append(dim)
                        self.relevant_spectrum[dim] = cnt
                    self.num_relevant_ops += cnt
                elif dim > 3.0: # irrelevant
                    if cnt < 0: # ignore terms with negative coefficients
                        continue

                tmp_dims.add(dim)

        self.dims = list(tmp_dims)
        self.dims.sort()
        self.relevant_dims.sort()
        # smallest dimension among all operators
        self.smallest_dim = self.dims[0]

        terms_fermionic = self.index.find_with(fermionic)
        tmp_dims = set()
        for term in terms_fermionic:
            dim = term.exponent('t')
            if dim is not None:
                dim = dim / 2.0 # TODO: is j1 always zero?
                cnt = -round(term.coefficient)

                if dim in self.fermion_spectrum:
                    self.fermion_spectrum[dim] += cnt
                else:
                    self.fermion_spectrum[dim] = cnt

                tmp_dims.add(dim)

        self.fermion_dims = list(tmp_dims)
        self.fermion_dims.sort()

        terms_bosonic = self.index.find_with(bosonic)
        tmp_dims = set()
        for term in terms_bosonic:
            dim = term.exponent('t')
            if dim is not None:
                dim = dim / 2.0
                cnt = round(term.coefficient)

                # For j2=1 adjoint rep, spin-0 operator is counted as scalar in spectral terms. Should be removed.
                if dim in self.spectrum:
                    self.spectrum[dim] -= cnt
                    if self.spectrum[dim] == 0:
                        del self.spectrum[dim]
                        if dim in self.dims:
                            self.dims.remove(dim)

                if dim < 3.0: # relevant
                    if dim in self.relevant_spectrum:
                        self.relevant_spectrum[dim] -= cnt
                        if self.relevant_spectrum[dim] == 0:
                            del self.relevant_spectrum[dim]
                            if dim in self.relevant_dims:
                                self.relevant_dims.remove(dim)
                        self.num_relevant_ops -= cnt

                if dim in self.boson_spectrum:
                    self.boson_spectrum[dim] += cnt
                else:
                    self.boson_spectrum[dim] = cnt

                tmp_dims.add(dim)

        self.boson_dims = list(tmp_dims)
        self.boson_dims.sort()

    def featurize_dimensions(self, grid: np.ndarray, kde_bandwidth: float) -> np.ndarray:
        """
        Gets the feature vector of dimensions of the spectrum.
        :param grid: vector of grid values. It should be a vector of sequential numbers with constant steps.
        :param kde_bandwidth: bandwidth of the feature grid.
        :return: feature vector of dimensions of the spectrum.
        """
        v = np.asarray(sorted(self.dims), dtype=float)
        if v.size == 0:
            return np.zeros(len(grid) + 7 + 9)

        kde = kernel_density_estimation(v, grid, kde_bandwidth)

        uniq = np.unique(v.round(4))
        gaps = np.diff(uniq) if uniq.size > 1 else np.array([0.0])
        gap_feat = [
            gaps.min(),
            gaps.max(),
            gaps.mean(),
            gaps.std(),
            np.median(gaps),
            np.quantile(gaps, 0.25),
            np.quantile(gaps, 0.75),
        ]

        # Why is there length of exponents vector and log of length together?
        summary = [
            len(v),
            np.log(len(v)),
            v.mean(),
            v.std(),
            v.min(),
            v.max(),
            np.median(v),
            np.quantile(v, 0.25),
            np.quantile(v, 0.75),
        ]

        return np.concatenate([kde, gap_feat, summary])

    def featurize_relevant_spectrum(self, grid: np.ndarray, kde_bandwidth: float) -> np.ndarray:
        """
        Gets the feature vector of relevant spectrum.
        :param grid: vector of grid values. It should be a vector of sequential numbers with constant steps.
        :param kde_bandwidth: bandwidth of the feature grid.
        :return: feature vector of relevant spectrum.
        """
        v = []
        for i in range(len(self.relevant_dims)):
            dim = self.relevant_dims[i]
            cnt = self.relevant_spectrum[dim]
            v += [dim for _ in range(cnt)]
        v = np.asarray(v, dtype=float)

        if v.size == 0:
            return np.zeros(len(grid) + 7 + 9)

        kde = kernel_density_estimation(v, grid, kde_bandwidth, normalize=False)

        uniq = np.unique(v.round(4))
        gaps = np.diff(uniq) if uniq.size > 1 else np.array([0.0])
        gap_feat = [
            gaps.min(),
            gaps.max(),
            gaps.mean(),
            gaps.std(),
            np.median(gaps),
            np.quantile(gaps, 0.25),
            np.quantile(gaps, 0.75),
        ]

        # Why is there length of exponents vector and log of length together?
        summary = [
            len(v),
            np.log(len(v)),
            v.mean(),
            v.std(),
            v.min(),
            v.max(),
            np.median(v),
            np.quantile(v, 0.25),
            np.quantile(v, 0.75),
        ]

        return np.concatenate([kde, gap_feat, summary])

    def featurize_sci(self, grid: np.ndarray, kde_bandwidth: float) -> np.ndarray:
        """
        Gets the feature vector of sci coefficients and exponents.
        :param grid: vector of grid values. It should be a vector of sequential numbers with constant steps.
        :param kde_bandwidth: bandwidth of the feature grid.
        :return: feature vector of sci coefficients and exponents.
        """
        v_plus = []
        v_minus = []
        for dim, coeff in self.spectrum.items():
            if coeff > 0:
                v_plus += [dim for _ in range(coeff)]
            else:
                v_minus += [dim for _ in range(-coeff)]
        v = v_plus + v_minus

        v_plus = np.asarray(v_plus, dtype=float)
        v_minus = np.asarray(v_minus, dtype=float)
        v = np.abs(np.asarray(v, dtype=float))

        if v_plus.size + v_minus.size == 0:
            return np.zeros(len(grid) + 7 + 9)

        kde_plus = kernel_density_estimation(v_plus, grid, kde_bandwidth, normalize=False)
        kde_minus = kernel_density_estimation(v_minus, grid, kde_bandwidth, normalize=False)
        kde = kde_plus - kde_minus

        uniq = np.unique(v.round(4))
        gaps = np.diff(uniq) if uniq.size > 1 else np.array([0.0])
        gap_feat = [
            gaps.min(),
            gaps.max(),
            gaps.mean(),
            gaps.std(),
            np.median(gaps),
            np.quantile(gaps, 0.25),
            np.quantile(gaps, 0.75),
        ]

        # Why is there length of exponents vector and log of length together?
        summary = [
            len(v),
            np.log(len(v)),
            v.mean(),
            v.std(),
            v.min(),
            v.max(),
            np.median(v),
            np.quantile(v, 0.25),
            np.quantile(v, 0.75),
        ]

        return np.concatenate([kde, gap_feat, summary])

    def featurize_sci_graph(self, min_dim: float, max_dim: float) -> Data:
        dims_sorted = []
        for dim in self.spectrum.keys():
            if min_dim <= dim <= max_dim:
                dims_sorted.append(dim)
        dims_sorted.sort()

        graph = nx.Graph()
        graph.add_nodes_from(
            [
                (dim, {"dim": dim, "coeff": self.spectrum[dim]}) for dim in dims_sorted
            ]
        )
        for i in range(len(dims_sorted) - 1):
            graph.add_edge(dims_sorted[i], dims_sorted[i + 1], delta=dims_sorted[i + 1] - dims_sorted[i])

        return from_networkx(
            graph,
            group_node_attrs=["dim", "coeff"],
            group_edge_attrs=["delta"]
        )
