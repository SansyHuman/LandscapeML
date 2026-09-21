from common.utils import *

from torch_geometric.data import Data
import numpy as np
import sympy as sp
import torch


t, y = sp.symbols("t y")


class SuperConformalIndex:
    """
    Class which contains the information of a superconformal index.
    """
    def __init__(self, index: str) -> None:
        """Parse a reduced, unrefined index with integer coefficients.

        The input must be a finite sum of t**a * y**b, with rational a and
        integer b. Spectra retain signed index contributions. For scalar
        chiral primaries, a / 2 is the scaling dimension; this interpretation
        does not apply to every multiplet contributing to the index.
        """
        self.index = sp.expand(sp.sympify(
            index,
            locals={"t": t, "y": y},
            rational=True,
            convert_xor=True
        ))
        self.terms_list = []
        short_terms = []
        for term in sp.Add.make_args(self.index):
            if term == 0:
                continue
            coeff, monomial = term.as_coeff_Mul()
            powers = monomial.as_powers_dict()
            t_exp = powers.get(t, sp.S.Zero)
            y_exp = powers.get(y, sp.S.Zero)
            if (coeff.is_Integer is not True
                    or t_exp.is_Rational is not True
                    or y_exp.is_Integer is not True
                    or monomial != t**t_exp * y**y_exp):
                raise ValueError(f"Unsupported index term: {term}")
            self.terms_list.append([int(coeff), float(t_exp), float(y_exp)])
            if t_exp < 6:
                short_terms.append(term)

        self.terms_list.sort(key=lambda row: (row[1], row[2]))
        self.short_index = sp.Add(*short_terms)
        # Marginal operators minus the dimension of the IR flavor symmetry.
        self.num_dim3_minus_f = int(self.index.coeff(y, 0).coeff(t, 6))

        # In an SU(2) character expansion, the spin-j coefficient is
        # [y**(2*j)] I - [y**(2*j+2)] I. Subtract before extracting powers of t:
        # the first coefficient may vanish even when the difference is nonzero.
        scalar = self._extract_spectrum(
            self.index.coeff(y, 0) - self.index.coeff(y, 2)
        )
        fermion = self._extract_spectrum(
            -self.index.coeff(y, 1) + self.index.coeff(y, 3)
        )
        boson = self._extract_spectrum(
            self.index.coeff(y, 2) - self.index.coeff(y, 4)
        )

        # Keep exact dimensions through subtraction and cutoff comparisons;
        # public fields retain their existing Python float/int representation.
        self.spectrum = {float(dim): cnt for dim, cnt in scalar.items()}
        self.dims = [
            float(dim) for dim, cnt in scalar.items()
            if dim <= 3 or cnt > 0
        ]
        self.relevant_spectrum = {
            float(dim): cnt for dim, cnt in scalar.items() if dim < 3
        }
        self.relevant_dims = list(self.relevant_spectrum)
        self.num_relevant_ops = sum(self.relevant_spectrum.values())
        self.smallest_dim = min(self.dims, default=None)

        # These fields select spin 1/2 and spin 1, respectively. Their keys
        # are half the t exponent, not universally the primary's dimension.
        self.fermion_spectrum = {float(dim): cnt for dim, cnt in fermion.items()}
        self.fermion_dims = list(self.fermion_spectrum)
        self.boson_spectrum = {float(dim): cnt for dim, cnt in boson.items()}
        self.boson_dims = list(self.boson_spectrum)

    @staticmethod
    def _extract_spectrum(expr: sp.Expr) -> dict[sp.Rational, int]:
        """Collect signed coefficients by exact half-exponent, excluding t**0."""
        spectrum = {}
        for term in sp.Add.make_args(sp.expand(expr)):
            if term == 0:
                continue
            coeff, exponent = term.as_coeff_exponent(t)
            # The identity is not an operator counted in the reduced spectrum.
            if exponent == 0:
                continue
            dim = exponent / 2
            spectrum[dim] = spectrum.get(dim, 0) + int(coeff)
        return {dim: cnt for dim, cnt in sorted(spectrum.items()) if cnt != 0}

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
        """Build a chain using terms with min_dim <= t exponent / 2 <= max_dim.

        Nodes contain [coefficient, t exponent, y exponent]. Each adjacent
        pair has edges in both directions, with the nonnegative gap between
        half-exponents as its edge attribute. Empty selections are supported.
        """
        if min_dim > max_dim:
            raise ValueError("min_dim must not exceed max_dim")
        terms = [
            row for row in self.terms_list
            if min_dim <= row[1] / 2 <= max_dim
        ]
        edges = []
        gaps = []
        for i in range(len(terms) - 1):
            edges.extend([(i, i + 1), (i + 1, i)])
            gap = (terms[i + 1][1] - terms[i][1]) / 2
            gaps.extend([gap, gap])

        return Data(
            x=torch.tensor(terms, dtype=torch.float32).reshape(-1, 3),
            edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
            edge_attr=torch.tensor(gaps, dtype=torch.float32).reshape(-1, 1),
            num_nodes=len(terms),
        )
