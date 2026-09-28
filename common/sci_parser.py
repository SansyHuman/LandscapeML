import re
from fractions import Fraction
from functools import cached_property

from common.utils import *

from torch_geometric.data import Data
import numpy as np
import sympy as sp
import torch


t, y = sp.symbols("t y")

# Match complete terms at a cursor: exponent signs must not split terms.
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_T_POWER = rf"(?:{_NUMBER}|\(\s*(?:[+-]?\d+\s*/\s*\d+|{_NUMBER})\s*\))"
_Y_POWER = r"(?:[+-]?\d+|\(\s*[+-]?\d+\s*\))"
_INDEX_TERM = re.compile(
    rf"""
    \s*(?P<sign>[+-]?)\s*
    (?:
        (?P<wrapped>\()?\s*
        (?(wrapped)(?P<inner_sign>[+-]?)\s*)
        (?:(?P<coefficient>\d+)\s*\*\s*)?t
        (?:\s*(?:\^|\*\*)\s*(?P<t_power>{_T_POWER}))?
        (?(wrapped)\s*\))
        (?:\s*(?P<y_operator>[*/])\s*y
            (?:\s*(?:\^|\*\*)\s*(?P<y_power>{_Y_POWER}))?
        )?
        |(?P<constant>\d+)
    )\s*
    """,
    re.VERBOSE | re.ASCII,
)


class SuperConformalIndex:
    """
    Class which contains the information of a superconformal index.
    """
    def __init__(self, index: str) -> None:
        """Parse a reduced, unrefined index with integer coefficients.

        The input must be a flat sum of integer-coefficient terms t^a,
        t^a*y^b, or t^a/y^b. Numerator parentheses such as (2*t^a)/y,
        constants, and ** notation are supported. Exponents a may be decimal,
        scientific-notation, or parenthesized rational literals; b is integer.
        Products involving sums and additional variables are rejected.

        Parsing and spectrum extraction use exact integers and fractions.
        The symbolic index and short_index are constructed only on access.
        Spectra retain signed index contributions. For scalar
        chiral primaries, a / 2 is the scaling dimension; this interpretation
        does not apply to every multiplet contributing to the index.
        """
        self._coefficients = self._parse_index(index)
        self.terms_list = [
            [coeff, float(t_exp), float(y_exp)]
            for (t_exp, y_exp), coeff in sorted(self._coefficients.items())
        ]
        self.terms_list.sort(key=lambda row: (row[1], row[2]))
        # Marginal operators minus the dimension of the IR flavor symmetry.
        self.num_dim3_minus_f = self._coefficients.get((Fraction(6), 0), 0)

        # In an SU(2) character expansion, the spin-j coefficient is
        # [y**(2*j)] I - [y**(2*j+2)] I. Subtract before extracting powers of t:
        # the first coefficient may vanish even when the difference is nonzero.
        scalar = self._extract_spectrum(self._coefficients, 0, 2)
        fermion = self._extract_spectrum(self._coefficients, 3, 1)
        boson = self._extract_spectrum(self._coefficients, 2, 4)

        # Keep exact dimensions through subtraction and cutoff comparisons;
        # public fields retain their existing Python float/int representation.
        self.spectrum = {float(dim): cnt for dim, cnt in scalar.items()}
        self.dims = [
            float(dim) for dim, cnt in scalar.items()
            if dim < 3 or cnt > 0
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
    def _parse_index(index: str) -> dict[tuple[Fraction, int], int]:
        """Collect flat monomials without evaluating symbolic expressions."""
        if not isinstance(index, str) or not index.strip():
            raise ValueError("Index must be a nonempty string; use '0' for zero.")

        coefficients = {}
        position = 0
        while position < len(index):
            match = _INDEX_TERM.match(index, position)
            if match is None or (position and not match["sign"]):
                raise ValueError(
                    f"Unsupported index syntax at position {position}: "
                    f"{index[position:position + 40]!r}"
                )
            sign = -1 if match["sign"] == "-" else 1
            if match["constant"] is not None:
                t_exp, y_exp = Fraction(0), 0
                coeff = int(match["constant"])
            else:
                if match["inner_sign"] == "-":
                    sign = -sign
                coeff = int(match["coefficient"] or "1")
                power = match["t_power"] or "1"
                if power.startswith("("):
                    power = "".join(power[1:-1].split())
                try:
                    t_exp = Fraction(power)
                except (ValueError, ZeroDivisionError) as error:
                    raise ValueError(f"Invalid t exponent: {power!r}") from error

                y_exp = 0
                if match["y_operator"] is not None:
                    power = match["y_power"] or "1"
                    y_exp = int("".join(power.strip("()").split()))
                    if match["y_operator"] == "/":
                        y_exp = -y_exp

            key = (t_exp, y_exp)
            coefficients[key] = coefficients.get(key, 0) + sign * coeff
            position = match.end()

        return {key: coeff for key, coeff in coefficients.items() if coeff != 0}

    @cached_property
    def index(self) -> sp.Expr:
        """Full symbolic expression, built lazily for API compatibility."""
        return self._symbolic_index(self._coefficients.items())

    @cached_property
    def short_index(self) -> sp.Expr:
        """Symbolic terms with exact t exponent < 6, built only on access."""
        return self._symbolic_index(
            (key, coeff) for key, coeff in self._coefficients.items() if key[0] < 6
        )

    @staticmethod
    def _symbolic_index(terms) -> sp.Expr:
        return sp.Add(*(
            coeff * t**sp.Rational(a.numerator, a.denominator) * y**b
            for (a, b), coeff in terms
        ))

    @staticmethod
    def _extract_spectrum(
        coefficients: dict[tuple[Fraction, int], int],
        positive_weight: int,
        negative_weight: int,
    ) -> dict[Fraction, int]:
        """Subtract character weights at exact half-exponents, excluding t**0."""
        spectrum = {}
        for (exponent, weight), coeff in coefficients.items():
            if weight == negative_weight:
                coeff = -coeff
            elif weight != positive_weight:
                continue
            # The identity is not an operator counted in the reduced spectrum.
            if exponent == 0:
                continue
            dim = exponent / 2
            spectrum[dim] = spectrum.get(dim, 0) + coeff
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
