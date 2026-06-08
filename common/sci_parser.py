from common.utils import *

marginal = Monomial('t', 6)
spectral = Monomial('y', 0)


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
        # # of marginal operators - rank of IR flavor symmetry
        self.num_dim3_minus_f = 0
        marginal_term = self.index.find_with(marginal)
        if len(marginal_term) > 0:
            self.num_dim3_minus_f = round(self.index.find_with(marginal)[0].coefficient)

        # dimensions of operators in the order of increasing
        self.dims: list[float] = []
        # the operator's dimensions and coefficient of the sci terms
        self.spectrum: dict[float, int] = dict()
        # dimensions of relevant operators in the order of increasing
        self.relevant_dims: list[float] = []
        # the number of operators with each dimensions
        self.relevant_spectrum: dict[float, int] = dict()
        # total number of relevant operators
        self.num_relevant_ops = 0

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
        self.smallest_dim = self.relevant_dims[0]

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
