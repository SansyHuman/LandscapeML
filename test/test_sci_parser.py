import unittest

import numpy as np
import sympy as sp
import torch
from torch_geometric.data import Batch

from common.sci_parser import SuperConformalIndex, t, y


class SuperConformalIndexTest(unittest.TestCase):
    def test_fractional_exponents_and_short_index(self):
        sci = SuperConformalIndex('t^2.172+t^2.172+(2*t^5.9)/y-3*t^6.')
        self.assertEqual(sci.index, 2*t**sp.Rational('2.172') + 2*t**sp.Rational('5.9')/y - 3*t**6)
        self.assertEqual(sci.short_index, sci.index + 3*t**6)
        self.assertEqual(sci.terms_list, [[2, 2.172, 0.0], [2, 5.9, -1.0], [-3, 6.0, 0.0]])
        self.assertEqual(sci.num_dim3_minus_f, -3)

    def test_missing_scalar_weight_from_csv_8184(self):
        # At t**8.75, CSV row 8184 has c_0=0 and c_2=-1.
        sci = SuperConformalIndex('t^2.2-t^8.75*(y^2+y^-2)')
        self.assertEqual(sci.spectrum, {1.1: 1, 4.375: 1})
        self.assertEqual(sci.dims, [1.1, 4.375])

    def test_relevant_scalar_weight_after_cancellation(self):
        sci = SuperConformalIndex('t^2-t^4*(y^2+y^-2)')
        self.assertEqual(sci.relevant_spectrum, {1.0: 1, 2.0: 1})
        self.assertEqual(sci.num_relevant_ops, 2)

    def test_minimum_after_character_subtraction(self):
        sci = SuperConformalIndex('t^2*(y^2+1+y^-2)+t^4')
        self.assertEqual(sci.spectrum, {2.0: 1})
        self.assertEqual(sci.relevant_dims, [2.0])
        self.assertEqual(sci.num_relevant_ops, 1)
        self.assertEqual(sci.smallest_dim, 2.0)

    def test_final_sign_controls_irrelevant_dimensions(self):
        positive = SuperConformalIndex('t^2-t^8-2*t^8*(y^2+y^-2)')
        negative = SuperConformalIndex('t^2+t^8+2*t^8*(y^2+y^-2)')
        self.assertEqual(positive.spectrum[4.0], 1)
        self.assertIn(4.0, positive.dims)
        self.assertEqual(negative.spectrum[4.0], -1)
        self.assertNotIn(4.0, negative.dims)

    def test_empty_scalar_dimensions(self):
        for expr in ('0', '-t^3*(y+1/y)', '-t^8', 't^2*(y^2+1+y^-2)'):
            with self.subTest(expr=expr):
                sci = SuperConformalIndex(expr)
                self.assertEqual(sci.dims, [])
                self.assertIsNone(sci.smallest_dim)
                self.assertEqual(sci.num_relevant_ops, 0)
                np.testing.assert_array_equal(sci.featurize_dimensions(np.arange(5.), 0.1), np.zeros(21))
        self.assertEqual(SuperConformalIndex('-t^3*(y+1/y)').fermion_spectrum, {1.5: 1})

    def test_identity_is_not_a_relevant_operator(self):
        sci = SuperConformalIndex('1+t^2')
        self.assertEqual(sci.short_index, 1+t**2)
        self.assertEqual(sci.spectrum, {1.0: 1})
        self.assertEqual(sci.num_relevant_ops, 1)
        self.assertEqual(sci.smallest_dim, 1.0)

    def test_integer_coefficients_stay_exact(self):
        n = 9007199254740993
        sci = SuperConformalIndex(f'{n}*t^2-{n}*t^6')
        self.assertEqual(sci.spectrum[1.0], n)
        self.assertEqual(sci.num_relevant_ops, n)
        self.assertEqual(sci.num_dim3_minus_f, -n)

    def test_higher_spins_do_not_contaminate_lower_spins(self):
        sci = SuperConformalIndex(
            't^2-2*t^5*(y+y^-1)-t^8*(y^3+y+y^-1+y^-3)'
            '+3*t^7*(y^2+1+y^-2)+t^9*(y^4+y^2+1+y^-2+y^-4)'
        )
        self.assertEqual(sci.spectrum, {1.0: 1})
        self.assertEqual(sci.fermion_spectrum, {2.5: 2})
        self.assertEqual(sci.boson_spectrum, {3.5: 3})

    def test_extra_fugacities_and_nonmonomials_are_rejected(self):
        for expr in ('t^2*u', 't^2/(1+y)', 'sin(t)', 't/0'):
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                SuperConformalIndex(expr)

    def test_graph_bounds_and_edges(self):
        sci = SuperConformalIndex('t^2+2*t^3/y+3*t^4*y+t^6')
        graph = sci.featurize_sci_graph(1.5, 2.0)
        torch.testing.assert_close(graph.x, torch.tensor([[2., 3., -1.], [3., 4., 1.]]))
        torch.testing.assert_close(graph.edge_index, torch.tensor([[0, 1], [1, 0]]))
        torch.testing.assert_close(graph.edge_attr, torch.tensor([[0.5], [0.5]]))

    def test_empty_and_single_node_graphs_can_be_batched(self):
        sci = SuperConformalIndex('t^2')
        empty = sci.featurize_sci_graph(2.0, 3.0)
        single = sci.featurize_sci_graph(1.0, 1.0)
        zero = SuperConformalIndex('0').featurize_sci_graph(0, 3)
        for graph, count in ((empty, 0), (single, 1), (zero, 0)):
            self.assertEqual(graph.num_nodes, count)
            self.assertEqual(tuple(graph.x.shape), (count, 3))
            self.assertEqual(tuple(graph.edge_index.shape), (2, 0))
            self.assertEqual(tuple(graph.edge_attr.shape), (0, 1))
        batch = Batch.from_data_list([empty, single, zero])
        self.assertEqual(batch.num_graphs, 3)
        self.assertEqual(batch.num_nodes, 1)
        with self.assertRaises(ValueError):
            sci.featurize_sci_graph(3, 2)

    def test_numerical_features_match_gaussian_sums(self):
        sci = SuperConformalIndex('2*t^2+t^4-2*t^6-t^8')
        grid = np.linspace(0, 5, 101)
        weights = {
            'featurize_dimensions': {1: 1, 2: 1, 3: 1},
            'featurize_relevant_spectrum': {1: 2, 2: 1},
            'featurize_sci': {1: 2, 2: 1, 3: -2, 4: -1},
        }
        for method, spectrum in weights.items():
            with self.subTest(method=method):
                actual = getattr(sci, method)(grid, 0.1)
                expected = sum(c*np.exp(-0.5*((grid-d)/0.1)**2) for d, c in spectrum.items())
                if method == 'featurize_dimensions':
                    expected /= expected.sum()*(grid[1]-grid[0])+1e-12
                self.assertEqual(actual.shape, (117,))
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual[:101], expected, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
