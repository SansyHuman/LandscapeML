from common.sci_parser import *

x = Term(3.5, Monomial('x', 1.2), Monomial('x', 1.5), Monomial('y', 2.2))
print(x)
y = Term(-2.4, Monomial('y', 1.2), Monomial('z', 3.2))
print(y)

xpy = Polynomial(x, y)
print(xpy)

"""
test_poly = to_poly('t^2.099+t^2.765+t^3.786+2*t^3.815+2*t^3.844+2*t^3.872+t^4.197+t^4.864+t^4.922+t^5.531+t^5.884+2*t^5.913+t^5.942-3*t^6.-2*t^6.029-t^6.058+t^6.296+t^6.551+2*t^6.58+2*t^6.609+2*t^6.638+t^6.962-2*t^7.049-2*t^7.078-2*t^7.107-t^7.136+t^7.571+2*t^7.6+5*t^7.629+4*t^7.658+5*t^7.687+2*t^7.716+2*t^7.745+t^7.983+2*t^8.012+t^8.041-3*t^8.099-2*t^8.128+t^8.296+t^8.394+t^8.649+2*t^8.678+t^8.707-3*t^8.765-2*t^8.794-2*t^8.823-t^4.078/y-t^6.177/y+t^7.864/y+t^7.98/y-t^8.275/y+t^8.884/y+(2*t^8.913)/y+(2*t^8.942)/y+(2*t^8.971)/y-t^4.078*y-t^6.177*y+t^7.864*y+t^7.98*y-t^8.275*y+t^8.884*y+2*t^8.913*y+2*t^8.942*y+2*t^8.971*y')
print(test_poly)

terms_spectrum = test_poly.find_with(Monomial('y', 0))
print([str(term) for term in terms_spectrum])
terms_relevant = filter(lambda term: term.exponent('t') < 6, terms_spectrum)
"""

sci = SuperConformalIndex('t^2.099+t^2.765+t^3.786+2*t^3.815+2*t^3.844+2*t^3.872+t^4.197+t^4.864+t^4.922+t^5.531+t^5.884+2*t^5.913+t^5.942-3*t^6.-2*t^6.029-t^6.058+t^6.296+t^6.551+2*t^6.58+2*t^6.609+2*t^6.638+t^6.962-2*t^7.049-2*t^7.078-2*t^7.107-t^7.136+t^7.571+2*t^7.6+5*t^7.629+4*t^7.658+5*t^7.687+2*t^7.716+2*t^7.745+t^7.983+2*t^8.012+t^8.041-3*t^8.099-2*t^8.128+t^8.296+t^8.394+t^8.649+2*t^8.678+t^8.707-3*t^8.765-2*t^8.794-2*t^8.823-t^4.078/y-t^6.177/y+t^7.864/y+t^7.98/y-t^8.275/y+t^8.884/y+(2*t^8.913)/y+(2*t^8.942)/y+(2*t^8.971)/y-t^4.078*y-t^6.177*y+t^7.864*y+t^7.98*y-t^8.275*y+t^8.884*y+2*t^8.913*y+2*t^8.942*y+2*t^8.971*y')
print(f'Full index: {sci.index}')
print(f'Short index: {sci.short_index}')
print(f'marginal-flavor: {sci.num_dim3_minus_f}')
print(f'total number of relevant operators: {sci.num_relevant_ops}')
print(f'dimensions of relevant operators: {sci.relevant_dims}')
print(f'dimensions of entire operators: {sci.dims}')
print(f'relevant spectrum: {sci.relevant_spectrum}')
print(f'smallest dimension: {sci.smallest_dim}')
print(f'fermion dimensions: {sci.fermion_dims}')
print(f'fermion spectrum: {sci.fermion_spectrum}')
print(f'feature vector: {sci.featurize_dimensions(np.arange(2, 9 + 0.05, 0.05), 0.1)}')