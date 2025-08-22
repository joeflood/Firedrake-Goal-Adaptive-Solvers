from __future__ import absolute_import

import defcon.backend as backend
from ufl import product
from defcon.backend import Function, Vector, assemble, derivative

__all__ = ['DeflationOperator', 'ShiftedDeflation']

class DeflationOperator(object):
    """
    Base class for deflation operators.
    """
    def set_parameters(self, params):
        self.parameters = params

    def deflate(self, roots):
        self.roots = roots

    def evaluate(self):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError

class ShiftedDeflation(DeflationOperator):
    """
    The shifted deflation operator presented in doi:10.1137/140984798.
    """
    def __init__(self, problem, power, shift):
        self.problem = problem
        self.power = power
        self.shift = shift
        self.roots = []

    def normsq(self, y, root):
        return self.problem.squared_norm(y, root, self.parameters)

    def evaluate(self, y):
        m = 1.0
        for root in self.roots:
            normsq = assemble(self.normsq(y, root))
            factor = normsq**(-self.power/2.0) + self.shift
            m *= factor

        return m

    def derivative(self, y):
        if len(self.roots) == 0:
            deta = Function(y.function_space()).vector()
            return deta

        p = self.power
        factors  = []
        dfactors = []
        dnormsqs = []
        normsqs  = []

        for root in self.roots:
            form = self.normsq(y, root)
            normsqs.append(assemble(form))
            dnormsqs.append(assemble(derivative(form, y)))

        for normsq in normsqs:
            factor = normsq**(-p/2.0) + self.shift
            dfactor = (-p/2.0) * normsq**((-p/2.0) - 1.0)

            factors.append(factor)
            dfactors.append(dfactor)

        eta = product(factors)

        deta = Function(y.function_space()).vector()

        for (solution, factor, dfactor, dnormsq) in zip(self.roots, factors, dfactors, dnormsqs):
            if backend.__name__ == "firedrake":
                dnormsq = dnormsq.vector()
            deta.axpy(float((eta/factor)*dfactor), dnormsq)

        return deta

