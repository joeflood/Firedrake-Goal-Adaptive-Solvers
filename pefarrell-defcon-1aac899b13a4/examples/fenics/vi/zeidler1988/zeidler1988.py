# -*- coding: utf-8 -*-
# pg 320, Zeidler (1988)
from defcon import *
from dolfin import *
import numpy as np
import os

alpha = 0.4
INF = 1e20
lb = Constant((-alpha, -INF, -INF))
ub = Constant((+alpha, +INF, +INF))

class ZeidlerProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = UnitIntervalMesh(comm, 1000)
        self.mesh = mesh
        return mesh

    def function_space(self, mesh):
        Be = FiniteElement("CG", mesh.ufl_cell(), 1)
        Ce = FiniteElement("DG", mesh.ufl_cell(), 0)
        Ze = MixedElement([Be, Be, Ce])
        Z = FunctionSpace(mesh, Ze)
        return Z

    def parameters(self):
        P = Constant(0)
        a = Constant(0)
        g = Constant(0)
        rho = Constant(0)
        return [(P, "P", r"$P$"),
                (a, "a", r"$a$"),
                (g, "g", r"$g$"),
                (rho, "rho", r"$\rho$"),
                ]

    def energy(self, z, params):
        (u, v, l) = split(z)
        (P, a, g, rho) = params
        x = SpatialCoordinate(self.mesh)[0]
        w = v.dx(0)
        E = (
            + 0.5 * a * inner(w, w)*dx
            - 0.5 * P * inner(v, v)*dx
            - rho*g*u*dx
            )
        return E

    def lagrangian(self, z, params):
        (u, v, l) = split(z)
        L = (
              self.energy(z, params)
            - inner(l, v - u.dx(0))*dx
            )
        return L

    def residual(self, z, params, v):
        L = self.lagrangian(z, params)
        F = derivative(L, z, v)
        return F

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(0), 0, "on_boundary")]

    def functionals(self):
        def pointeval(z, params):
            g = z((0.25,))[0]
            return g

        def energy(z, params):
            j = assemble(self.energy(z, params))
            return j

        return [
                (pointeval, "pointeval", r"$u(\frac{1}{4})$"),
                (energy, "energy", r"$E(z)$"),
               ]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        return Function(Z)

    def number_solutions(self, params):
        return 3
        return float("inf")

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 2000,
               "snes_atol": 1.0e-10,
               "snes_rtol": 0.0,
               "snes_monitor": None,
               "snes_linesearch_type": "l2",
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_damping": 1.0,
               "snes_linesearch_monitor": None,
               "ksp_type": "preonly",
               "ksp_monitor": None,
               "ksp_rtol": 1.0e-10,
               "ksp_atol": 1.0e-10,
               "pc_type": "lu",
               "pc_factor_mat_solver_package": "umfpack",
               "pc_factor_mat_solver_type": "umfpack",
               }

    def monitorx(self, params, branchid, solution, functionals):
        x = np.linspace(0, 1, 10000)
        u = np.array([solution((x_,))[0] for x_ in x])
        import gnuplotlib as gp
        gp.plot((x, u), _with="lines", terminal="dumb 80 40", unset="grid")

    def render(self, params, branchid, solution, window):
        try:
            os.makedirs('output/figures/%2.6f' % (params[0],))
        except:
            pass

        import matplotlib.pyplot as plt
        x = np.linspace(0, 1, 10000)
        u = [solution((x_,))[0] for x_ in x]
        v = [solution((x_,))[1] for x_ in x]
        w = [solution((x_,))[2] for x_ in x]
        plt.clf()
        if window is None:
            #h = plt.figure(figsize=(10, 10))
            h = plt.figure()
        plt.plot(x, u, '-b', label=r'$u$', linewidth=2, markersize=1, markevery=1)
        plt.plot(x, [alpha]*len(x), '--r', linewidth=3)
        plt.plot(x, [-alpha]*len(x), '--r', linewidth=3)
        #plt.plot(x, v, '-r', label=r'$v$', linewidth=2, markersize=1, markevery=1)
        #plt.plot(x, w, '-g', label=r'$w$', linewidth=2, markersize=1, markevery=1)
        plt.grid()
        plt.xlabel(r'$s$')
        plt.ylabel(r'$y(s)$')
        #plt.legend(loc='best')
        plt.ylim([-1, 1])
        plt.xlim([0, 1])
        #plt.title(r'$P = %.3f$' % params[0])
        if window is not None:
            plt.show()
            plt.savefig('output/figures/%2.6f/branchid-%d.png' % (params[0], branchid))
        else:
            plt.savefig('output/figures/%2.6f/branchid-%d.pdf' % (params[0], branchid))
            plt.close(h)

    def postprocess(self, solution, params, branchid, window):
        #self.monitor(params, branchid, solution, None)
        self.render(params, branchid, solution, window)

    def bounds(self, V, params, initial_guess):
        l = interpolate(lb, V)
        u = interpolate(ub, V)
        return (l, u)

    def squared_norm(self, z1, z2, params):
        u1 = split(z1)[0]
        u2 = split(z2)[0]

        diff = u1 - u2
        return inner(diff, diff)*dx #+ inner(grad(diff), grad(diff))*dx

if __name__ == "__main__":
    problem = ZeidlerProblem()
    deflation = ShiftedDeflation(problem, power=2, shift=1)
    dc = DeflatedContinuation(problem=problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True, profile=False, continue_backwards=True, logfiles=False)
    dc.run(values=dict(P=10.4, g=-1, a=1, rho=1))
