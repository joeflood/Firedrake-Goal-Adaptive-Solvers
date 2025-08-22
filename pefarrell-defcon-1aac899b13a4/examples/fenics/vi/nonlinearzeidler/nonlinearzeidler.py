# -*- coding: utf-8 -*-
# pg 320, Zeidler (1988)
from defcon import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import gnuplotlib as gp
import os

alpha = 0.75
INF = 1e20
lb = Constant((-alpha, -INF, -INF, -INF, -INF))
ub = Constant((+alpha, +INF, +INF, +INF, +INF))

class ZeidlerProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = UnitIntervalMesh(comm, 200)
        self.mesh = mesh
        return mesh

    def function_space(self, mesh):
        Be = FiniteElement("CG", mesh.ufl_cell(), 1)
        Ce = FiniteElement("DG", mesh.ufl_cell(), 0)
        Ze = MixedElement([Be, Be, Ce, Ce, Ce])
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
        (u, v, w, l1, l2) = split(z)
        (P, a, g, rho) = params
        x = SpatialCoordinate(self.mesh)[0]
        E = (
            + 0.5 * a * inner(w, w)*dx
            + 0.5 * P * cos(v)*dx
            - rho*g*u*dx
            )
        return E

    def lagrangian(self, z, params):
        (u, v, w, l1, l2) = split(z)
        L = (
              self.energy(z, params)
            - inner(l1, sin(v) - u.dx(0))*dx
            - inner(l2, w - v.dx(0))*dx
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

        def length(z, params):
            (u, v) = split(z)[:2]
            j = assemble(sqrt(u.dx(0)**2 + cos(v)**2)*dx)
            return j

        def energy(z, params):
            j = assemble(self.energy(z, params))
            return j

        return [
                (pointeval, "pointeval", r"$u(\frac{1}{4})$"),
                (length, "length", r"length"),
                (energy, "energy", r"$E(z)$"),
               ]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        g = Expression(("3*x[0]*(x[0]-1)", "0", "0", "0", "0"), element=Z.ufl_element(), mpi_comm=Z.mesh().mpi_comm())
        return interpolate(g, Z)

    def number_solutions(self, params):
        #if params[0] < 9.5: return 1
        #return 3
        return float("inf")

    def solver_parameters(self, params, task, **kwargs):
        # Use damping = 1 for first go

        if isinstance(task, ContinuationTask):
            damping = 1
        else:
            if hasattr(self, "_called"):
                damping = 1
            else:
                damping = 1

        self._called = True

        return {
               "snes_max_it": 200,
               "snes_atol": 1.0e-9,
               "snes_rtol": 1.0e-9,
               "snes_monitor": None,
               "snes_linesearch_type": "basic",
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_damping": damping,
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
        gp.plot((x, u), _with="lines", terminal="dumb 80 40", unset="grid")

    def render(self, params, branchid, solution, window):
        try:
            os.makedirs('output/figures/%2.6f' % (params[0],))
        except:
            pass

        x = np.linspace(0, 1, 2000)
        u = [solution((x_,))[0] for x_ in x]
        v = [solution((x_,))[1] for x_ in x]
        w = [solution((x_,))[2] for x_ in x]
        plt.clf()
        if window is None:
            h = plt.figure(figsize=(10, 10))
        plt.plot(x, u, '-b', label=r'$u$', linewidth=2, markersize=1, markevery=1)
        plt.plot(x, [alpha]*len(x), '--r', linewidth=3)
        plt.plot(x, [-alpha]*len(x), '--r', linewidth=3)
        #plt.plot(x, v, '-r', label=r'$v$', linewidth=2, markersize=1, markevery=1)
        #plt.plot(x, w, '-g', label=r'$w$', linewidth=2, markersize=1, markevery=1)
        plt.grid()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x)$')
        #plt.legend(loc='best')
        plt.ylim([-1, 1])
        plt.xlim([0, 1])
        plt.title(r'$P = %.3f$' % params[0])
        if window is not None:
            plt.show()
            plt.savefig('output/figures/%2.6f/branchid-%d.png' % (params[0], branchid))
        else:
            plt.savefig('output/figures/%2.6f/branchid-%d.png' % (params[0], branchid))
            plt.close(h)

    def postprocess(self, solution, params, branchid, window):
        if float(params[0]) not in [11.0] and window is None:
            return
        self.render(params, branchid, solution, window)

    def boundsx(self, V, params, initial_guess):
        l = interpolate(lb, V)
        u = interpolate(ub, V)
        return (l, u)

    def squared_norm(self, z1, z2, params):
        u1 = split(z1)[0]
        u2 = split(z2)[0]

        diff = u1 - u2
        return inner(diff, diff)*dx + inner(grad(diff), grad(diff))*dx

if __name__ == "__main__":
    problem = ZeidlerProblem()
    dc = DeflatedContinuation(problem=problem, teamsize=1, verbose=True, clear_output=True, profile=False, continue_backwards=True, logfiles=False)
    dc.run(values=dict(P=linspace(0, 20, 401), g=-1, a=1, rho=1), freeparam="P")
    #dc.run(values=dict(P=10.4, g=-1, a=1, rho=1))
