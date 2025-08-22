# -*- coding: utf-8 -*-
# doi:10.1002/oca.4660120103
from defcon import *
from dolfin import *
import numpy as np
import os

d = 0.05

class MaurerProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = UnitIntervalMesh(comm, 100)
        self.mesh = mesh
        return mesh

    def function_space(self, mesh):
        cell = mesh.ufl_cell()
        ele_x = FiniteElement("CG", cell, 1)
        ele_theta = FiniteElement("CG", cell, 1)
        ele_lmbda = FiniteElement("DG", cell, 0)

        Ze = MixedElement([ele_x, ele_theta, ele_lmbda])
        Z = FunctionSpace(mesh, Ze)

        return Z

    def parameters(self):
        alpha = Constant(0)
        return [(alpha, "alpha", r"$\alpha$")]

    def energy(self, z, params):
        alpha = params[0]
        (x, theta, _) = split(z)

        E = (
            + 0.5 * inner(grad(theta), grad(theta)) * dx
            + alpha * cos(theta) * dx
            )

        return E

    def lagrangian(self, z, params):
        (x, theta, lmbda) = split(z)
        L = (
              self.energy(z, params)
            - inner(lmbda, x.dx(0) - sin(theta))*dx
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
            g = z((0.5,))[0]
            return g

        def energy(z, params):
            j = assemble(self.energy(z, params)) - float(params[0])
            return j

        return [
                (energy, "energy", r"$E(z, \alpha) - \alpha$"),
                (pointeval, "pointeval", r"$u(\frac{1}{2})$"),
               ]

    def number_initial_guesses(self, params):
        return 5

    def initial_guess(self, Z, params, n):
        comm = self.mesh.mpi_comm()
        #expr = Expression(("-4*d*x[0]*(x[0]-1)", "0", "0"), d=d, mpi_comm=comm, element=Z.ufl_element())
        expr = Expression(("d*sin(n*pi*x[0])", "0", "0"), d=d, n=n+1, mpi_comm=comm, element=Z.ufl_element())
        return interpolate(expr, Z)

    def trivial_solutions(self, Z, params, freeindex):
        return [Function(Z)]

    def number_solutions(self, params):
        return float("inf")

    def solver_parameters(self, params, task, **kwargs):
        # Use damping = 1 for first go
        ic = (isinstance(task, DeflationTask) and task.oldparams is None)
        if "averaging" in kwargs or ic:
            damping = 0.1
            maxit = 1000
        else:
            damping = 1
            maxit = 100

        return {
               "snes_max_it": maxit,
               "snes_max_funcs": 200000,
               "snes_atol": 1.0e-9,
               "snes_rtol": 1.0e-9,
               "snes_monitor": None,
               "snes_linesearch_type": "l2",
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
        import gnuplotlib as gp
        gp.plot((x, u), _with="lines", terminal="dumb 80 40", unset="grid")

    def render(self, params, branchid, solution, window):
        try:
            os.makedirs('output/figures/%2.6f' % (params[0],))
        except:
            pass

        j = assemble(self.energy(split(solution)[0], params)) - float(params[0])

        import matplotlib.pyplot as plt
        s = np.linspace(0, 1, 10000)
        x = [solution((s_,))[0] for s_ in s]
        theta = [solution((s_,))[1] for s_ in s]
        plt.clf()
        if window is None:
            #h = plt.figure(figsize=(10, 10))
            h = plt.figure()
        plt.plot(s, x, '-b', label=r'$u$', linewidth=2, markersize=1, markevery=1)
        plt.plot(s, [+d]*len(x), '--r', linewidth=3)
        plt.plot(s, [-d]*len(x), '--r', linewidth=3)
        #plt.plot(x, v, '-r', label=r'$v$', linewidth=2, markersize=1, markevery=1)
        #plt.plot(x, w, '-g', label=r'$w$', linewidth=2, markersize=1, markevery=1)
        plt.grid()
        plt.xlabel(r'$s$')
        plt.ylabel(r'$y(s)$')
        #plt.legend(loc='best')
        plt.ylim([-1.1*d, 1.1*d])
        plt.xlim([0, 1])
        plt.title(r'$\alpha = %3.3f, E = %3.3f$' % (params[0], j))
        if window is not None:
            plt.show()
            plt.savefig('output/figures/%2.6f/branchid-%d.png' % (params[0], branchid))
        else:
            plt.savefig('output/figures/%2.6f/branchid-%d.png' % (params[0], branchid))
            plt.close(h)

    def postprocess(self, solution, params, branchid, window):
        #self.monitor(params, branchid, solution, None)
        self.render(params, branchid, solution, window)

    def bounds(self, Z, params, initial_guess):
        inf = 1e20
        l = interpolate(Constant((-d, -inf, -inf)), Z)
        u = interpolate(Constant((+d, +inf, +inf)), Z)
        return (l, u)

    def squared_norm(self, z1, z2, params):
        u1 = split(z1)[0]
        u2 = split(z2)[0]

        diff = u1 - u2
        return inner(diff, diff)*dx #+ inner(grad(diff), grad(diff))*dx

if __name__ == "__main__":
    problem = MaurerProblem()
    deflation = ShiftedDeflation(problem, power=2, shift=1)
    dc = DeflatedContinuation(problem=problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True, profile=False, continue_backwards=True, logfiles=False)
    #dc.run(values=dict(alpha=linspace(9.88, 156.6737, 500)))
    #dc.run(values=dict(alpha=linspace(91.469, 156.673, 101)))
    dc.run(values=dict(alpha=linspace(156.673, 91.469, 101)))
