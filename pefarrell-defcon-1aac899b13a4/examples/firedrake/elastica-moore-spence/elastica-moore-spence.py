# -*- coding: utf-8 -*-
"""
Use the Moore-Spence system to calculate the first bifurcation point
in lambda for buckling of an Euler elastica, as the parameter mu varies.

The equation for the Euler elastica on the unit interval is given by:

- d^2/dx^2 theta - lambda^2 sin(theta) + mu cos(theta) = 0

with boundary conditions

theta(0) = 0 = theta(1)

where lambda and mu are parameters.
"""

import sys

from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc
from defcon import *

import matplotlib.pyplot as plt

class ElasticaMooreSpenceProblem(BifurcationProblem):
    def mesh(self, comm):
        return IntervalMesh(1000, 0, 1, comm=comm)

    def function_space(self, mesh):
        V = FunctionSpace(mesh, "CG", 1)
        R = FunctionSpace(mesh, "R", 0)
        return MixedFunctionSpace([V, R, V])

    def parameters(self):
        mu    = Constant(0)

        return [(mu, "mu", r"$\mu$")]

    # Original PDE residual
    def pde_residual(self, theta, lmbda, ttheta, params):
        mu = params[0]
        F = (
            inner(grad(theta), grad(ttheta))*dx
            - lmbda**2*sin(theta)*ttheta*dx
            + mu*cos(theta)*ttheta*dx
        )
        return F

    def residual(self, z, params, w):
        mu = params[0]
        theta, lmbda, phi = split(z)
        ttheta, tlmbda, tphi = split(w)


        # Moore-Spence system
        F1 = self.pde_residual(theta, lmbda, ttheta, params)
        F2 = derivative(self.pde_residual(theta, lmbda, tphi, params), z, as_vector([phi, 0, 0]))
        F3 = inner(dot(phi, phi) - 1, tlmbda)*dx

        F = F1 + F2 + F3
        return F

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(0), 0.0, "on_boundary"), DirichletBC(Z.sub(2), 0.0, "on_boundary")]

    def functionals(self):
        def lambda_bif(z, params):
            with z.sub(1).dat.vec_ro as x:
                myparam = x.norm()
            return myparam

        def signedL2(z, params):
            (theta, lmbda, phi) = z.subfunctions
            j = sqrt(assemble(inner(theta, theta)*dx))
            g = project(grad(theta)[0], theta.function_space())
            return j*g((0.0,))

        return [(lambda_bif, "lambda_bif", r"$\lambda$"), (signedL2, "signedL2", r"$\theta'(0) \|\theta\|$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        mu = params[0]
        x = SpatialCoordinate(Z.mesh())
        z = Function(Z)
        theta, lmbda, phi = z.subfunctions
        # Now do initial solve
        msh = Z.mesh()
        V = FunctionSpace(msh, "CG", 1)
        th = Function(V)
        tth = TestFunction(V)
        lm = Constant(3.142) # Assign an initial guess of lambda (lm)
        mu_ig = Constant(mu) # Use the given value of mu for the initial guess solve

        # Using guess for parameter lm, solve for state theta (th)
        A = self.pde_residual(th, lm, tth, params)
        bcs = [DirichletBC(V, 0.0, "on_boundary")]
        solve(A == 0, th, bcs=bcs)

        # Now solve eigenvalue problem for $F_u(u, \lambda)\phi = r\phi$
        # Want eigenmode phi with minimal eigenvalue r
        evtest, evtrial = TestFunction(V), TrialFunction(V)
        eigenproblem = LinearEigenproblem(
            A=derivative(self.pde_residual(th, lm, evtest, params), th, evtrial),
            M=inner(evtrial, evtest)*dx,
            bcs=bcs)

        # Eigensolver options
        opts = {"eps_target_magnitude": None,
                "eps_target": 0,
                "st_type": "sinvert"}

        # Set up eigensolver, asking for 1 eigenvalue, and solve
        eigensolver = LinearEigensolver(eigenproblem, n_evals=1, solver_parameters=opts)
        nconv = eigensolver.solve()

        # Extract required eigenfuncion
        ev_re, ev_im = eigensolver.eigenfunction(0)

        # Return initial guess
        theta.assign(th)
        lmbda.assign(lm)
        phi.assign(ev_re/norm(ev_re)) # Normalise the real part of the eigenvector
        return z

    def number_solutions(self, params):
        return 1 # Search only for the first bifurcation point

    def squared_norm(self, a, b, params):
        (theta, lmbda, phi) = split(a)
        (theta2, lmbda2, phi2) = split(b)
        return (
                 inner(theta - theta2, theta - theta2)*dx
               + inner(grad(theta - theta2), grad(theta - theta2))*dx
               + inner(lmbda - lmbda2, lmbda - lmbda2)*dx
               )

    def save_pvd(self, z, pvd, params):
        (theta, lmbda, phi) = z.split()
        theta.rename("theta", "theta")
        phi.rename("phi", "phi")
        pvd.write(theta, phi)

    def solver_parameters(self, params, task, **kwargs):
        return {
            "snes_max_it": 100,
            "snes_atol": 1.0e-8,
            "snes_rtol": 0.0,
            "mat_type": "matfree",
            "snes_type": "newtonls",
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_linesearch_type": "basic",
            "ksp_type": "fgmres",
            "ksp_monitor_true_residual": None,
            "ksp_max_it": 10,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_0_fields": "0,2",
            "pc_fieldsplit_1_fields": "1",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "python",
            "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
            "fieldsplit_0_assembled_pc_type": "lu",
            "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
            "fieldsplit_0_assembled_mat_mumps_icntl_14": 200,
            "mat_mumps_icntl_14": 200,
            "fieldsplit_1_ksp_type": "gmres",
            "fieldsplit_1_ksp_monitor_true_residual": None,
            "fieldsplit_1_ksp_max_it": 1,
            "fieldsplit_1_ksp_convergence_test": "skip",
            "fieldsplit_1_pc_type": "none",
         }

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=ElasticaMooreSpenceProblem(), teamsize=1, verbose=True)
    dc.run(values={"mu": linspace(0.05, 1.0, 20)})

    dc.bifurcation_diagram("lambda_bif")
    plt.title(r"First bifurcation point for buckling of an Euler elastica as $\mu$ varies")
    plt.savefig("bifurcation_moore_spence.pdf")

