# -*- coding: utf-8 -*-
import sys
from   math import floor

from firedrake import *
from defcon import *

from petsc4py import PETSc
from slepc4py import SLEPc

class HyperelasticityProblem(BifurcationProblem):
    def mesh(self, comm):
        return RectangleMesh(40, 40, 1, 0.1, comm=comm)

    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", 1)

        # Construct rigid body modes used in algebraic multigrid preconditioner later on
        x = SpatialCoordinate(mesh)
        rbms = [Constant((0, 1)),
                Constant((1, 0)),
                as_vector([-x[1], x[0]])]
        self.rbms = [Function(V).interpolate(rbm) for rbm in rbms]

        return V

    def parameters(self):
        eps = Constant(0)

        return [(eps, "eps", r"$\epsilon$")]

    def residual(self, u, params, v):
        B   = Constant((0.0, -1000)) # Body force per unit volume

        # Kinematics
        I = Identity(2)             # Identity tensor
        F = I + grad(u)             # Deformation gradient
        C = F.T*F                   # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        Ic = tr(C)
        J  = det(F)

        # Elasticity parameters
        E, nu = 1000000.0, 0.3
        mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

        # Total potential energy
        Energy = psi*dx - dot(B, u)*dx #- dot(T, u)*ds
        F = derivative(Energy, u, v)

        return F

    def boundary_conditions(self, V, params):
        eps = params[0]
        bcl = DirichletBC(V, Constant((0.0,  0.0)), 1)
        bcr = DirichletBC(V, Constant((-eps, 0.0)), 2)

        return [bcl, bcr]

    def functionals(self):
        def total_vertical_displacement(u, params):
            return assemble(u[1]*dx) / 0.1

        def pointeval(u, params):
            return u((0.25, 0.05))[1]

        return [(total_vertical_displacement, "total_vertical_displacement", r"$\frac{1}{|\Omega|} \int_\Omega u_1 \ \mathrm{d}x$", lambda u, params: Constant(10) * u[1]*dx),
                (pointeval, "pointeval", r"$u_1(0.25, 0.05)$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of eps.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.
        eps = params[0]
        if eps < 0.03:
            return 1
        if eps < 0.07:
            return 3
        if eps < 0.12:
            return 5
        if eps < 0.18:
            return 7
        if eps < 0.20:
            return 9
        return float("inf")

    def squared_norm(self, a, b, params):
        return inner(a - b, a - b)*dx + inner(grad(a - b), grad(a - b))*dx

    def solver(self, problem, params, solver_params, prefix="", **kwargs):
        # Set the rigid body modes for use in AMG

        s = BifurcationProblem.solver(self, problem, params, solver_params, prefix=prefix, **kwargs)
        snes = s.snes

        if snes.ksp.type != "preonly":
            # Convert rigid body modes (computed in self.function_space above) to PETSc Vec
            with self.rbms[0].dat.vec_ro as rbm_0, self.rbms[1].dat.vec_ro as rbm_1, self.rbms[2].dat.vec_ro as rbm_2:
                rbms = [rbm_0, rbm_1, rbm_2]

                # Create the PETSc nullspace
                nullsp = PETSc.NullSpace().create(vectors=rbms, constant=False, comm=snes.comm)

                (A, P) = snes.ksp.getOperators()
                A.setNearNullSpace(nullsp)
                P.setNearNullSpace(nullsp)

        return s

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 100,
               "snes_atol": 1.0e-7,
               "snes_rtol": 1.0e-10,
               "snes_max_linear_solve_fail": 100,
               "snes_linesearch_type": "l2",
               "snes_linesearch_maxstep": 1.0,
               "snes_monitor": None,
               "snes_linesearch_monitor": None,
               "snes_converged_reason": None,
               "mat_type": "aij",
               "ksp_type": "gmres",
               "ksp_monitor_cancel": None,
               "ksp_converged_reason": None,
               "ksp_max_it": 2000,
               "pc_type": "lu", # switch to "gamg" for an inexact solver
               "pc_factor_mat_solver_type": "mumps",
               "eps_type": "krylovschur",
               "eps_target": -1,
               "eps_monitor_all": None,
               "eps_converged_reason": None,
               "eps_nev": 1,
               "st_type": "sinvert",
               "st_ksp_type": "preonly",
               "st_pc_type": "lu",
               "st_pc_factor_mat_solver_type": "mumps",
               }


    def compute_stability(self, params, branchid, u, hint=None):
        V = u.function_space()
        trial = TrialFunction(V)
        test = TestFunction(V)

        bcs = self.boundary_conditions(V, params)
        comm = V.mesh().comm

        F = self.residual(u, list(map(Constant, params)), test)
        J = derivative(F, u, trial)

        A = assemble(J, bcs=bcs)
        M = assemble(inner(test,trial)*dx, bcs=bcs)

        # There must be a better way of doing this
        from firedrake.preconditioners.patch import bcdofs
        lgmap = V.dof_dset.lgmap
        for bc in bcs:
            # Ensure symmetry of M
            M.M.handle.zeroRowsColumns(lgmap.apply(bcdofs(bc)), diag=0)

        # Create the SLEPc eigensolver
        eps = SLEPc.EPS().create(comm=comm)
        eps.setOperators(A.M.handle, M.M.handle)
        eps.setWhichEigenpairs(eps.Which.SMALLEST_MAGNITUDE)
        eps.setProblemType(eps.ProblemType.GHEP)
        eps.setFromOptions()

        # If we have a hint, use it - eigenfunctions from previous solve
        if hint is not None:
            initial_space = []
            for x in hint:
                # Read only eigenfuction
                with x.dat.vec_ro as y:
                    initial_space.append(y.copy())
            eps.setInitialSpace(initial_space)

        eps.solve()
        eigenvalues = []
        eigenfunctions = []
        eigenfunction = Function(V, name="Eigenfunction")

        for i in range(eps.getConverged()):
            lmbda = eps.getEigenvalue(i)
            assert lmbda.imag == 0
            eigenvalues.append(lmbda.real)
            with eigenfunction.dat.vec_wo as x:
                eps.getEigenvector(i,x)
            eigenfunctions.append(eigenfunction.copy(deepcopy=True))

        if min(eigenvalues) < 0:
            is_stable = False
        else:
            is_stable = True

        d = {"stable": is_stable,
             "eigenvalues": eigenvalues,
             "eigenfunctions": eigenfunctions,
             "hint": eigenfunctions}

        return d

    def estimate_error(self, *args, **kwargs):
        return estimate_error_dwr(self, *args, **kwargs)


if __name__ == "__main__":
    dc = DeflatedContinuation(problem=HyperelasticityProblem(), teamsize=1, verbose=True, clear_output=True)
    params = list(arange(0.0, 0.2, 0.001)) + [0.2]
    dc.run(values={"eps": params})
