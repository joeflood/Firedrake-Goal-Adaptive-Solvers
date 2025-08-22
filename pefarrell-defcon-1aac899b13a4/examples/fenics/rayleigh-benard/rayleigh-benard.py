# -*- coding: utf-8 -*-
from defcon import BifurcationProblem, DeflatedContinuation
from dolfin import (
    RectangleMesh, VectorElement, FiniteElement, MixedElement,
    FunctionSpace, Constant, split, as_vector, Point, triangle,
    inner, grad, div, dx, DirichletBC, dot, assemble,
    Expression, interpolate
    )
import matplotlib.pyplot as plt


class RayleighBenardProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = RectangleMesh(comm, Point(0, 0), Point(5, 1), 50, 50)
        return mesh

    def function_space(self, mesh):
        Ve = VectorElement("CG", triangle, 2)
        Qe = FiniteElement("CG", triangle, 1)
        Te = FiniteElement("CG", triangle, 1)
        Ze = MixedElement([Ve, Qe, Te])
        Z  = FunctionSpace(mesh, Ze)
        return Z

    def parameters(self):
        Ra = Constant(0)
        Pr = Constant(0)
        return [
                (Ra, "Ra", r"$\mathrm{Ra}$"),
                (Pr, "Pr", r"$\mathrm{Pr}$")
               ]

    def residual(self, z, params, w):
        (Ra, Pr)  = params
        (u, p, T) = split(z)
        (v, q, S) = split(w)

        g = as_vector([0, 1])

        F = (
              inner(grad(u), grad(v))*dx
            + inner(dot(grad(u), u), v)*dx
            - inner(p, div(v))*dx
            - Ra*Pr*inner(T*g, v)*dx
            + inner(div(u), q)*dx
            + inner(dot(grad(T), u), S)*dx
            + 1/Pr * inner(grad(T), grad(S))*dx
            )

        return F

    def boundary_conditions(self, Z, params):
        bcs = [
               DirichletBC(Z.sub(0), (0, 0), "on_boundary"),
               DirichletBC(Z.sub(2), 1, "near(x[1], 0.0)"),
               DirichletBC(Z.sub(2), 0, "near(x[1], 1.0)"),
               DirichletBC(Z.sub(1), 0, "x[0] == 0.0 && x[1] == 0.0", "pointwise")
              ]
        return bcs

    def functionals(self):
        def sqL2(z, params):
            (u, p, T) = split(z)
            j = assemble(inner(u, u)*dx)
            return j

        return [(sqL2, "sqL2", r"$\|u\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        comm = Z.mesh().mpi_comm()
        guess = Expression(("0.09*+sin(4*pi*x[0])*sin(3*pi*x[1])", "0.17*-sin(5.5*pi*x[0])*sin(2*pi*x[1])", "5800*x[1]", "1 - x[1]"), degree=5, mpi_comm=comm)
        out = interpolate(guess, Z)
        return out

    def number_solutions(self, params):
        (Ra, Pr) = params
        if Ra < 1700:
            return 1
        if Ra < 1720:
            return 3
        return float("inf")

    def squared_norm(self, z, w, params):
        (zu, zp, zT) = split(z)
        (wu, wp, wT) = split(w)
        diffu = zu - wu
        diffp = zp - wp
        diffT = zT - wT
        return (
            inner(diffu, diffu)*dx + inner(grad(diffu), grad(diffu))*dx
            + inner(diffp, diffp)*dx + inner(diffT, diffT)*dx
            )

    def save_pvd(self, z, pvd, params):
        (u, p, T) = z.split()
        u.rename("Velocity", "Velocity")
        p.rename("Pressure", "Pressure")
        T.rename("Temperature", "Temperature")
        pvd << u

    def solver_parameters_fieldsplit(self, params, task):
        solver_params = {
            "snes_max_it": 100,
            "snes_atol": 1.0e-9,
            "snes_rtol": 0.0,
            "snes_monitor": None,
            "snes_converged_reason": None,
            "ksp_type": "fgmres",
            "ksp_gmres_restart": 100,
            "ksp_converged_reason": None,
            "ksp_monitor": None,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative",
            "pc_fieldsplit_0_fields": "0,1",
            "pc_fieldsplit_1_fields": "2",
            "fieldsplit_0_ksp_type": "fgmres",
            "fieldsplit_0_ksp_gmres_modifiedgramschmidt": True,
            "fieldsplit_0_ksp_rtol": 1e-4,
            "fieldsplit_0_pc_type": "fieldsplit",
            "fieldsplit_0_pc_fieldsplit_type": "schur",
            "fieldsplit_0_pc_fieldsplit_schur_fact_type": "lower",
            "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_fieldsplit_0_pc_type": "lu",
            "fieldsplit_0_fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_0_fieldsplit_1_pc_type": "none",
            "fieldsplit_1_ksp_type": "gmres",
            "fieldsplit_1_ksp_rtol": "1e-4",
            "fieldsplit_1_pc_type": "hypre"
        }
        return solver_params

    def solver_parameters(self, params, task, **kwargs):
        solver_params = {
            "snes_max_it": 100,
            "snes_atol": 1.0e-9,
            "snes_rtol": 0.0,
            "snes_monitor": None,
            "snes_converged_reason": None,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        }
        return solver_params



if __name__ == "__main__":
    dc = DeflatedContinuation(problem=RayleighBenardProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"Ra": [1701], "Pr": [6.8]}, freeparam="Ra")
