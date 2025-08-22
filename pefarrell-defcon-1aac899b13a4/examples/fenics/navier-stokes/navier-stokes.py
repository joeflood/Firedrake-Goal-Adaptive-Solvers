# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

import matplotlib.pyplot as plt

class NavierStokesProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = Mesh(comm, "mesh/mesh.xml.gz")
        return mesh

    def function_space(self, mesh):
        Ve = VectorElement("CG", triangle, 2)
        Qe = FiniteElement("CG", triangle, 1)
        Ze = MixedElement([Ve, Qe])
        Z  = FunctionSpace(mesh, Ze)
        return Z

    def parameters(self):
        Re = Constant(0)
        return [(Re, "Re", r"$\mathrm{Re}$")]

    def residual(self, z, params, w):
        (u, p) = split(z)
        (v, q) = split(w)

        Re = params[0]
        mesh = z.function_space().mesh()

        # build the surface labeling we'll use in the
        # residual definition
        colours = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        colours.set_all(0)

        class Outflow(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], 150.0)
        Outflow().mark(colours, 1)
        ds_outflow = ds(subdomain_data=colours)(1)

        n = FacetNormal(mesh)

        F = (
              1.0/Re * inner(grad(u), grad(v))*dx
            + inner(grad(u)*u, v)*dx
            - div(v)*p*dx
            + q*div(u)*dx
            )

        return F

    def boundary_conditions(self, Z, params):
        # Inlet BC
        poiseuille = Expression(("-(x[1] + 1) * (x[1] - 1)", "0.0"), degree=1, mpi_comm=Z.mesh().mpi_comm())
        def inflow(x, on_boundary):
          return on_boundary and near(x[0], 0.0)
        bc_inflow = DirichletBC(Z.sub(0), poiseuille, inflow)

        # Wall BC
        def wall(x, on_boundary):
          return on_boundary and not near(x[0], 0.0) and not near(x[0], 150.0)
        bc_wall = DirichletBC(Z.sub(0), (0, 0), wall)

        bcs = [bc_inflow, bc_wall]
        return bcs

    def functionals(self):
        def sqL2(z, params):
            (u, p) = split(z)
            j = assemble(inner(u, u)*dx)
            return j

        return [(sqL2, "sqL2", r"$\|u\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        return Function(Z)

    def number_solutions(self, params):
        Re = params[0]
        if   Re < 18:  return 1
        elif Re < 41:  return 3
        elif Re < 75:  return 5
        elif Re < 100: return 8
        else:          return float("inf")

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 50,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_monitor": None,
               "snes_converged_reason": None,
               "ksp_type": "preonly",
               "pc_type": "lu",
               "pc_factor_mat_solver_package": "mumps",
               "pc_factor_mat_solver_type": "mumps",
               "ksp_converged_reason": None,
               }

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=NavierStokesProblem(), teamsize=1, verbose=True)
    dc.run(values={"Re": linspace(10.0, 100.0, 181)})

    dc.bifurcation_diagram("sqL2")
    plt.title(r"Bifurcation diagram for sudden expansion in a channel")
    plt.savefig("bifurcation.pdf")
