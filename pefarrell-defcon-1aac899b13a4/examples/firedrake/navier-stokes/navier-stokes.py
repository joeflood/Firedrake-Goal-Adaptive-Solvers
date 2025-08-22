# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *

import matplotlib.pyplot as plt

class NavierStokesProblem(BifurcationProblem):
    def mesh(self, comm):
        # Markers: 10 = inlet, 11 = outlet, 12 = wall
        mesh = Mesh('mesh/pipe.msh', comm=comm)
        return mesh

    def function_space(self, mesh):
        self.V = VectorFunctionSpace(mesh, "CG", 2)
        self.Q = FunctionSpace(mesh, "CG", 1)
        Z = MixedFunctionSpace([self.V, self.Q])
        return Z

    def parameters(self):
        Re = Constant(0)
        return [(Re, "Re", r"$\mathrm{Re}$")]

    def residual(self, z, params, w):
        (u, p) = split(z)
        (v, q) = split(w)

        Re = params[0]
        mesh = z.function_space().mesh()

        F = (
              1.0/Re * inner(grad(u), grad(v))*dx
            + inner(grad(u)*u, v)*dx
            - div(v)*p*dx
            + q*div(u)*dx
            )

        return F

    def boundary_conditions(self, Z, params):
        # Inlet BC
        x = SpatialCoordinate(Z.mesh())
        poiseuille = as_vector([-(x[1] + 1) * (x[1] - 1), 0.0])
        bc_inflow = DirichletBC(Z.sub(0), poiseuille, 10)

        # Wall
        bc_wall = DirichletBC(Z.sub(0), (0, 0), 12)

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

    def save_pvd(self, rc, pvd, params):
        (u, p) = rc.split()
        u.rename("Velocity", "Velocity")
        p.rename("Pressure", "Pressure")
        pvd.write(u, p)
    
    def solver_parameters(self, params, task, **kwargs):
        params = {
                "mat_type": "aij",
                "snes_monitor": None,
                "snes_linesearch_type": "basic",
                "snes_max_it": 100,
                "snes_atol": 1.0e-9,
                "snes_rtol": 0.0,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
        return params

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=NavierStokesProblem(), teamsize=1, verbose=True)
    dc.run(values={"Re": linspace(10.0, 100.0, 181)})

    dc.bifurcation_diagram("sqL2")
    plt.title(r"Bifurcation diagram for sudden expansion in a channel")
    plt.savefig("bifurcation.pdf")
