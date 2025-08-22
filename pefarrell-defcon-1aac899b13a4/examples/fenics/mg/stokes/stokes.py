from defcon import *
from dolfin import *

from defcon.backendimporter import get_deep_submat

from petsc4py import PETSc

class StokesProblem(BifurcationProblem):
    def mesh(self, comm):
        self.comm = comm

        mesh = Mesh(comm, "mesh/canal.xml.gz")
        facets = MeshFunction("size_t", mesh, "mesh/facets.xml.gz")
        self.facets = facets
        return mesh

    def coarse_meshes(self, comm):
        return [Mesh(comm, "mesh/canal.1.xml.gz")]

    def function_space(self, mesh):
        Ve = VectorElement("CG", mesh.ufl_cell(), 2)
        Qe = FiniteElement("CG", mesh.ufl_cell(), 1)
        Ze = MixedElement([Ve, Qe])
        Z  = FunctionSpace(mesh, Ze)
        self.Z = Z
        return Z

    def parameters(self):
        # Dummy parameter, not used
        f = Constant(0)
        return [(f, "f", r"$f$")]

    def residual(self, z, params, w):
        (u, p) = split(z)
        (v, q) = split(w)

        F = (
              inner(grad(u), grad(v)) * dx
            - div(v) * p * dx
            - q * div(u) * dx
            )

        return F

    def boundary_conditions(self, Z, params):
        facets = self.facets

        zerovector = Constant((0.0, 0.0))
        inflow = Expression(("0.25 * (4 - x[1] * x[1])", "0.0"), mpi_comm=self.comm, degree=2)
        bcs = [DirichletBC(Z.sub(0), inflow, facets, 2),
               DirichletBC(Z.sub(0), (0, 0), facets, 1),
               DirichletBC(Z.sub(0), (0, 0), facets, 4)]
        return bcs

    def functionals(self):
        def L2(z, params):
            (u, _) = split(z)
            j = sqrt(assemble(inner(u, u)*dx))
            return j

        return [(L2, "L2", r"$\|u\|$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        return 1

    def solver(self, problem, params, solver_params, prefix="", **kwargs):
        solver = SNUFLSolver(problem, prefix=prefix, solver_parameters=solver_params, **kwargs)
        snes = solver.snes
        dm = snes.dm

        if snes.ksp.pc.type == "fieldsplit":
            # Set the Schur complement approximation: use solves on the mass matrix to precondition
            # the Schur complement
            (names, ises, dms) = dm.createFieldDecomposition() # fetch subdm corresponding to pressure space

            Z = self.Z
            (u, p) = TrialFunctions(Z)
            (v, q) = TestFunctions(Z)
            form = -inner(p, q)*dx
            M = PETScMatrix(self.comm)
            assemble(form, tensor=M)
            mass = get_deep_submat(M.mat(), ises[1], ises[1]) # fetch submatrix

            ksp_mass = PETSc.KSP().create(comm=self.comm)
            ksp_mass.setDM(dms[1])
            ksp_mass.setDMActive(False) # don't try to build the operator from the DM
            ksp_mass.setOperators(mass)
            ksp_mass.setOptionsPrefix("mass_")
            ksp_mass.setFromOptions()

            class SchurApproxInv(object):
                def mult(self, mat, x, y):
                    ksp_mass.solve(x, y)
            schurpc = PETSc.Mat()
            schurpc.createPython(mass.getSizes(), SchurApproxInv(), comm=self.comm)
            schurpc.setUp()

            snes.ksp.pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.USER, schurpc)

        return solver


    def solver_parameters(self, params, task, **kwargs):
        args = {
               "snes_type": "ksponly",
               "snes_max_it": 10,
               "snes_atol": 1.0e-9,
               "snes_rtol": 1.0e-9,
               "snes_monitor": None,
               "ksp_type": "gcr",
               "ksp_monitor_true_residual": None,
               "ksp_atol": 1.0e-10,
               "ksp_rtol": 1.0e-10,
               "pc_type": "fieldsplit",
               "pc_fieldsplit_type": "schur",
               "pc_fieldsplit_schur_factorization_type": "full",
               "pc_fieldsplit_schur_precondition": "user",
               "fieldsplit_0_ksp_type": "chebyshev",
               "fieldsplit_0_ksp_max_it": 1,
               "fieldsplit_0_pc_type":  "mg",
               "fieldsplit_0_pc_mg_galerkin":  None,
               "fieldsplit_0_mg_levels_ksp_type": "chebyshev",
               "fieldsplit_0_mg_levels_ksp_max_it": 5,
               "fieldsplit_0_mg_levels_pc_type": "sor",
               "fieldsplit_1_ksp_type": "gmres",
               "fieldsplit_1_ksp_converged_reason": None,
               "fieldsplit_1_ksp_atol": 1.0e-10,
               "fieldsplit_1_ksp_rtol": 1.0e-2,
               "fieldsplit_1_pc_type":  "mat",
               "mass_ksp_type": "richardson",
               "mass_ksp_max_it": 1,
               "mass_pc_type": "mg",
               "mass_pc_mg_galerkin": None,
               }
        return args

    def save_pvd(self, z, pvd, params):
        (u, p) = z.split(deepcopy=True)
        u.rename("Velocity", "Velocity")
        p.rename("Pressure", "Pressure")
        pvd << u

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=StokesProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"f": [1.0]})
