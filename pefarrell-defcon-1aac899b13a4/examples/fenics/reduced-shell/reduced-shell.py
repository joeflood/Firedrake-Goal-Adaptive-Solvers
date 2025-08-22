# -*- coding: utf-8 -*-
# From Vidoli & Maurini,
# Tristability of thin orthotropic shells with uniform initial curvature
# doi:10.1098/rspa.2008.0094

# This example shows how to do multi-parameter continuation
# (see the branch_found method of the ReducedNaghdi class).

from petsc4py import PETSc
from dolfin import *
from defcon import *

parameters.form_compiler.quadrature_degree = 4

# These two functions are borrowed from fenics-shells
def strain_to_voigt(e):
    r"""Returns the pseudo-vector in the Voigt notation associate to a 2x2
    symmetric strain tensor, according to the following rule (see e.g.
    https://en.wikipedia.org/wiki/Voigt_notation).
    """
    return as_vector((e[0,0], e[1,1], 2*e[0,1]))

def stress_from_voigt(sigma_voigt):
    r"""Inverse operation of stress_to_voigt.
    """
    return as_matrix(((sigma_voigt[0], sigma_voigt[2]), (sigma_voigt[2], sigma_voigt[1])))

class ReducedNaghdi(BifurcationProblem):

    def mesh(self, comm):
        return UnitIntervalMesh(comm, 2)

    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "R", 0, dim=3)
        return V

    def parameters(self):
        c_0 = Constant(0)
        c_I = Constant(0)
        return [(c_0, "c_0", r"$c_{0}$"),
                (c_I, "c_I", r"$c_{I}$")]

    def energy(self, u, params):
        c_0, c_I = params
        mesh = u.function_space().mesh()

        (kx, ky, kxy) = split(u)
        K = as_tensor([[kx, kxy], [kxy, ky]])

        # Geometric and material parameters (thickness)
        # Define the material parameters
        E = Constant(1.0)
        nu = Constant(0.3)
        t = Constant(1E-2)
        beta = Constant(1.)
        gamma_iso = (1-nu**2/beta)/(2*(1+nu))
        gamma = Constant(1.3)*gamma_iso # rho*(1-nu^2/beta)
        # Curvature scaling
        # This gives 1 as critical inelastic curvature in the isotropic case
        # see Meccanica paper, eq. 10-15-28
        radius = 1. # radius of the disc
        psi_factor = (1-nu**2)/(192*pi**2) # a factor depending on shape and material properties
        R = sqrt(12*psi_factor)*(pi*radius**2)/t # The  radius of curvature, required to get 1 in front of the bending energy
        cI_star = 2*sqrt(1-nu)/(1+nu) # critical load at bifurcation
        cs = cI_star #

        # Target curvature (here I scale only by cI_star, because the energy is nondimensional)
        k0x, k0y = c_0, c_0
        kIx, kIy = c_I, c_I
        k_0 = cI_star*as_tensor([[k0x, 0.0], [0.0, k0y]]) # initial curvature
        k_I = cI_star*as_tensor([[kIx, 0.0], [0.0, kIy]]) # inelastic curvature
        k_T = k_I + k_0 # target curvature
        k_eff = K - k_T # Kinematics

        # Generalized forces
        EI_eq = 1. #(E*t**3)/(12.0*(1.0 - nu**2)) #this factor here is accounted for by R above and the orm of the energy
        es = 1. # EI_eq/(cs**2) # scaling factor for the stiffness (FIXME: think better nondimensional form)
        D = EI_eq*(as_matrix([[1.,nu,0],[nu,beta,0.],[0.,0.,2*gamma]]))
        M_voigt = D*strain_to_voigt(k_eff)
        M = stress_from_voigt(M_voigt) # bending moment

        # Energy (see eq in Meccanica paper)
        Pi = 0.5*inner(M, k_eff)*dx + 0.5*((det(K) - det(k_0))**2)*dx
        return Pi

    def residual(self, u, params, u_t):
        Pi = self.energy(u, params)
        F = derivative(Pi, u, u_t)
        return F

    def boundary_conditions(self, U, params):
        return None

    def functionals(self):

        def kxx(u, params):
            return u.vector().get_local()[0]

        def kyy(u, params):
            return u.vector().get_local()[1]

        def kxy(u, params):
            return u.vector().get_local()[2]

        def energy(u, params):
            params = [Constant(x) for x in params]
            return assemble(self.energy(u, params))

        return [(kxx, "kxx", r"$K_{xx}$"),
                (kxy, "kxy", r"$K_{xy}$"),
                (kyy, "kyy", r"$K_{yy}$"),
                (energy, "energy", r"$\mathcal{E}$")]

    def compute_stability(self, params, branchid, u, hint=None):
        c0 = float(params[0])
        cI = float(params[1])

        U = u.function_space()
        u_t = TestFunction(U)
        u_ = TrialFunction(U)
        comm = U.mesh().mpi_comm()

        F = self.residual(u, [Constant(x) for x in params], u_t)
        J = derivative(F, u, u_)

        A = PETScMatrix(comm)
        asm = SystemAssembler(J, F, bcs=None)
        asm.assemble(A)

        pc = PETSc.PC().create(comm)
        pc.setOperators(A.mat())
        pc.setType("cholesky")
        pc.setFactorSolverPackage("mumps")
        pc.setUp()

        F = pc.getFactorMatrix()
        (neg, zero, pos) = F.getInertia()

        print("Inertia: (%s, %s, %s)" % (neg, zero, pos))

        expected_dim = 0

        # Nocedal & Wright, theorem 16.3
        if neg == expected_dim:
            is_stable = True
        else:
            is_stable = False

        d = {"stable": is_stable}
        return d

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        if params == (0, 0):
            return 1
        else:
            return float("inf")

    def transform_guess(self, state, task, io):
        # Perturb the guess to break the symmetry --- need to
        # get off the Z_2 symmetric manifold
        copy = state.vector().get_local()
        copy[0] *= 1.01
        state.vector().set_local(copy)

    def solver_parameters(self, params, task, **kwargs):
        return {
               "snes_max_it": 20,
               "snes_atol": 1.0e-10,
               "snes_rtol": 1.0e-10,
               "snes_monitor": None,
               "snes_converged_reason": None,
               "ksp_type": "preonly",
               "pc_type": "cholesky",
               "pc_factor_mat_solver_package": "mumps",
               "pc_factor_mat_solver_type": "mumps",
               "mat_mumps_icntl_24": 1,
               "mat_mumps_icntl_13": 1
               }

    def branch_found(self, task):
        # If we find a branch along cI = 0, continue in cI
        if task.freeindex == 0 and task.oldparams[1] == 0:
            conttask = ContinuationTask(taskid=task.taskid + 1,
                                        oldparams=task.oldparams,
                                        freeindex=1,
                                        branchid=task.branchid,
                                        newparams=None,
                                        direction=+1)
            return [conttask]
        else:
            return []


c0max = 3
cImax = 3
Nc0   = 151
NcI   = 151
c0loadings = linspace(0, c0max, Nc0)
cIloadings = linspace(0, cImax, NcI)

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=ReducedNaghdi(), teamsize=1, verbose=True, clear_output=True, logfiles=False)
    dc.run(values={"c_0": c0loadings, "c_I": cIloadings}, freeparam="c_0")
