# -*- coding: utf-8 -*-
from defcon import *
from dolfin import *
import ufl

parameters.form_compiler.quadrature_degree = 4
parameters.ghost_mode = "shared_facet"

# Borrowed from fenics-shells, to avoid dependency.
# Credit to Jack Hale, Corrado Maurini, Matteo Brunetti, Stephane Bordas

def strain_to_voigt(e):
    return as_vector((e[0,0], e[1,1], 2*e[0,1]))
def stress_from_voigt(sigma_voigt):
    return as_matrix(((sigma_voigt[0], sigma_voigt[2]), (sigma_voigt[2], sigma_voigt[1])))
def inner_e(x, y, restrict_to_one_side=False, quadrature_degree=1):
    dSp = Measure('dS', metadata={'quadrature_degree': quadrature_degree})
    dsp = Measure('ds', metadata={'quadrature_degree': quadrature_degree})
    n = ufl.geometry.FacetNormal(x.ufl_domain())
    t = as_vector((-n[1], n[0]))
    a = (inner(x, t)*inner(y, t))('+')*dSp + \
        (inner(x, t)*inner(y, t))*dsp
    if not restrict_to_one_side:
        a += (inner(x, t)*inner(y, t))('-')*dSp
    return a

# Rigid body modes
rbms = [Constant((0, 0, 1)),
        Constant((0, 1, 0)),
        Constant((1, 0, 0)),
        Expression(("-x[1]", "x[0]", "0.0"), degree=1),
        Expression(("x[2]", "0.0", "-x[0]"), degree=1),
        Expression(("0.0", "-x[2]", "x[1]"), degree=1)]

class Naghdi(BifurcationProblem):

    def __init__(self, N=15):
        self.N = N # mesh size

    def d_naghdi(self, theta):
        # Director vector
        return as_vector([sin(theta[1])*cos(theta[0]), -sin(theta[0]), cos(theta[1])*cos(theta[0])])
    def F_naghdi(self, displacement):
        # Deformation gradient
        return as_tensor([[1.0, 0.0],[0.0, 1.0],[0., 0.]]) + grad(displacement)
    def e_naghdi(self, F):
        # Stretching tensor (1st Naghdi strain measure)
        return 0.5*(F.T*F - Identity(2))
    def k_naghdi(self, F, d):
        # Curvature tensor (2nd Naghdi strain measure)
        return 0.5*(F.T*grad(d) + grad(d).T*F)
    def g_naghdi(self, F, d):
        # Shear strain vector (3rd Naghdi strain measure)
        return F.T*d

    def mesh(self, comm):
        mesh = Mesh(comm, "meshes/mesh-%d.xml.gz" % self.N)
        mesh = refine(mesh)

        return mesh

    def function_space(self, mesh):
        # In-plane displacements, rotations, out-of-plane displacements
        # shear strains and Lagrange multiplier field.
        element = MixedElement([VectorElement("Lagrange", triangle, 1, dim=3),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("N1curl", triangle, 1),
                        RestrictedElement(FiniteElement("N1curl", triangle, 1), "edge"),
                        VectorElement("R", triangle, 0, dim=6)])

        # Define the Function Space
        U = FunctionSpace(mesh, element)
        print("Degrees of freedom: ", U.dim())
        self.tfs = TensorFunctionSpace(mesh, "DG", 0)
        self.V = VectorFunctionSpace(mesh, "CG", 1, dim=3)

        self.rbms = [interpolate(rbm, self.V) for rbm in rbms]
        return U


    def parameters(self):
        c_0 = Constant(0)
        c_I = Constant(0)
        return [(c_0, "c_0", r"$c_{0}$"),
                (c_I, "c_I", r"$c_{I}$")]

    def energy(self, u, params):
        c_0, c_I = params
        mesh = u.ufl_domain().ufl_cargo()

        # Displacement u, Rotation theta, Reduced shear deformation theta, and lagrange multiplier p
        (z, theta, Rgamma, p, lmbda) = split(u)

        # Geometric and material parameters (thickness)
        # Define the material parameters
        E = Constant(1.0)
        nu = Constant(0.3)
        t = Constant(1E-2)
        #eps = Constant(0.5*thickness)
        beta = Constant(0.99)
        gamma_iso = (1-nu**2/beta)/(2*(1+nu))
        gamma = Constant(2.3)*gamma_iso # rho*(1-nu^2/beta)
        print("gamma = %s" % float(gamma))
        print("gamma_iso = %s" % float(gamma_iso))
        cs = 0.0516 # curvature scaling (FIXME: put here explicit function of t and geometry)

        # Target curvature
        k0x, k0y = c_0, c_0
        kIx, kIy = c_I, c_I
        k_0 = cs*as_tensor([[k0x, 0.0], [0.0, k0y]]) # initial curvature
        k_I = cs*as_tensor([[kIx, 0.0], [0.0, kIy]]) # inelastic curvature
        k_T = k_I + k_0 # target curvature

        # Target metric
        x = SpatialCoordinate(mesh)
        e_Tx = -k0x*k0y*x[1]**2/2.
        e_Ty = -k0x*k0y*x[0]**2/2.
        e_T = (cs**2)*as_tensor([[e_Tx, 0.0], [0.0, e_Ty]])

        # Kinematics
        F = self.F_naghdi(z)
        d = self.d_naghdi(theta)
        e_eff = self.e_naghdi(F) - e_T
        k_eff = self.k_naghdi(F, d) - k_T
        g_eff = self.g_naghdi(F, d)

        # Generlized forces
        EI_eq = (E*t**3)/(12.0*(1.0 - nu**2))
        ES_eq = (E*t)/((1.0 - nu**2))
        GS_eq = E*t/(2*(1+nu))
        es = 1 # EI_eq/(cs**2) # scaling factor for the stiffness (FIXME: think better nondimensional form)
        A = es*ES_eq*(as_matrix([[1.,nu,0],[nu,beta,0.],[0.,0.,gamma]]))
        D = es*EI_eq*(as_matrix([[1.,nu,0],[nu,beta,0.],[0.,0.,gamma]]))
        M_voigt = D*strain_to_voigt(k_eff)
        M = stress_from_voigt(M_voigt) # bending moment
        N_voigt = A*strain_to_voigt(e_eff)
        N = stress_from_voigt(N_voigt) # membrane stress
        T = es*GS_eq*g_eff # shear stress
        RT = es*GS_eq*Rgamma # reduced shear stress

        # Energies
        psi_m = .5*inner(N, e_eff) # Membrane energy density
        psi_b = .5*inner(M, k_eff) # Bending energy density
        psi_s = .5*inner(RT, Rgamma) # Shear energy density
        psi = (psi_m + psi_b + psi_s) # Stored strain energy density
        Pi = psi*dx # Total energy
        return Constant(1e5)*Pi

    def constraint(self, u, params):
        (z, theta, Rgamma, p, lmbda) = split(u)
        F = self.F_naghdi(z)
        d = self.d_naghdi(theta)
        g_eff = self.g_naghdi(F, d)
        constraint = inner_e(p, g_eff - Rgamma)

        for i in range(6):
            constraint += lmbda[i]*inner(z, self.rbms[i])*dx + lmbda[i]*inner(grad(z), grad(self.rbms[i]))*dx
        return constraint

    def lagrangian(self, u, params):
        return self.energy(u, params) + self.constraint(u, params)

    def residual(self, u, params, u_t):
        Pi = self.lagrangian(u, params)
        F = derivative(Pi, u, u_t)
        return F

    def boundary_conditions(self, U, params):
        return []

    def functionals(self):

        def K_h(u, params):
            (z_h, theta_h, Rgamma_h, p_h, lmbda_h) = u.split(deepcopy=True)
            k = self.k_naghdi(self.F_naghdi(z_h), self.d_naghdi(theta_h))
            K_h = project(k, self.tfs)
            return K_h

        def kxx(u, params):
            Kxx = assemble(K_h(u, params)[0,0]*dx)/(2*pi)
            return Kxx

        def kxy(u, params):
            Kxy = assemble(K_h(u, params)[0,1]*dx)/(2*pi)
            return Kxy

        def kyy(u, params):
            Kyy = assemble(K_h(u, params)[1,1]*dx)/(2*pi)
            return Kyy

        def energy(u, params):
            params = [Constant(x) for x in params]
            return assemble(self.energy(u, params))

        return [(kxx, "kxx", r"$K_{xx}$"),
                (kxy, "kxy", r"$K_{xy}$"),
                (kyy, "kyy", r"$K_{yy}$"),
                (energy, "energy", r"$\mathcal{E}$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        return 3

    def squared_norm(self, a, b, params):
        (az, atheta, _, _, _) = split(a)
        (bz, btheta, _, _, _) = split(b)
        return inner(az - bz, az - bz)*dx #+ inner(atheta - btheta, atheta - btheta)*dx

    def save_pvd(self, u, pvd, params):
        (z, _, _, _, _) = u.split()
        z.rename("Displacement", "Displacement")
        pvd << z

    def solver_parameters(self, params, task, **kwargs):
        if isinstance(task, DeflationTask):
            maxit = 200
            damping = 1.0
        else:
            maxit = 20
            damping = 1.0

        return {
               "snes_max_it": maxit,
               "snes_atol": 1.0e-8,
               "snes_rtol": 1.0e-8,
               "snes_monitor": None,
               "snes_converged_reason": None,
               "snes_linesearch_type": "l2",
               "snes_linesearch_damping": damping,
               "ksp_type": "preonly",
               "pc_type": "cholesky",
               "pc_factor_mat_solver_package": "mumps",
               "pc_factor_mat_solver_type": "mumps",
               "mat_mumps_icntl_24": 1,
               "mat_mumps_icntl_13": 1
               }

    def monitorx(self, params, branchid, solution, functionals):
        """Grid sequencing."""

        comm = solution.function_space().mesh().mpi_comm()
        N = 88
        _refined_u = File(comm, "refined-bounds/displacement-%d.pvd" % branchid)
        _refined_nb_u = File(comm, "refined-nobounds/displacement-%d.pvd" % branchid)

        class RefinedNaghdi(Naghdi):
            def monitor(self, *args, **kwargs):
                pass

            def initial_guess(self, Z, params, n):
                sub = solution.split(deepcopy=True)[0]
                # FIXME: use defcon's interpolator here
                sub.set_allow_extrapolation(True)
                return interpolate(sub, Z)

            def bounds(self, *args, **kwargs):
                return Naghdi.bounds(self, *args, **kwargs)

        refinedproblem = RefinedNaghdi(N)
        print("Solving refined problem (bounds) ...")
        (success, iters, z) = dcsolve(refinedproblem, params, comm=comm)
        print("Refined problem solved.")
        assert success

        refined_u = z.split(deepcopy=True)[0].split(deepcopy=True)[0]
        refined_u.rename("Displacement", "Displacement")
        _refined_u << refined_u

        File(comm, "refined-bounds/state-%d.xml.gz" % branchid) << z

        class NaghdiNoBounds(RefinedNaghdi):
            def initial_guess(self, Z, params, n):
                sub = z.split(deepcopy=True)[0]
                sub.set_allow_extrapolation(True)
                return interpolate(sub, Z)

        refinedproblem = NaghdiNoBounds(N)
        print("Solving refined problem (no bounds) ...")
        (success, iters, z) = dcsolve(refinedproblem, params, comm=comm)
        print("Refined problem solved.")
        assert success

        refined_u = z.split(deepcopy=True)[0]
        refined_u.rename("Displacement", "Displacement")
        _refined_nb_u << refined_u

        File(comm, "refined-nobounds/state-%d.xml.gz" % branchid) << z

    def bounds(self, Z, params, initial_guess):
        # FEniCS makes this extremely hard.
        l = Function(Z)
        l.vector()[:] = -1e20

        V = FunctionSpace(Z.mesh(), "CG", 1)
        lb = interpolate(Constant(-0.06), V)
        File(Z.mesh().mpi_comm(), "output/obstacle.pvd") << lb
        assign(l.sub(0).sub(2), lb)

        u = Function(Z)
        u.vector()[:] = +1e20
        return (l, u)

c0max = 3
cImax = 3
Nc0   = 151
NcI   = 151
c0loadings = linspace(0, c0max, Nc0)
cIloadings = linspace(0, cImax, NcI)

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=Naghdi(), teamsize=1, verbose=True, clear_output=True, logfiles=False)
    #dc.run(values={"c_0": c0loadings[:2], "c_I": cIloadings[-1]}, freeparam="c_0")
    dc.run(values={"c_0": c0loadings[:1], "c_I": cIloadings[-1]}, freeparam="c_0")
