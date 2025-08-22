# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import defcon.backend as backend
from defcon import iomodule
from defcon import branchio
from defcon import nonlinearproblem
from defcon import nonlinearsolver

firedrake_solver_args = ['nullspace', 'transpose_nullspace', 'appctx', 'pre_jacobian_callback', 'post_jacobian_callback', 'pre_function_callback', 'post_function_callback']

class BifurcationProblem(object):
    """
    A base class for bifurcation problems.

    This object is overridden by the user to implement his particular problem.
    """

    def mesh(self, comm):
        """
        This method loads/builds the mesh with a given communicator.

        *Arguments*
          comm (MPI communicator)
            The MPI communicator to use in building the mesh.

            Typically, the MPI communicator passed in here will be shared among
            a small number of the processors (called a team). Each team solves a
            PDE independently.
        *Returns*
          mesh (:py:class:`dolfin.Mesh`)
        """
        raise NotImplementedError

    def coarse_meshes(self, comm):
        """
        This method supplies additional coarsenings of the mesh for use
        in multigrid methods.

        Return

        [coarsest_mesh, next_finer, next_finer, ...]

        where the last mesh returned is the coarsening of the mesh returned by
        the .mesh() method.

        Of course, you may wish to make the mesh returned by .mesh() by refining
        the meshes returned here; defcon guarantees that this method is called
        after .mesh(), so construct the hierarchy of meshes in .mesh(), return the
        finest, and return the remainder here.

        The reason why this is split into two routines is because many users
        will not want to use multigrid, and this API design is simpler for them.
        """
        return []

    def function_space(self, mesh):
        """
        This method creates the function space for the prognostic variables of
        the problem.

        *Arguments*
          mesh (:py:class:`dolfin.Mesh`)
            The mesh to use.
        *Returns*
          functionspace (:py:class:`dolfin.FunctionSpace`)
        """
        raise NotImplementedError

    def parameters(self):
        r"""
        This method returns a list of tuples. Each tuple contains (Constant,
        asciiname, tex). For example, if there is one parameter

        lmbda = Constant(...)

        to be varied, this routine should return

        [(lmbda, "lambda", r"$\lambda$")]

        You may need to add

        # -*- coding: utf-8 -*-

        to the top of the Python script to use UTF characters.

        The values in the Constants are irrelevant; they are initialised in the
        continuation.

        *Returns*
          params
            A list of [(Constant, asciiname, tex), ...]
        """
        raise NotImplementedError

    def residual(self, state, params, test):
        """
        This method defines the PDE to be solved: if you would solve the PDE via

        solve(F == 0, state, ...)

        then this method should return F.

        The parameters will be varied internally by the continuation algorithm,
        i,e.  this will not be called multiple times for multiple parameters.

        *Arguments*
          state (:py:class:`dolfin.Function`)
            a Function in the FunctionSpace
          params (tuple of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
          test  (:py:class:`dolfin.TestFunction`)
            the test function to use in defining the residual
        *Returns*
          residual (:py:class:`ufl.form.Form`)
            the form for the residual
        """
        raise NotImplementedError

    def jacobian(self, F, state, params, test, trial):
        """
        This method defines the Jacobian to use in Newton's method.

        If you would solve the PDE via

        solve(F == 0, state, ...)

        then this method should return derivative(F, state, trial).

        The parameters will be varied internally by the continuation algorithm,
        i,e.  this will not be called multiple times for multiple parameters.

        *Arguments*
          F (:py:class:`ufl.form.Form`)
            the output of BifurcationProblem.residual
          state (:py:class:`dolfin.Function`)
            a Function in the FunctionSpace
          params (tuple of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
          test (:py:class:`dolfin.TestFunction`)
            the test function to use in defining the Jacobian
          trial (:py:class:`dolfin.TrialFunction`)
            the trial function to use in defining the Jacobian
        *Returns*
          jacobian (:py:class:`ufl.form.Form`)
            the form for the Jacobian
        """
        return backend.derivative(F, state, trial)

    def assembler(self, J, F, state, bcs):
        """
        This method defines the assembler to use to assemble
        the linear systems.

        Users rarely need to change this. It might be useful
        to override this in cases where one does not want to impose the Dirichlet
        BCs symmetrically, or where one wants to use a custom
        assembler (e.g. with fenics-shells).

        *Arguments*
          J (:py:class:`ufl.form.Form`)
            the output of BifurcationProblem.jacobian
          F (:py:class:`ufl.form.Form`)
            the output of BifurcationProblem.residual
          state (:py:class:`dolfin.Function`)
            the function we're solving for
          bcs (:py:class:`dolfin.DirichletBC`)
            the output of BifurcationProblem.boundary_conditions
        *Returns*
          an assembler for the problem
        """

        return backend.SystemAssembler(J, F, bcs)

    def objective(self, F, state, params):
        """
        WARNING: this method is currently unused, pending the implementation
        of necessary features in PETSc.

        This method computes the norm of the residual.

        The default is to compute the l_2 norm of the assembled
        residual vector. This has the advantage of speed, but is
        mesh-dependent. For mesh-independent termination criteria
        the appropriate dual norm should be coded here.

        Most users will not override this method; if you're not sure
        you want it, you don't need it.

        *Arguments*
          F (:py:class:`ufl.form.Form`)
            the output of BifurcationProblem.residual
          state (:py:class:`dolfin.Function`)
            a Function in the FunctionSpace
          params (tuple of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
        *Returns*
          objective (:py:class:`float`)
            the dual norm of the residual
        """
        pass

    def boundary_conditions(self, function_space, params):
        """
        This method returns a list of DirichletBC objects to impose on the
        problem.

        *Arguments*
          function_space (:py:class:`dolfin.FunctionSpace`)
            the function space returned by self.function_space()
          params (list of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
        *Returns*
          bcs (list of :py:class:`dolfin.DirichletBC`)
        """
        raise NotImplementedError

    def functionals(self):
        r"""
        This method returns a list of functionals. Each functional is a tuple
        consisting of a callable, an ascii name, a tex label, and (optionally)
        a function that returns UFL for the functional. The callable J is called via

          j = J(state, params)

         and should return a float.

        For example, this routine might consist of

        def functionals(self):
            def squared_L2norm(state, param):
                return assemble(inner(state, state)*dx)
            return [(L2norm, "L2norm", r"\|y\|", lambda state: inner(state, state)*dx)]

        *Returns*
          functionals (list of tuples)
        """
        raise NotImplementedError

    def number_initial_guesses(self, params):
        """
        Return the number of initial guesses we wish to search from at the
        very first deflation step, i.e. when initialising the continuation.
        Defaults to 1.
        """
        return 1

    def initial_guess(self, function_space, params, n):
        """
        Return the n^{th} initial guess.
        """
        raise NotImplementedError

    def number_solutions(self, params):
        """
        If the number of solutions for a given set of parameters is analytically
        known, then this function can return it. In this case the continuation
        algorithm will stop looking when it has found that many solutions.
        Otherwise the routine should return float("inf").

        *Arguments*
          params (tuple of :py:class:`float`)
            parameters to use, in the same order returned by parameters()

        *Returns*
          nsolutions (int)
            number of solutions, known from analysis, or float("inf")
        """
        return float("inf")

    def squared_norm(self, state1, state2, params):
        """
        This method computes the squared-norm between two vectors in the Hilbert
        space defined in the function_space method.

        This is used to define the norm used in deflation, and to test two
        functions for equality, among other things.

        The default is
            def squared_norm(self, state1, state2, params)
                return inner(state1 - state2, state1 - state2)*dx

        *Arguments*
        """
        return backend.inner(state1 - state2, state1 - state2)*backend.dx

    def trivial_solutions(self, function_space, params, freeindex):
        r"""
        This method returns any trivial solutions of the problem,
        i.e. solutions u such that f(u, \lambda) = 0 for all \lambda.
        These will be deflated at every computation.

        freeindex is the index into params that is being modified by this
        current continuation run. This is useful if the trivial solutions
        depend on the values of parameters in params other than freeindex.
        """
        return []

    def monitor(self, params, branchid, solution, functionals):
        """
        This method is called whenever a solution is computed.
        The user can specify custom processing tasks here.
        """
        pass

    def monitor_ac(self, branchid, sign, params, freeindex, solution, functionals, index, s):
        """
        This method is called whenever a solution is found with arclenth.
        The user can specify custom processing tasks here.
        """
        pass


    def io(self, prefix="output", comm=None):
        """
        Return an IO object that defcon will use to save solutions and functionals.

        The default is usually a good choice.
        """

        return iomodule.SolutionIO(prefix, comm=comm)

    def save_pvd(self, state, pvd, params, time=None):
        """
        Save the function state to a PVD file.

        The default is

        pvd.write(state)
        """
        if 'f_' in state.name():
            state.rename("Solution", "Solution")

        if backend.__name__ == "dolfin":
            if time is None:
                pvd << state
            else:
                pvd << (state, time)
        else:
            state.rename("Solution")
            if time is None:
                pvd.write(state)
            else:
                pvd.write(state, time=time)

    def save_xmf(self, state, xmf, time=None):
        """
        Save the function state to a XDMF file.

        The default is

        xmf.write(
        pvd << state
        """
        if 'f_' in state.name():
            state.rename("Solution", "Solution")

        if time is None:
            xmf.write(state)
        else:
            xmf.write(state, time)

    def nonlinear_problem(self, F, J, state, bcs):
        """
        The class used to assemble the nonlinear problem.

        Most users will never need to override this: it's only useful if you
        want to do something unusual in the assembly process.
        """
        if backend.__name__ == "dolfin":
            return nonlinearproblem.GeneralProblem(F, state, bcs, J=J, problem=self)
        else:
            return backend.NonlinearVariationalProblem(F, state, bcs, J=J)

    def solver(self, problem, params, solver_params, prefix="", **kwargs):
        """
        The class used to solve the nonlinear problem.

        Users might want to override this if they want to customize
        how the nonlinear solver is set up. For example, look at the
        hyperelasticity demo to see how this is used to pass a near-nullspace
        into an algebraic multigrid preconditioner.
        """

        if backend.__name__ == "dolfin":
            return nonlinearsolver.SNUFLSolver(
                problem, prefix=prefix,
                solver_parameters=solver_params,
                **kwargs
            )
        else:
            valid_kwargs = {key: value for (key, value) in kwargs.items() if key in firedrake_solver_args}
            return backend.NonlinearVariationalSolver(
                problem, options_prefix=prefix,
                solver_parameters=solver_params,
                **valid_kwargs
            )

    def compute_stability(self, params, branchid, solution, hint=None):
        """
        This method allows the user to compute whether a solution on a given branch is
        stable.

        Stability means different things to different problems. For example, in an unconstrained
        energy minimisation problem, investigating stability involves checking the positive
        definiteness of the Hessian at a critical point; in constrained minimisation, one needs
        to check the inertia of the Hessian and verify that the number of negative eigenvalues
        is as expected, given the constraints.

        With this in mind, defcon does not attempt to compute the stability of solutions;
        it just provides a hook for you to calculate whatever test you wish.

        It is often the case that some results from one stability analysis can be used
        to accelerate the calculation of the next. For example, in SLEPc one can
        set an initial guess for the eigenspace, and if it is a good guess the algorithms
        converge much faster. This is the purpose of the hint input. For the first
        calculation on a given branch, it is None, but on subsequent calculations it
        is passed along from the outputs of one calculation to the inputs of the next.

        This routine should return the following: a dictionary like

        d = {"stable": is_stable,
             "eigenvalues":    [list of eigenvalues],
             "eigenfunctions": [list of eigenfunctions],
             "hint": hint}

        is_stable can be anything that can be evaluated with literal_eval in Python.
        Its value has meaning only to you. The reason why this is not True/False
        is because stability is more subtle than that -- for example, one could
        pass None to indicate that the computation failed. In other problems
        (such as the Gross-Pitaevskii equation in quantum mechanics) there are
        different ways a problem can be unstable; these could be distinguished
        by returning "exponential" or "oscillatory".

        eigenvalues is a list of real or complex numbers. These will be saved
        by the I/O object. The list can be empty.

        eigenfunctions is a list of Functions. These will also be saved by
        the I/O object. Again, the list can be empty.

        hint is any data that you wish to pass on to the next calculation.

        For an example of this in action, see the elastica demo.
        """
        pass

    def solver_parameters(self, params, task, **kwargs):
        """
        Returns a dictionary with the PETSc options to configure
        the backend nonlinear solver.  Users should
        override this method in their own subclasses to set
        solver/preconditioner preferences.

        params is the set of continuation parameters, which is present
        so that users could adapt the solver strategy depending on the
        parameter regime if needed.

        task is an instance of {DeflationTask, ContinuationTask, StabilityTask, ArclengthTask},
        and allows for the user to tune the solver parameters as required."""
        return {}

    def transform_guess(self, state, task, io):
        """
        When performing deflation, it's sometimes useful to modify the
        initial guess in some way (e.g. Hermite promotion in the case
        of a nonlinear Schroedinger equation, or perturbation in the case
        of perfect Z_2 symmetry). This method provides exactly such a hook.
        It should modify state in place, not return anything.
        """
        pass

    def branch_found(self, task):
        """
        This hook is experimental. Its interface will probably change.

        When a new branch is discovered, this method is called. It is called
        with the ContinuationTask representing the new branch.

        This method should return a list of additional tasks to perform,
        such as continuation tasks in another parameter dimension. The
        taskids of these additional tasks should be task.taskid + 1,
        task.taskid + 2, ...
        """

        return []

    def postprocess(self, solution, params, branchid, window):
        """
        This hook is experimental. Its interface might change.

        When the 'Postprocess' button in the GUI is called, it
        executes this code.

        window is the QtGui.QMainWindow object representing the GUI.
        """

        print("To customise postprocessing, override the BifurcationProblem.postprocess method.")

    def continuation_filter(self, params, branchid, functionals, io):
        """
        In multiparameter continuation, we may not want to use all the branches
        we found in run #1 (along an initial parameter) for run #2 (along a different
        parameter).

        In this case, override this method and return False if you *don't* want to
        use a particular solution.

        For efficiency, it would be better to only use the functionals already computed;
        but if this isn't possible, you can use the io object to read the solution from
        disk. This is a bad idea, though, as this only gets executed on the master
        process (on one core).
        """

        return True

    def bounds(self, function_space, params, initial_guess):
        """
        This method supports the solution of variational inequalities with box constraints.

        Interpolate the (lower, upper) bounds to functions in function_space,
        and return the tuple (lower, upper).
        """
        pass

    def predict(self, problem, solution, oldparams, newparams, hint):
        """
        This method asks the problem to make a (cheap) prediction for how the
        branch will change when we move in parameter space from oldparams to
        newparams. For example, the user might solve a tangent linearisation to
        construct a guess, or use past information to construct a secant
        linearisation.

        On entry, solution contains the solution for oldparams. On exit, the
        guess for the solution at newparams should be written into solution.

        The problem variable is _usually_ the same as self, but might not be
        when additional problems are derived from base ones (e.g. if you are
        solving a VI, a VIBifurcationProblem is constructed from the
        BifurcationProblem you supply, so self will be a BifurcationProblem and
        problem will be the VIBifurcationProblem you actually want to use).

        The hint variable is a device for this method to communicate with itself
        across calls (e.g. for secant or arclength continuation). Whatever this
        routine returns on the first call will be passed as the hint on the
        second, and so on. The hint is None for the first call.

        By default, this routine does nothing, and hence implements zero-order
        continuation.

        Standard implementations of various prediction algorithms are available
        in defcon/prediction.py. Use like

        def predict(self, *args, **kwargs):
            return tangent(*args, **kwargs)
        """
        pass

    def ac_residual(self, ac_state, params, ac_test):
        """
        This method is an ugly workaround for a serious design flaw in UFL.

        In UFL, one cannot split or differentiate a form with respect to the
        output of split. This is a problem for us when we do arclength continuation,
        as I need to re-use the user's residual in a bigger system for state x parameter.
        However, if the user's code uses split or derivative (quite reasonable and
        common things to do), this breaks.

        When the arclength code is calculating the state residual, it first tries to
        call BifurcationProblem.residual on the state component of the bigger mixed
        function space; if that fails, it calls this, on the _entire_ arclength
        state and test functions. This means that one can do something like

        def residual(self, u, params, v):
            # Normal state residual
            Energy = self.energy(u, params)
            L = derivative(Energy, u, v)

            return L

        def ac_residual(self, ac, params, w):
            # State residual for use in arclength calculation
            (u, _) = split(ac)
            Energy = self.energy(u, params)
            L = derivative(Energy, ac, w)

            return L

        to take the derivative with respect to the whole function instead of the original
        split state.

        *Arguments*
          state (:py:class:`dolfin.Function`)
            a Function in the FunctionSpace
          params (tuple of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
          test  (:py:class:`dolfin.TestFunction`)
            the test function to use in defining the residual
        *Returns*
          residual (:py:class:`ufl.form.Form`)
            the form for the residual
        """
        raise NotImplementedError

    def launch_paraview(self, filename):
        """
        This can be used to set a default visualisation for paraview.
        For example, if you override this and set it to

        subprocess.Popen(["paraview", "--script=viz.py", filename])

        then paraview will execute the Python script viz.py on launch.
        (You can make such scripts with 'Start trace' in paraview.)
        """

        from subprocess import Popen
        Popen(["paraview", filename])

    def estimate_error(self, F, J, state, bcs, params):
        """
        Compute an estimate for the error in the evaluation of the functional
        J due to the finite element approximation.

        Defcon includes a simple implementation of a standard dual-weighted
        residual error estimator. To use this, do

        def estimate_error(self, *args, **kwargs):
            return estimate_error_dwr(self, *args, **kwargs)
        """
        pass

    def enrich_function_space(self, V):
        """
        Make a richer function space. Used for
        defining the function space in which to solve
        the adjoint problem in the default implementation
        of the dual-weighted residual error estimator.
        """

        from defcon.backend import MixedElement, VectorElement, TensorElement, TensorProductElement, FunctionSpace

        ele = V.ufl_element()
        if isinstance(ele, MixedElement) and not isinstance(ele, (VectorElement, TensorElement)):
            raise NotImplementedError("Implement this method yourself")

        if backend.__name__ != "firedrake":
            raise NotImplementedError("Only implemented for Firedrake, sorry. Implement this method yourself")

        N = ele.degree()
        try:
            N, = set(N)
        except TypeError:
            pass
        except ValueError:
            raise NotImplementedError("Different degrees on TensorProductElement")

        from firedrake.preconditioners.pmg import PMGBase

        if isinstance(ele, TensorElement):
            sub = ele.sub_elements
            new_ele = TensorElement(PMGBase.reconstruct_degree(sub[0], N+1), shape=V.value_shape, symmetry=ele.symmetry())
        elif isinstance(ele, VectorElement):
            sub = ele.sub_elements
            new_ele = VectorElement(PMGBase.reconstruct_degree(sub[0], N+1), dim=len(sub))
        elif isinstance(ele, TensorProductElement):
            new_ele = TensorProductElement(*(PMGBase.reconstruct_degree(sub, N) for sub in ele.sub_elements), cell=ele.cell)
        else:
            new_ele = ele.reconstruct(degree=N+1)

        return FunctionSpace(V.mesh(), new_ele)
