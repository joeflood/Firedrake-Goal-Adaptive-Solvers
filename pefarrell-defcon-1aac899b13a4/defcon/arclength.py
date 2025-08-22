# -*- coding: utf-8 -*-
from __future__ import absolute_import

import defcon

from mpi4py import MPI
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms import expand_derivatives
import ufl

from math import copysign, sqrt
from heapq import heappush, heappop
import gc
import json
import six
import os.path
import sys

import defcon.backend as backend
from defcon.master import DefconMaster
from defcon.worker import DefconWorker
from defcon.newton import newton
from defcon.tasks import QuitTask, ArclengthTask, Response
from defcon.parametertools import parameters_to_string
from defcon.compatibility import make_comm


class ArclengthContinuation(object):
    """
    This class is the main driver for arclength continuation.
    """

    def __init__(self, problem, **kwargs):
        """
        Constructor.

        *Arguments*
          problem (:py:class:`defcon.BifurcationProblem`)
            A class representing the bifurcation problem to be solved.
          teamsize (:py:class:`int`)
            How many processors should coordinate to solve any individual PDE.
          verbose (:py:class:`bool`)
            Activate verbose output.
          debug (:py:class:`bool`)
            Activate debugging output.
          logfiles (:py:class:`bool`)
            Whether defcon should remap stdout/stderr to logfiles (useful for many processes).
          sleep_time (:py:class:`double`)
            How long in seconds master sleeps between repeated attempts when
            probing workers for response. Negative value means busy waiting,
            positive value saves up to one core of CPU time. Unspecified or
            None results in adaptive value given as 5 percent of last response
            time but at most 1.0 second.
          comm (MPI.Comm)
            The communicator that gathers all processes involved in this computation
        """

        worldcomm = kwargs.get("comm", MPI.COMM_WORLD).Dup()
        kwargs["comm"] = worldcomm

        if not isinstance(problem, ArclengthProblem):
            problem = ArclengthProblem(problem)
        self.problem = problem

        if worldcomm.rank == 0:
            self.thread = ArclengthMaster(problem, **kwargs)
        else:
            self.thread = ArclengthWorker(problem, **kwargs)

    def run(self, params, free, ds, sign, bounds, funcbounds=None, branchids=None):
        """
        The main execution routine.

        *Arguments*
          params (:py:class:`tuple`)
            A tuple of parameter values to start from. All known solutions for these
            values will be used for the arclength continuation, unless branchids is
            specified.
          free (:py:class:`str`)
            The name of the parameter that will be varied in the arclength continuation.
          ds (:py:class:`float`)
            The value of the step to take in arclength.
          sign (:py:class:`int`)
            The initial direction of travel for the parameter (must be +1 or -1)
          bounds (:py:class:`tuple`)
            The bounds of interest for the parameter (param_min, param_max)
          funcbounds (:py:class:`tuple`)
            The bounds of interest for the functional (func_name, func_min, func_max)
          branchids (:py:class:`list`)
            The list of branchids to continue (or None for all of them)
        """

        # First, check we're parallel enough.
        if self.thread.worldcomm.size < 2:
            msg = """
Defcon started with only 1 process.
At least 2 processes are required (one master, one worker).

Launch with mpiexec: mpiexec -n <number of processes> python %s
""" % sys.argv[0]
            self.thread.log(msg, warning=True)
            sys.exit(1)

        # Next, check arguments

        problem_parameters = self.problem.parameters()
        assert len(problem_parameters) == len(params)
        assert sign in [+1, -1]
        assert ds > 0
        assert len(bounds) == 2
        assert bounds[0] < bounds[1]
        if funcbounds is not None:
            assert len(funcbounds) == 3
            assert funcbounds[1] < funcbounds[2]

        # Fix the fixed parameters and identify the free parameter.
        freeindex = None
        for (index, param) in enumerate(problem_parameters):
            if param[1] == free:
                freeindex = index
                break

        if freeindex is None:
            backend.info_red("Cannot find %s in parameters %s." % (free, [param[1] for param in problem_parameters]))
            assert freeindex is not None

        assert bounds[0] <= params[freeindex] <= bounds[1]

        # If we have bounds on the functional, figure out the functional index
        if funcbounds is not None:
            funcname = funcbounds[0]

            funcindex = None
            for (i, functionaldata) in enumerate(self.problem.functionals()):
                if functionaldata[1] == funcname:
                    funcindex = i
                    break
            assert funcindex is not None
            funcbounds = (funcindex, funcbounds[1], funcbounds[2])

        # Aaaand .. run.

        self.thread.run(problem_parameters, freeindex, params, ds, sign, bounds, funcbounds, branchids)

    def bifurcation_diagram(self, functional, parameter, branchids=None, style="o-k", **kwargs):
        if self.thread.rank != 0:
            return

        if branchids is None:
            branchids = ["*"] # find all

        import matplotlib.pyplot as plt
        import glob
        if "linewidth" not in kwargs: kwargs["linewidth"] = 2
        if "markersize" not in kwargs: kwargs["markersize"] = 1

        functionals = self.problem.functionals()
        parameters  = self.problem.parameters()
        io = self.problem.io()
        io.setup(parameters, functionals, None)

        # Find the functional index.
        funcindex = None
        for (i, functionaldata) in enumerate(functionals):
            if functionaldata[1] == functional:
                funcindex = i
                break
        assert funcindex is not None

        # And find the variable index.
        paramindex = None
        for (i, param) in enumerate(parameters):
            if param[1] == parameter:
                paramindex = i
                break

        for branchid in branchids:
            for jsonfile in glob.glob(io.directory + "/arclength/*freeindex-%s-branchid-%s-*/*freeindex-%s-branchid-%s-*.json" % (paramindex, branchid, paramindex, branchid)):
                self.thread.log("Reading JSON file %s" % jsonfile)
                try:
                    data = json.load(open(jsonfile, "r"))
                    x = [entry[0] for entry in data]
                    y = [entry[1][funcindex] for entry in data]

                    plt.plot(x, y, style, **kwargs)
                except ValueError:
                    self.thread.log("Error: could not load %s" % jsonfile, warning=True)
                    import traceback; traceback.print_exc()

        plt.grid()
        plt.xlabel(parameters[paramindex][2])
        plt.ylabel(functionals[funcindex][2])

class ArclengthWorker(DefconWorker):
    """
    This class handles the actual execution of the tasks necessary
    to do arclength continuation.
    """
    def __init__(self, problem, **kwargs):
        DefconWorker.__init__(self, problem, **kwargs)

        # A map from the type of task we've received to the code that handles it.
        self.callbacks = {ArclengthTask: self.arclength_task}

    def fetch_data(self):
        problem = self.problem
        (self.function_space, self.R, self.ac_space) = problem.setup_spaces(make_comm(self.teamcomm))

        # Configure garbage collection frequency:
        self.determine_gc_frequency(self.function_space)

        self.consts  = [param[0] for param in self.parameters]

        self.state    = backend.Function(self.ac_space)
        self.prev     = backend.Function(self.ac_space)
        # Keep one previous history to deflate, to make sure we don't get stuck in a loop
        self.prevprev = backend.Function(self.ac_space)
        # And a Function for the tangent
        self.tangent = backend.Function(self.ac_space)

        self.test     = backend.TestFunction(self.ac_space)
        self.ds       = backend.Constant(0)

        # Override the constant with the value of the parameter we're solving for
        self.consts[self.freeindex] = problem.ac_to_parameter(self.state, deep=False)
        self.ac_residual = problem.ac_residual(self.state, self.prev, self.ds, self.consts, self.test)
        self.ac_jacobian = problem.ac_jacobian(self.ac_residual, self.state, backend.TrialFunction(self.ac_space))

        (self.bcs, self.hbcs) = problem.boundary_conditions()

    def run(self, problem_parameters, freeindex, *args):

        self.parameters = problem_parameters
        self.functionals = self.problem.functionals()
        self.freeindex = freeindex
        self.fetch_data()

        dummy = Dummy(self.parameters, self.consts)
        self.configure_io(dummy)
        self.construct_deflation(dummy)

        task = None
        while True:
            self.collect()

            if task is None:
                task = self.fetch_task()

            if isinstance(task, QuitTask):
                self.log("Quitting gracefully")
                return
            else:
                self.log("Executing task %s" % task)
                task = self.callbacks[task.__class__](task)
        return

    def compute_functionals(self, solution):
        funcs = []
        for functional in self.functionals:
            func = functional[0]
            j = func(solution, self.consts)
            assert isinstance(j, float)
            funcs.append(j)
        return funcs

    def arclength_task(self, task):
        params    = task.params
        branchid  = task.branchid
        bounds    = task.bounds
        funcbounds  = task.funcbounds
        ds_       = task.ds
        sign      = task.sign
        problem   = self.problem

        param = params[self.freeindex]
        paramname = self.parameters[self.freeindex][1]
        self.ds.assign(ds_)

        self.firsttime = True
        self.tangent_prev = None

        # Configure the parameters
        for (const, value) in zip(self.consts, params):
            if isinstance(const, backend.Constant):
                const.assign(value)

        # Load the solution into the previous value
        solution = self.io.fetch_solutions(params, [branchid])[0]
        problem.load_solution(solution, backend.Constant(param), self.state)

        # Data about functionals
        functionals = self.compute_functionals(solution)
        data = [(param, functionals)]
        self.log("Initialising arclength at %s = %.15e with functionals %s" % (self.parameters[self.freeindex][1], param, functionals))

        # Data for step halving for robustness
        num_halvings = 0

        if backend.__name__ == "dolfin":
            arcpath = os.path.join(self.io.directory, "arclength", "params-%s-freeindex-%s-branchid-%s-ds-%.14e.xdmf" % (parameters_to_string(self.io.parameters, params), self.freeindex, branchid, self.ds))
            arcxmf = backend.XDMFFile(make_comm(self.teamcomm), arcpath)
            arcxmf.parameters["flush_output"] = True
            arcxmf.parameters["functions_share_mesh"] = True
            arcxmf.parameters["rewrite_function_mesh"] = False
        elif backend.__name__ == "firedrake":
            paramspath = "params-%s-freeindex-%s-branchid-%s-ds-%.14e-sign-%d" % (parameters_to_string(self.io.parameters, params), self.freeindex, branchid, ds_, sign)
            arcpath = os.path.join(self.io.directory, "arclength", paramspath, "solutions", paramspath+".pvd")
            arcpvd = backend.File(arcpath, comm=make_comm(self.teamcomm))

        index = -1.0 # needs to be a float, otherwise dolfin does the Wrong Thing. Argh!
        s = 0.0

        # And begin the main loop
        while bounds[0] <= param <= bounds[1]:
            self.collect()
            index += 1

            current_params = list(params)
            current_params[self.freeindex] = param

            # Step 1. Compute the tangent linearisation at self.state
            F = problem.tangent_residual(self.state, self.tangent, self.tangent_prev, sign, self.test)
            J = problem.tangent_jacobian(F, self.tangent, backend.TrialFunction(self.ac_space))

            self.log("Computing tangent")
            solverparams = self.problem.solver_parameters(current_params, task)
            solverparams["snes_linesearch_type"] = "basic"
            solverparams["snes_max_it"] = 3 # maybe need some iterative refinement
            (success, iters) = newton(F, J, self.tangent, self.hbcs,
                                      current_params,
                                      self.problem,
                                      solverparams,
                                      self.teamno)
            if not success:
                self.log("Warning: failed to compute tangent", warning=True)
                break
            delta_param = self.fetch_R(problem.ac_to_parameter(self.tangent, deep=True))
            self.log("Tangent δ%s = %.15e" % (paramname, delta_param))

            # Step 2. Update the state guess with the tangent
            self.prevprev.assign(self.prev)
            self.prev.assign(self.state)
            z_tlm = problem.ac_to_state(self.tangent, deep=False)
            lmbda_tlm = problem.ac_to_parameter(self.tangent, deep=False)
            nrm = sqrt(backend.assemble(self.problem.squared_norm(z_tlm, ufl.zero(*z_tlm.ufl_shape), self.consts) + backend.inner(lmbda_tlm, lmbda_tlm)*backend.dx))

            # Step 3. Solve the arclength system
            # I will employ an adaptive loop: if the continuation doesn't
            # converge, try halving ds, until we give up after 10 halvings
            for adaptive_loop in range(10):
                self.state.assign(self.prev + (float(self.ds)/nrm) * self.tangent)

                # Deflate the past solution, to make sure we don't converge to that again
                if self.firsttime:
                    self.deflation.deflate([])
                else:
                    self.deflation.deflate([self.prevprev])

                self.log("Computing arclength step")
                (success, iters) = newton(self.ac_residual, self.ac_jacobian, self.state, self.bcs,
                                          current_params,
                                          self.problem,
                                          self.problem.solver_parameters(current_params, task),
                                          self.teamno, self.deflation)

                if success: # exit adaptive loop
                    break
                else:
                    self.log("Warning: failed to solve arclength system with step %s. Halving step" % float(self.ds), warning=True)
                    self.ds.assign(0.5*float(self.ds))
                    num_halvings += 1

            self.firsttime = False # start deflating prevprev

            if success:
                s += float(self.ds)

                if num_halvings > 0 and adaptive_loop == 0 and iters <= 4: # we have halved the step before, and this worked
                    self.ds.assign(2.0*float(self.ds))
                    self.log("Doubling step to %s" % float(self.ds))
                    num_halvings -= 1
            elif not success: # exit arclength loop
                break

            # Step 4. Compute functionals and save information
            z_ = problem.ac_to_state(self.state, deep=True)
            lmbda_ = problem.ac_to_parameter(self.state, deep=True)
            lmbda_.rename(paramname, paramname)

            self.log("Saving with index = %s" % index)

            if backend.__name__ == "dolfin":
                problem.save_xmf(z_, arcxmf, index)
                arcxmf.write(lmbda_, index)
            elif backend.__name__ == "firedrake":
                problem.save_pvd(z_, arcpvd, current_params, time=index)

            functionals = self.compute_functionals(z_)

            param = self.fetch_R(lmbda_)

            arcpath = os.path.join(self.io.directory, "arclength", "params-%s-freeindex-%s-branchid-%s-ds-%.14e-sign-%d" % (parameters_to_string(self.io.parameters, params), self.freeindex, branchid, ds_, sign), \
                                   "index=%d-"%(index) + parameters_to_string(self.parameters, [param]))
            self.log("Saving solution with path = %s" % arcpath)
            self.io.save_solution(z_, functionals, [param], branchid, save_dir=arcpath)

            problem.monitor_ac(branchid, task.sign, current_params, self.freeindex, z_, functionals, index, s)

            del z_
            del lmbda_

            data.append((param, functionals))
            self.log("Continued arclength to %s = %.15e with functionals %s" % (self.parameters[self.freeindex][1], param, functionals))

            # Step 5. Cycle the tangent linear variables
            if self.tangent_prev is None:
                self.tangent_prev = backend.Function(self.ac_space)
            self.tangent_prev.assign(self.tangent)

            # FIXME: this is quadratic in ds^-1; it's doing work of O(num_steps), O(num_steps) times
            problem.monitor(params, branchid, problem.ac_to_state(self.state, deep=True), functionals)
            self.io.save_arclength(params, self.freeindex, branchid, task.ds, task.sign, data)

            # Finally check if we should carry on wrt the functional
            if funcbounds is not None:
                (funcindex, funclo, funchi) = funcbounds
                if not funclo <= functionals[funcindex] <= funchi:
                    self.log("Breaking arclength continuation due to functional bounds")
                    break

        response = Response(task.taskid, success=success)
        if self.teamrank == 0:
            self.log("Sending response %s to master" % response)
            self.worldcomm.send(response, dest=0, tag=self.responsetag)

    def fetch_R(self, r):
        """
        Given a Function in FunctionSpace(mesh, "R", 0), return its value as a float.
        """
        if backend.__name__ == "dolfin":
            rval = r.vector().get_local()
            if len(rval) == 0:
                rval = 0.0
            else:
                rval = rval[0]
            rval = backend.MPI.sum(r.function_space().mesh().mpi_comm(), rval)
            return rval
        else:
            with r.dat.vec_ro as v:
                out = v.sum()
            return out

class ArclengthMaster(DefconMaster):
    """
    This class implements the core logic of running arclength continuation
    in parallel.
    """
    def __init__(self, *args, **kwargs):
        DefconMaster.__init__(self, *args, **kwargs)

        # Don't need DefconMaster's callbacks
        del self.callbacks

    def seed_initial_tasks(self, params, ds, sign, bounds, funcbounds, branchids):
        if branchids is None:
            branchids = self.io.known_branches(params)

        for branchid in branchids:
            task = ArclengthTask(taskid=self.taskid_counter,
                                 params=params,
                                 branchid=branchid,
                                 bounds=bounds,
                                 funcbounds=funcbounds,
                                 sign=sign,
                                 ds=ds)
            heappush(self.new_tasks, (branchid, task))
            self.taskid_counter += 1

    def finished(self):
        return len(self.new_tasks) + len(self.wait_tasks) == 0

    def debug_print(self):
        if self.debug:
            self.log("DEBUG: new_tasks = %s" % [(priority, str(x)) for (priority, x) in self.new_tasks])
            self.log("DEBUG: wait_tasks = %s" % [(key, str(self.wait_tasks[key][0]), self.wait_tasks[key][1]) for key in self.wait_tasks])
            self.log("DEBUG: idle_teams = %s" % self.idle_teams)

        # Also, a sanity check: idle_teams and busy_teams should be a disjoint partitioning of range(self.nteams)
        busy_teams = set([self.wait_tasks[key][1] for key in self.wait_tasks])
        if len(set(self.idle_teams).intersection(busy_teams)) > 0:
            self.log("ALERT: intersection of idle_teams and wait_tasks: \n%s\n%s" % (self.idle_teams, [(key, str(self.wait_tasks[key][0])) for key in self.wait_tasks]), warning=True)
        if set(self.idle_teams).union(busy_teams) != set(range(self.nteams)):
            self.log("ALERT: team lost! idle_teams and wait_tasks: \n%s\n%s" % (self.idle_teams, [(key, str(self.wait_tasks[key][0])) for key in self.wait_tasks]), warning=True)


    def run(self, problem_parameters, freeindex, params, ds, sign, bounds, funcbounds, branchids):
        self.functionals = self.problem.functionals
        self.parameters  = problem_parameters
        self.freeindex   = freeindex

        dummy = Dummy(problem_parameters)
        self.configure_io(dummy)

        # List of idle teams
        self.idle_teams = list(six.moves.xrange(self.nteams))

        # Task id counter
        self.taskid_counter = 0

        # Data structures for lists of tasks in various states
        self.new_tasks  = [] # tasks yet to be dispatched
        self.wait_tasks = {} # tasks dispatched, waiting to hear back

        # Seed initial tasks
        self.seed_initial_tasks(params, ds, sign, bounds, funcbounds, branchids)

        # The main master loop.
        while not self.finished():
            self.debug_print()

            # Dispatch any tasks that can be dispatched
            while len(self.new_tasks) > 0 and len(self.idle_teams) > 0:
                self.dispatch_task()

            # We can't send out any more tasks, either because we have no
            # tasks to send out or we have no free processors.
            # If we aren't waiting for anything to finish, we'll exit the loop
            # here. otherwise, we wait for responses and deal with consequences.
            if len(self.wait_tasks) > 0:
                self.log("Cannot dispatch any tasks, waiting for response.")
                self.collect()

                response = self.fetch_response()
                self.handle_response(response)

        # Finished the main loop, tell everyone to quit
        quit = QuitTask()
        for teamno in range(self.nteams):
            self.send_task(quit, teamno)

    def dispatch_task(self):
        (priority, task) = heappop(self.new_tasks)
        idleteam = self.idle_teams.pop(0)
        self.send_task(task, idleteam)
        self.wait_tasks[task.taskid] = (task, idleteam)

    def handle_response(self, response):
        (task, team) = self.wait_tasks[response.taskid]
        self.log("Received response %s about task %s from team %s" % (response, task, team))
        del self.wait_tasks[response.taskid]
        self.idle_teams.append(team)

class Dummy(object):
    def __init__(self, parameters, constants=None):
        self.parameters = parameters
        self.constants  = constants

class ArclengthProblem(object):
    def __init__(self, problem):
        self.problem = problem

        # A list of functions to just call the underlying problem on
        self.passthrough = ["parameters", "functionals", "io", "solver_parameters",
                            "nonlinear_problem", "solver", "squared_norm", "save_xmf","save_pvd", "monitor", "monitor_ac"]

    def setup_spaces(self, comm):
        self.state_residual = None
        self.state_residual_derivative = None

        problem = self.problem
        mesh = problem.mesh(comm)

        self.state_space = problem.function_space(mesh)
        state_element = self.state_space.ufl_element()

        R_element = backend.FiniteElement("R", state_element.cell, 0)
        self.R = backend.FunctionSpace(mesh, R_element)

        ac_element = backend.MixedElement([state_element, R_element])
        self.ac_space = backend.FunctionSpace(mesh, ac_element)

        return (self.state_space, self.R, self.ac_space)

    def ac_to_state(self, ac, deep=True):
        # Given the mixed space representing what we're solving the augmented
        # arclength system for, return the part of it that denotes the state
        # we're solving for
        if deep:
            if backend.__name__ == "dolfin":
                return ac.split(deepcopy=True)[0]
            else:
                return ac.subfunctions[0]
        else:
            return backend.split(ac)[0]

    def ac_to_parameter(self, ac, deep=True):
        # Given the mixed space representing what we're solving the augmented
        # arclength system for, return the part of it that denotes the parameter
        # we're solving for
        if deep:
            if backend.__name__ == "dolfin":
                return ac.split(deepcopy=True)[1]
            else:
                return ac.subfunctions[1]
        else:
            return backend.split(ac)[1]

    def ac_residual(self, ac, ac_prev, ds, params, test):
        # At this point, params[freeindex] is *itself* a part of the state ac --
        # not a Constant. This is because we're solving for it.
        # See the call to ac_to_parameter() in ArclengthWorker.fetch_data.
        problem = self.problem

        (z,  lmbda)  = backend.split(ac)
        (z_, lmbda_) = backend.split(ac_prev)
        (w,  mu)     = backend.split(test)

        if self.state_residual is None:
            try:
                self.state_residual = problem.residual(z, params, w)
            except:
                self.state_residual = problem.ac_residual(ac, params, test)

        workaround = lambda form: mu*form
        ac_residual = (
                       self.state_residual
                       # Want to write
                       # + mu * problem.squared_norm(z, z_)
                       # but cannot. This is a workaround
                       + map_integrands(workaround, problem.squared_norm(z, z_, params))
                       + mu*backend.inner(lmbda - lmbda_, lmbda - lmbda_)*backend.dx
                       - mu*ds**2*backend.dx
                      )

        return ac_residual

    def ac_jacobian(self, ac_residual, ac, trial):
        return expand_derivatives(backend.derivative(ac_residual, ac, trial))

    def tangent_residual(self, ac, tangent, tangent_prev, sign, test):
        problem = self.problem

        if self.state_residual_derivative is None:
            self.state_residual_derivative = expand_derivatives(backend.derivative(self.state_residual, ac, tangent))

        lmbda_tlm = self.ac_to_parameter(tangent, deep=False)
        mu        = self.ac_to_parameter(test,    deep=False)

        if tangent_prev is not None:
            # point in the same direction as before
            normalisation_condition = backend.inner(tangent, tangent_prev) - backend.Constant(1)
        else:
            # start going in direction of sign
            normalisation_condition = lmbda_tlm - backend.Constant(copysign(1.0, sign))

        tangent_residual = self.state_residual_derivative + mu*normalisation_condition*backend.dx

        return tangent_residual

    def tangent_jacobian(self, tangent_residual, tangent, trial):
        return expand_derivatives(backend.derivative(tangent_residual, tangent, trial))

    def boundary_conditions(self):
        # We pass in None here for the parameters because for arclength we
        # can't have the boundary conditions depend on the parameter values
        # (well, one could, but it would be a lot of work)
        problem = self.problem

        bcs = problem.boundary_conditions(self.ac_space.sub(0), None)

        # Why do they break the interface at every opportunity?
        hbcs = problem.boundary_conditions(self.ac_space.sub(0), None)
        [bc.homogenize() for bc in hbcs]

        return (bcs, hbcs)

    def load_solution(self, z, param, ac_state):

        if backend.__name__  == "dolfin":
            backend.assign(ac_state.sub(0), z)
            r = backend.Function(self.R)
            r.assign(param)
            backend.assign(ac_state.sub(1), r)

        elif backend.__name__ == "firedrake":
            ac_state.subfunctions[0].assign(z)
            ac_state.subfunctions[1].assign(backend.Constant(param))

    def __getattr__(self, name):
        if name in self.passthrough:
            return getattr(self.problem, name)
        else:
            raise AttributeError
