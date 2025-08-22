from __future__ import absolute_import, print_function, division

from petsc4py import PETSc

import traceback
import resource

import defcon.backend as backend
from defcon.newton import newton
from defcon.tasks import QuitTask, ContinuationTask, DeflationTask, StabilityTask, Response
from defcon.mg import create_dm
from defcon.thread import DefconThread
from defcon.operatordeflation import ShiftedDeflation
from defcon.profiling import DummyEvent
from defcon.compatibility import function_space_dimension, make_comm
from defcon.bifurcationproblem import BifurcationProblem


class DefconWorker(DefconThread):
    """
    This class handles the actual execution of the tasks necessary
    to do deflated continuation.
    """
    def __init__(self, problem, **kwargs):
        DefconThread.__init__(self, problem, **kwargs)

        # Override parent's gc_frequency, None means auto determined later
        self.gc_frequency = kwargs.get("gc_frequency")

        # A map from the type of task we've received to the code that handles it.
        self.callbacks = {DeflationTask:    self.deflation_task,
                          StabilityTask:    self.stability_task,
                          ContinuationTask: self.continuation_task}

        self.profile = kwargs.get("profile", True)
        global Event
        if self.profile:
            Event = PETSc.Log.Event
            PETSc.Log.begin()
        else:
            Event = DummyEvent

    def determine_gc_frequency(self, function_space):
        """Set garbage collection frequency according to the size of
        the problem if not already set."""
        if self.gc_frequency is None:
            dofs_per_core = function_space_dimension(function_space) // self.teamcomm.size
            if   dofs_per_core > 100000: self.gc_frequency = 1
            elif dofs_per_core <  10000: self.gc_frequency = 10
            else:                        self.gc_frequency = 5

    def collect(self):
        with Event("garbage"):
            DefconThread.collect(self)

    def run(self, parameters, freeparam):

        runev = Event("run")
        runev.begin()

        initev = Event("initialisation")
        initev.begin()

        self.parameters = parameters

        # If we're going downwards in continuation parameter, we need to change
        # signs in a few places
        self.signs = []
        for label in self.parameters.labels:
            values = self.parameters.values[label]
            if values[0] < values[-1]:
                self.signs.append(+1)
            else:
                self.signs.append(-1)

        # Fetch data from the problem.

        self.mesh = self.problem.mesh(make_comm(self.teamcomm))
        self.function_space = self.problem.function_space(self.mesh)
        self.dm = create_dm(self.function_space, self.problem)
        self.problem._dm = self.dm

        # Configure garbage collection frequency:
        self.determine_gc_frequency(self.function_space)

        self.state = backend.Function(self.function_space)
        self.prevstate = backend.Function(self.function_space)
        self.trivial_solutions = None
        test_ = backend.TestFunction(self.function_space)
        self.residual = self.problem.residual(self.state, parameters.constants, test_)
        self.jacobian = self.problem.jacobian(self.residual, self.state, parameters.constants, test_, backend.TrialFunction(self.function_space))

        self.configure_io(parameters)
        self.construct_deflation(parameters)

        # Should we do error estimation? Let's see if the user
        # has overridden the estimate_error method or not
        estimate_error = False

        for klass in self.problem.__class__.__mro__:
            if klass is BifurcationProblem:
                break

            if "estimate_error" in klass.__dict__:
                estimate_error = True
                break

        self.estimate_error = estimate_error

        # We keep track of what solution we actually have in memory in self.state
        # for efficiency. FIXME: investigate if this actually saves us any
        # time; print out cache hits/misses in self.load_solution
        self.state_id = (None, None)

        initev.end()

        task = None
        while True:
            self.collect()

            if task is None:
                task = self.fetch_task()

            if isinstance(task, QuitTask):
                runev.end()
                self.report_profile()
                self.log("Quitting gracefully")
                return
            else:
                self.log("Executing task %s" % task)
                task = self.callbacks[task.__class__](task)
                self.log("Memory used: %s" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return

    def construct_deflation(self, parameters):
        if self.deflation is None:
            self.deflation = ShiftedDeflation(self.problem, power=2, shift=1)
        self.deflation.set_parameters(parameters.constants)

    def log(self, msg, warning=False):
        DefconThread.log(self, msg, master=False, warning=warning)

    def fetch_task(self):
        with Event("fetching task"):
            self.log("Fetching task")
            task = self.mastercomm.bcast(None)
        return task

    def fetch_response(self, block=False):
        with Event("fetching response"):
            self.log("Fetching response")
            response = self.mastercomm.bcast(None)

            if block:
                self.mastercomm.barrier()

        return response

    def send_response(self, response):
        with Event("sending response"):
            if self.teamrank == 0:
                self.log("Sending response %s" % response)
                self.worldcomm.send(response, dest=0, tag=self.responsetag)

    def load_solution(self, oldparams, branchid, newparams):
        with Event("loading solution"):
            if (oldparams, branchid) == self.state_id:
                # We already have it in memory
                return

            if oldparams is None:
                # We're dealing with an initial guess
                guess = self.problem.initial_guess(self.function_space, newparams, branchid)
                assert guess is not None
                self.state.assign(guess)
                self.state_id = (oldparams, branchid)
                return

            # We need to load from disk.
            fetched = self.io.fetch_solutions(oldparams, [branchid])
            self.state.assign(fetched[0])
            self.state_id = (oldparams, branchid)

        return

    def load_parameters(self, params):
        self.parameters.update_constants(params)

    def compute_functionals(self, solution):
        with Event("computing functionals"):
            funcs = []
            for functional in self.functionals:
                func = functional[0]
                j = func(solution, self.parameters.constants)
                try:
                    j = float(j)
                except:
                    raise ValueError("Functional %s has returned something that can't be cast to a float" % func)
                funcs.append(j)

        return funcs

    def deflation_task(self, task):
        deflev = Event("deflation task")
        deflev.begin()
        # First, load trivial solutions
        if self.trivial_solutions is None:
            self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, task.freeindex)

        # Set up the problem
        with Event("deflation: loading"):
            self.load_solution(task.oldparams, task.branchid, task.newparams)
            out = self.problem.transform_guess(self.state, task, self.io); assert out is None

            self.load_parameters(task.newparams)
            other_solutions = self.io.fetch_solutions(task.newparams, task.ensure_branches)

        with Event("deflation: boundary conditions"):
            bcs = self.problem.boundary_conditions(self.function_space, task.newparams)

        # Deflate and solve
        with Event("deflation: solve"):
            self.log("Deflating other branches %s" % task.ensure_branches)
            self.deflation.deflate(other_solutions + self.trivial_solutions)
            (success, iters) = newton(self.residual, self.jacobian, self.state, bcs,
                             task.newparams,
                             self.problem,
                             self.problem.solver_parameters(task.newparams, task),
                             self.teamno, self.deflation, self.dm)

            if iters == 0 and success:
                self.log("Appear to have detected trivial branch.")
                success = False

        self.state_id = (None, None) # not sure if it is a solution we care about yet

        # Get the functionals now, so we can send them to the master.
        with Event("deflation: functionals"):
            if success: functionals = self.compute_functionals(self.state)
            else: functionals = None

        # If the user has supplied a method to calculate the error in a given functional,
        # then let's do that now
        error_estimates = None
        if self.estimate_error and success:
            error_estimates = [None] * len(self.functionals)
            with Event("deflation: error estimation"):
                for (idx, functional) in enumerate(self.functionals):
                    try:
                        J_ufl = functional[3]
                    except IndexError:
                        continue
                    if J_ufl is None:
                        continue

                    error_estimates[idx] = self.problem.estimate_error(self.residual, J_ufl(self.state, task.newparams), self.state, bcs, task.newparams)


        with Event("deflation: sending"):
            response = Response(task.taskid, success=success, data={"functionals": functionals, "iterations": iters, "error_estimates": error_estimates})
            self.send_response(response)

        if not success:
            # that's it, move on
            deflev.end()
            return

        # Get a Response from the master telling us if we should go ahead or not
        with Event("deflation: receiving"):
            response = self.fetch_response()

        if not response.success:
            # the master has instructed us not to bother with this solution.
            # move on.
            deflev.end()
            return
        branchid = response.data["branchid"]

        # We do care about this solution, so record the fact we have it in memory
        self.state_id = (task.newparams, branchid)
        # Save it to disk with the I/O module
        with Event("deflation: saving"):
            self.log("Found new solution at parameters %s (branchid=%s) with functionals %s" % (task.newparams, branchid, functionals))
            self.problem.monitor(task.newparams, branchid, self.state, functionals)
            self.io.save_solution(self.state, functionals, task.newparams, branchid)
            self.log("Saved solution to %s to disk" % task)

        # We want to add one more synchronisation point with master, so that it doesn't go
        # haring off with this solution until it's written to disk (i.e. now)
        with Event("deflation: receiving"):
            response = self.fetch_response(block=True)

        # Automatically start onto the continuation
        newparams = self.parameters.next(task.newparams, task.freeindex)
        if newparams is not None:
            task = ContinuationTask(taskid=task.taskid,
                                    oldparams=task.newparams,
                                    freeindex=task.freeindex,
                                    branchid=branchid,
                                    newparams=newparams,
                                    direction=+1)
            task.ensure(response.data["ensure_branches"])
            deflev.end()
            return task
        else:
            # Reached the end of the continuation, don't want to continue, move on
            deflev.end()
            return

    def continuation_task(self, task):
        contev = Event("continuation task")
        contev.begin()

        # Check for trivial solutions
        if self.trivial_solutions is None:
            self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, task.freeindex)

        # Set up the problem

        # We want each (branch, freeindex) pairing to have a unique branch id,
        # to allow for one HDF5 file per branchid (otherwise two teams could
        # write to the same file, causing a race condition). This case occurs
        # when we do multi-parameter continuation, and need to continue a branch
        # in another direction -- we want to load the old branch, but store it
        # with a new branchid.
        with Event("continuation: loading"):
            if hasattr(task, 'source_branchid'):
                self.load_solution(task.oldparams, task.source_branchid, task.newparams)
            else:
                self.load_solution(task.oldparams, task.branchid, task.newparams)

            out = self.problem.transform_guess(self.state, task, self.io); assert out is None
            self.load_parameters(task.newparams)
            other_solutions = self.io.fetch_solutions(task.newparams, task.ensure_branches)

        with Event("continuation: boundary conditions"):
            bcs = self.problem.boundary_conditions(self.function_space, task.newparams)

        self.prevstate.assign(self.state)
        with Event("continuation: predictor"):
            hint = self.problem.predict(self.problem, self.state, task.oldparams, task.newparams, task.hint)

        # Try to solve it
        with Event("continuation: solve"):
            self.log("Deflating other branches %s" % task.ensure_branches)
            self.deflation.deflate(other_solutions + self.trivial_solutions)
            (success, iters) = newton(self.residual, self.jacobian, self.state, bcs,
                             task.newparams,
                             self.problem,
                             self.problem.solver_parameters(task.newparams, task),
                             self.teamno, self.deflation, self.dm)

        if not success:
            functionals = None
            self.state_id = (None, None)
            avg = [0.5*(x + y) for (x, y) in zip(task.oldparams, task.newparams)]
            self.log("Attempting average of parameters: %s" % (avg,))

            self.state.assign(self.prevstate)
            with Event("continuation: predictor"):
                hint = self.problem.predict(self.problem, self.state, task.oldparams, avg, task.hint)

            # average parameters and try again
            self.load_parameters(avg)
            self.deflation.deflate(other_solutions + self.trivial_solutions)
            (success_, iters) = newton(self.residual, self.jacobian, self.state, bcs,
                             avg,
                             self.problem,
                             self.problem.solver_parameters(avg, task, averaging=True),
                             self.teamno, self.deflation, self.dm)

            if not success_:
                self.log("Averaging failed.")
                success = False
            else:
                self.log("Averaging half-succeeded; using as initial guess for desired step")
                task.hint = hint
                with Event("continuation: predictor"):
                    hint = self.problem.predict(self.problem, self.state, avg, task.newparams, task.hint)
                self.load_parameters(task.newparams)
                self.deflation.deflate(other_solutions + self.trivial_solutions)
                (success, iters) = newton(self.residual, self.jacobian, self.state, bcs,
                                 task.newparams,
                                 self.problem,
                                 self.problem.solver_parameters(task.newparams, task, averaging=True),
                                 self.teamno, self.deflation, self.dm)
                if success:
                    self.log("Averaging succeeded!")
                else:
                    self.log("Averaging half-failed.")


        go_backwards = False
        if success:
            self.state_id = (task.newparams, task.branchid)

            # Save it to disk with the I/O module
            with Event("continuation: functionals"):
                functionals = self.compute_functionals(self.state)
                self.problem.monitor(task.newparams, task.branchid, self.state, functionals)

            with Event("continuation: saving"):
                self.io.save_solution(self.state, functionals, task.newparams, task.branchid)

            # Compute the distance between the previous solution and
            # new solution. If it's suspiciously high, instruct the master process
            # to send a continuation backwards: we may have inadvertently jumped branch
            sqdist = backend.assemble(self.problem.squared_norm(self.state, self.prevstate, task.newparams))
            if task.prevsqdist is not None and sqdist > 5*task.prevsqdist and self.continue_backwards:
                self.log("Size of previous update: %s" % task.prevsqdist)
                self.log("Size of current  update: %s" % sqdist)
                self.log("Size of update on branchid=%d suspiciously large. Inserting continuation task backwards." % task.branchid)
                go_backwards = True
        else:
            functionals = None
            self.state_id = (None, None)

        # If the user has supplied a method to calculate the error in a given functional,
        # then let's do that now
        error_estimates = None
        if self.estimate_error and success:
            error_estimates = [None] * len(self.functionals)
            with Event("deflation: error estimation"):
                for (idx, functional) in enumerate(self.functionals):
                    try:
                        J_ufl = functional[3]
                    except IndexError:
                        continue
                    if J_ufl is None:
                        continue

                    error_estimates[idx] = self.problem.estimate_error(self.residual, J_ufl(self.state, task.newparams), self.state, bcs, task.newparams)


        response = Response(task.taskid, success=success, data={"functionals": functionals, "go_backwards": go_backwards, "error_estimates": error_estimates})
        with Event("continuation: sending"):
            self.send_response(response)

        if not success:
            # Continuation didn't work. Move on.
            # FIXME: we could make this adaptive; try halving the step and doing
            # two steps?
            contev.end()
            return

        if task.direction > 0:
            newparams = self.parameters.next(task.newparams, task.freeindex)
        else:
            newparams = self.parameters.previous(task.newparams, task.freeindex)

        if newparams is None:
            # we have no more continuation to do, move on.
            contev.end()
            return
        else:
            with Event("continuation: receiving"):
                response = self.fetch_response()
            task = ContinuationTask(taskid=task.taskid,
                                    oldparams=task.newparams,
                                    freeindex=task.freeindex,
                                    branchid=task.branchid,
                                    newparams=newparams,
                                    direction=task.direction,
                                    hint=hint)
            task.ensure(response.data["ensure_branches"])
            task.prevsqdist = sqdist

            contev.end()
            return task

    def stability_task(self, task):
        stabev = Event("stability task")
        stabev.begin()

        options = self.problem.solver_parameters(task.oldparams, task)
        opts = PETSc.Options()
        for k in options:
            opts[k] = options[k]

        try:
            with Event("stability: loading"):
                self.load_solution(task.oldparams, task.branchid, -1)

            self.load_parameters(task.oldparams)

            with Event("stability: solve"):
                d = self.problem.compute_stability(task.oldparams, task.branchid, self.state, hint=task.hint)

            success = True
            response = Response(task.taskid, success=success, data={"stable": d["stable"]})
        except:
            self.log("Stability task %s failed; exception follows." % task, warning=True)
            traceback.print_exc()
            success = False
            response = Response(task.taskid, success=success)

        if success:
            # Save the data to disk with the I/O module
            with Event("stability: saving"):
                self.io.save_stability(d["stable"], d.get("eigenvalues", []), d.get("eigenfunctions", []), task.oldparams, task.branchid)

        # Send the news to master.
        with Event("stability: sending"):
            self.send_response(response)

        if not success:
            # Couldn't compute stability. Likely something is wrong. Abort and get
            # another task.
            stabev.end()
            return

        # If we're successful, we expect a command from master: should we go ahead, or not?
        proceed = True
        sign = self.signs[task.freeindex]
        if task.direction > 0:
            if sign*task.oldparams[task.freeindex] >= sign*task.extent[1]:
                proceed = False
        else:
            if sign*task.oldparams[task.freeindex] <= sign*task.extent[0]:
                proceed = False

        if not proceed:
            # Master doesn't want us to continue. This is probably because the
            # ContinuationTask that needs to be finished before we can compute
            # its stability is still ongoing. We'll pick it up later.
            self.log("Stability task not proceeding (computed up to: %s, extent: %s)" % (task.oldparams, task.extent))
            stabev.end()
            return

        if task.direction > 0:
            newparams = self.parameters.next(task.oldparams, task.freeindex)
        else:
            newparams = self.parameters.previous(task.oldparams, task.freeindex)

        if newparams is not None:
            newtask = StabilityTask(taskid=task.taskid,
                                 oldparams=newparams,
                                 freeindex=task.freeindex,
                                 branchid=task.branchid,
                                 direction=task.direction,
                                 hint=d.get("hint", None))
            newtask.set_extent(task.extent)
            stabev.end()
            return newtask
        else:
            # No more continuation to do, we're finished
            stabev.end()
            return

    def report_profile(self):
        if not self.profile:
            return

        print("-" * 80)
        print("| Profiling statistics collected by team %3d" % self.teamno + " "*35 + "|")
        print("-" * 80)

        print(" " + "*"*21)
        print(" * Global statistics *")
        print(" " + "*"*21)

        total_time = Event("run").getPerfInfo()['time']
        init_time  = Event("initialisation").getPerfInfo()['time']
        fetch_time = Event("fetching task").getPerfInfo()['time']
        gc_time    = Event("garbage").getPerfInfo()['time']
        recv_time  = Event("fetching response").getPerfInfo()['time']
        send_time  = Event("sending response").getPerfInfo()['time']
        load_time  = Event("loading solution").getPerfInfo()['time']
        func_time  = Event("computing functionals").getPerfInfo()['time']
        defl_time  = Event("deflation task").getPerfInfo()['time']
        cont_time  = Event("continuation task").getPerfInfo()['time']
        stab_time  = Event("stability task").getPerfInfo()['time']
        defl_cnt   = Event("deflation task").getPerfInfo()['count']
        cont_cnt   = Event("continuation task").getPerfInfo()['count']
        stab_cnt   = Event("stability task").getPerfInfo()['count']
        print("     total time:            %12.4f s" % total_time)
        print("     initialisation:        %12.4f s (%05.2f%%)" % (init_time,  100*init_time/total_time))
        print("     garbage collection:    %12.4f s (%05.2f%%)" % (gc_time,    100*gc_time/total_time))
        print("     fetching tasks:        %12.4f s (%05.2f%%)" % (fetch_time, 100*fetch_time/total_time))
        print("     receiving responses:   %12.4f s (%05.2f%%)" % (recv_time, 100*recv_time/total_time))
        print("     sending responses:     %12.4f s (%05.2f%%)" % (send_time, 100*send_time/total_time))
        print("     loading solutions:     %12.4f s (%05.2f%%)" % (load_time, 100*load_time/total_time))
        print("     computing functionals: %12.4f s (%05.2f%%)" % (func_time, 100*func_time/total_time))
        print("     deflation    [%04d]:   %12.4f s (%05.2f%%)" % (defl_cnt, defl_time, 100*defl_time/total_time))
        print("     continuation [%04d]:   %12.4f s (%05.2f%%)" % (cont_cnt, cont_time, 100*cont_time/total_time))
        print("     stability    [%04d]:   %12.4f s (%05.2f%%)" % (stab_cnt, stab_time, 100*stab_time/total_time))

        if defl_time > 0:
            print()

            print(" " + "*"*23)
            print(" * Deflation breakdown *")
            print(" " + "*"*23)

            load_time  = Event("deflation: loading").getPerfInfo()['time']
            bc_time    = Event("deflation: boundary conditions").getPerfInfo()['time']
            solve_time = Event("deflation: solve").getPerfInfo()['time']
            recv_time  = Event("deflation: receiving").getPerfInfo()['time']
            send_time  = Event("deflation: sending").getPerfInfo()['time']
            func_time  = Event("deflation: functionals").getPerfInfo()['time']
            save_time  = Event("deflation: saving").getPerfInfo()['time']
            print("     total time:            %12.4f s" % defl_time)
            print("     time per execution:    %12.4f s" % (defl_time/defl_cnt))
            print("     loading solutions:     %12.4f s (%05.2f%%)" % (load_time, 100*load_time/defl_time))
            print("     constructing BCs:      %12.4f s (%05.2f%%)" % (bc_time, 100*bc_time/defl_time))
            print("     solve:                 %12.4f s (%05.2f%%)" % (solve_time, 100*solve_time/defl_time))
            print("     receiving responses:   %12.4f s (%05.2f%%)" % (recv_time, 100*recv_time/defl_time))
            print("     sending responses:     %12.4f s (%05.2f%%)" % (send_time, 100*send_time/defl_time))
            print("     computing functionals: %12.4f s (%05.2f%%)" % (func_time, 100*func_time/defl_time))
            print("     saving to disk:        %12.4f s (%05.2f%%)" % (save_time, 100*save_time/defl_time))

        if cont_time > 0:
            print()

            print(" " + "*"*26)
            print(" * Continuation breakdown *")
            print(" " + "*"*26)

            load_time  = Event("continuation: loading").getPerfInfo()['time']
            bc_time    = Event("continuation: boundary conditions").getPerfInfo()['time']
            solve_time = Event("continuation: solve").getPerfInfo()['time']
            recv_time  = Event("continuation: receiving").getPerfInfo()['time']
            send_time  = Event("continuation: sending").getPerfInfo()['time']
            func_time  = Event("continuation: functionals").getPerfInfo()['time']
            save_time  = Event("continuation: saving").getPerfInfo()['time']
            print("     total time:            %12.4f s" % cont_time)
            print("     time per execution:    %12.4f s" % (cont_time/cont_cnt))
            print("     loading solutions:     %12.4f s (%05.2f%%)" % (load_time, 100*load_time/cont_time))
            print("     constructing BCs:      %12.4f s (%05.2f%%)" % (bc_time, 100*bc_time/cont_time))
            print("     solve:                 %12.4f s (%05.2f%%)" % (solve_time, 100*solve_time/cont_time))
            print("     receiving responses:   %12.4f s (%05.2f%%)" % (recv_time, 100*recv_time/cont_time))
            print("     sending responses:     %12.4f s (%05.2f%%)" % (send_time, 100*send_time/cont_time))
            print("     computing functionals: %12.4f s (%05.2f%%)" % (func_time, 100*func_time/cont_time))
            print("     saving to disk:        %12.4f s (%05.2f%%)" % (save_time, 100*save_time/cont_time))

        if stab_time > 0:
            print()

            print(" " + "*"*23)
            print(" * Stability breakdown *")
            print(" " + "*"*23)

            load_time  = Event("stability: loading").getPerfInfo()['time']
            solve_time = Event("stability: solve").getPerfInfo()['time']
            recv_time  = Event("stability: receiving").getPerfInfo()['time']
            send_time  = Event("stability: sending").getPerfInfo()['time']
            save_time  = Event("stability: saving").getPerfInfo()['time']
            print("     total time:            %12.4f s" % stab_time)
            print("     time per execution:    %12.4f s" % (stab_time/stab_cnt))
            print("     loading solutions:     %12.4f s (%05.2f%%)" % (load_time, 100*load_time/stab_time))
            print("     solve:                 %12.4f s (%05.2f%%)" % (solve_time, 100*solve_time/stab_time))
            print("     receiving responses:   %12.4f s (%05.2f%%)" % (recv_time, 100*recv_time/stab_time))
            print("     sending responses:     %12.4f s (%05.2f%%)" % (send_time, 100*send_time/stab_time))
            print("     saving to disk:        %12.4f s (%05.2f%%)" % (save_time, 100*save_time/stab_time))

        print()
