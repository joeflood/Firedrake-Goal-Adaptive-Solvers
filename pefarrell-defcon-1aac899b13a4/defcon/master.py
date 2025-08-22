from __future__ import absolute_import, print_function

from mpi4py import MPI
from numpy  import isinf
import six

import time
import resource

from defcon.bifurcationproblem import BifurcationProblem
from defcon.thread import DefconThread
from defcon.tasks import QuitTask, ContinuationTask, DeflationTask, StabilityTask, Response
from defcon.journal import FileJournal, task_to_code
from defcon.graph import DefconGraph, ProfiledDefconGraph

class DefconMaster(DefconThread):
    """
    This class implements the core logic of running deflated continuation
    in parallel.
    """
    def __init__(self, *args, **kwargs):
        DefconThread.__init__(self, *args, **kwargs)

        # Master should always collect infrequently, it never allocates
        # anything large
        self.gc_frequency = 100

        # Sleep time in seconds (negative is busy-waiting, None is adaptive)
        self.sleep_time = kwargs.get("sleep_time", None)

        # Collect profiling statistics? This decides whether we instantiate
        # a DefconGraph or a ProfiledDefconGraph.
        self.profile = kwargs.get("profile", True)

        # Continue the existing branches (do standard deflation if False)
        self.disable_deflation = kwargs.get("disable_deflation", False)

        # A map from the type of task we're dealing with to the code that handles it.
        self.callbacks = {DeflationTask:    self.deflation_task,
                          StabilityTask:    self.stability_task,
                          ContinuationTask: self.continuation_task}

    def log(self, msg, warning=False):
        DefconThread.log(self, msg, master=True, warning=warning)

    def send_task(self, task, team):
        self.log("Sending task %s to team %s" % (task, team))
        self.teamcomms[team].bcast(task)

    def send_response(self, response, team, block=False):
        self.log("Sending response %s to team %s" % (response, team))
        self.teamcomms[team].bcast(response)

        if block:
            self.teamcomms[team].barrier()

    def fetch_response(self):
        t = time.time()

        # Assume last waiting took 20 milli-seconds for first time
        if not hasattr(self, "_last_delay"):
            self._last_delay = 0.02

        # Sleep for given time or use adaptive value
        sleep_time = self.sleep_time or min(0.05*self._last_delay, 1.0)
        # Negative value means busy waiting
        if sleep_time >= 0:
            while not self.worldcomm.Iprobe(source=MPI.ANY_SOURCE, tag=self.responsetag):
                time.sleep(sleep_time)

        # Receive response (or busy-wait for it) and return it
        response = self.worldcomm.recv(source=MPI.ANY_SOURCE, tag=self.responsetag)

        # Store waiting time for future
        self._last_delay = time.time() - t

        return response

    def seed_initial_tasks(self, parameters, values, freeindex):
        # Queue initial tasks
        initialparams = parameters.floats(value=values[0], freeindex=freeindex)
        self.initialparams = initialparams

        # Send off initial tasks
        knownbranches = self.io.known_branches(initialparams)
        self.branchid_counter = len(knownbranches)
        if len(knownbranches) > 0:
            self.used_initial_guesses = False
            nguesses = len(knownbranches)
            self.log("Using %d known solutions at %s" % (nguesses, initialparams,))

            for branch in knownbranches:
                functionals = self.io.fetch_functionals([initialparams], branch)[0]
                if self.problem.continuation_filter(initialparams, branch, functionals, self.io):
                    self.insert_continuation_task(initialparams, freeindex, branch, priority=float("-inf"))
        else:
            self.log("Using user-supplied initial guesses at %s" % (initialparams,))
            self.used_initial_guesses = True
            oldparams = None
            nguesses = self.problem.number_initial_guesses(initialparams)
            for guess in list(range(nguesses))[::-1]:
                task = DeflationTask(taskid=self.taskid_counter,
                                     oldparams=oldparams,
                                     freeindex=freeindex,
                                     branchid=self.taskid_counter,
                                     newparams=initialparams)
                self.graph.push(task, float("-inf"))
                self.taskid_counter += 1

    def finished(self):
        return self.graph.all_tasks() == 0

    def debug_print(self):
        if self.debug:
            self.graph.debug(self.log)
            self.log("DEBUG: idle_teams = %s" % self.idle_teams)

        # Also, a sanity check: idle_teams and busy_teams should be a disjoint partitioning of range(self.nteams)
        busy_teams = set([self.graph.wait_tasks[key][1] for key in self.graph.wait_tasks])
        if len(set(self.idle_teams).intersection(busy_teams)) > 0:
            self.log("ALERT: intersection of idle_teams and wait_tasks: \n%s\n%s" % (self.idle_teams, [(key, str(self.graph.wait_tasks[key][0])) for key in self.wait_tasks]), warning=True)
        if set(self.idle_teams).union(busy_teams) != set(range(self.nteams)):
            self.log("ALERT: team lost! idle_teams and wait_tasks: \n%s\n%s" % (self.idle_teams, [(key, str(self.graph.wait_tasks[key][0])) for key in self.graph.wait_tasks]), warning=True)

    def run(self, parameters, freeparam):
        self.parameters = parameters
        self.freeparam = freeparam
        freeindex = self.parameters.labels.index(freeparam)

        self.configure_io(parameters)

        # List of idle teams
        self.idle_teams = list(six.moves.xrange(self.nteams))

        # Task id counter
        self.taskid_counter = 0

        # Branch id counter
        self.branchid_counter = 0

        # Should we insert stability tasks? Let's see if the user
        # has overridden the compute_stability method or not
        compute_stability = False

        for klass in self.problem.__class__.__mro__:
            if klass is BifurcationProblem:
                break

            if "compute_stability" in klass.__dict__:
                compute_stability = True
                break

        self.compute_stability = compute_stability

        if self.profile:
            self.graph = ProfiledDefconGraph(self.nteams)
        else:
            self.graph = DefconGraph()

        # We need to keep a map of parameters -> branches.
        # FIXME: make disk writes atomic and get rid of this.
        self.parameter_map = self.io.parameter_map()

        # In parallel, we might make a discovery with deflation that invalidates
        # the results of other deflations ongoing. This set keeps track of the tasks
        # whose results we need to ignore.
        self.invalidated_tasks = set()

        # A map from (branchid, freeindex) -> (min, max) parameter value known.
        # This is needed to keep stability tasks from outrunning the
        # continuation on which they depend.
        self.branch_extent = {}

        # If we're going downwards in continuation parameter, we need to change
        # signs in a few places
        self.signs = []
        self.minvals = []
        for label in self.parameters.labels:
            values = self.parameters.values[label]
            if values[0] < values[-1]:
                self.signs.append(+1)
                self.minvals.append(min)
            else:
                self.signs.append(-1)
                self.minvals.append(max)

        # We also want to keep some statistics around how many Newton iterations it took
        # for any successful deflations to succeed
        self.max_deflation = 0
        self.total_deflation_iterations = 0
        self.total_deflation_successes  = 0

        # Initialise Journal
        is_vi = False
        self.journal = FileJournal(self.io.directory, self.parameters.parameters, self.functionals, freeindex, self.signs[freeindex], is_vi)
        self.journal.setup(self.nteams, self.teamsize, min(self.parameters.values[freeparam]), max(self.parameters.values[freeparam]))
        self.journal.sweep(self.parameters.values[freeparam][0])

        # Seed initial tasks
        self.seed_initial_tasks(parameters, parameters.values[freeparam], freeindex)

        # The main master loop.
        while not self.finished():
            self.debug_print()

            # Dispatch any tasks that can be dispatched
            while self.graph.executable_tasks() > 0 and len(self.idle_teams) > 0:
                self.dispatch_task()
                self.log("Memory used: %s" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            # We can't send out any more tasks, either because we have no
            # tasks to send out or we have no free processors.
            # If we aren't waiting for anything to finish, we'll exit the loop
            # here. otherwise, we wait for responses and deal with consequences.
            if len(self.graph.waiting()) > 0:
                self.compute_sweep(freeindex, freeparam)
                self.log("Cannot dispatch any tasks, waiting for response.")
                self.collect()

                response = self.fetch_response()
                self.handle_response(response)

            self.graph.reschedule(len(self.idle_teams))

        # Finished the main loop, tell everyone to quit
        self.journal.sweep(values[-1])
        quit = QuitTask()
        for teamno in range(self.nteams):
            self.send_task(quit, teamno)
            self.journal.team_job(teamno, "q")

        # Delete self.parameter_map, to flush it to disk in case it's backed by a file
        self.io.close_parameter_map()
        del self.parameter_map

        if self.profile:
            self.graph.report_profile()
            self.report_statistics()

    def report_statistics(self):
        print("-" * 80)
        print("| Deflation statistics" + " "*57 + "|")
        print("-" * 80)
        print()
        print("    total number of successful deflations: %d" % self.total_deflation_successes)
        if self.total_deflation_successes > 0:
            avg = float(self.total_deflation_iterations) / self.total_deflation_successes
            print("    maximum number of iterations required: %d" % self.max_deflation)
            print("    average number of iterations required: %.2f" % avg)

    def handle_response(self, response):
        (task, team) = self.graph.finish(response.taskid)
        self.log("Received response %s about task %s from team %s" % (response, task, team))
        self.callbacks[task.__class__](task, team, response)

    def dispatch_task(self):
        (task, priority) = self.graph.pop()

        send = True

        if hasattr(task, 'ensure'):
            known_branches = self.parameter_map[task.newparams]

        if isinstance(task, DeflationTask):
            if len(known_branches) >= self.problem.number_solutions(task.newparams):
            # We've found all the branches the user's asked us for, let's roll.
                self.log("Master not dispatching %s because we have enough solutions" % task)
                return

            # If there's a continuation task that hasn't reached us,
            # we want to not send this task out now and look at it again later.
            # This is because the currently running task might find a branch that we will need
            # to deflate here.
            for t in self.graph.waiting(ContinuationTask):
                if task.freeindex == t.freeindex and self.signs[t.freeindex]*t.newparams[task.freeindex]<=self.signs[t.freeindex]*task.newparams[task.freeindex] and t.direction == +1:
                    send = False
                    break

        if isinstance(task, StabilityTask):
            if (task.branchid, task.freeindex) not in self.branch_extent:
                send = False
            else:
                if task.direction > 0:
                    if self.signs[task.freeindex]*task.oldparams[task.freeindex] > self.signs[task.freeindex]*self.branch_extent[(task.branchid, task.freeindex)][1]:
                        send = False
                else:
                    if self.signs[task.freeindex]*task.oldparams[task.freeindex] < self.signs[task.freeindex]*self.branch_extent[(task.branchid, task.freeindex)][0]:
                        send = False

            # Two cases if we have decided not to send it:
            # If the continuation is still ongoing, we'll defer it.
            # If not, we'll kill it, as we'll never send it.
            if not send:
                continuation_ongoing = False
                for currenttask in self.graph.waiting(ContinuationTask):
                    if currenttask.branchid == task.branchid:
                        continuation_ongoing = True

                # We also need to check the executable tasks
                if not continuation_ongoing:
                    for currenttask in self.graph.executable(ContinuationTask):
                        if currenttask.branchid == task.branchid:
                            continuation_ongoing = True

                # OK, now we have computed whether the continuation is ongoing or not.
                if not continuation_ongoing:
                    self.log("Master not dispatching %s because the continuation has ended" % task)
                    return

        if send:
            # OK, we're happy to send it out. Let's tell it about all of the
            # solutions we know about.
            if hasattr(task, 'ensure'):
                task.ensure(known_branches)
            if hasattr(task, 'extent'):
                task.set_extent(self.branch_extent[(task.branchid, task.freeindex)])

            idleteam = self.idle_teams.pop(0)
            self.send_task(task, idleteam)
            self.graph.wait(task.taskid, idleteam, task)

            if hasattr(task, 'newparams'):
                self.journal.team_job(idleteam, task_to_code(task), task.newparams, task.branchid)
            else:
                self.journal.team_job(idleteam, task_to_code(task), task.oldparams, task.branchid)
        else:
            # Best reschedule for later, as there is still pertinent information yet to come in.
            self.log("Deferring task %s." % task)
            self.graph.defer(task, priority)

    def deflation_task(self, task, team, response):
        if not response.success:
            # As is typical, deflation found nothing interesting. The team
            # is now idle.
            self.idle_team(team)

            # One more check. If this was an initial guess, and it failed, it might be
            # because the user doesn't know when a problem begins to have a nontrivial
            # branch. In this case keep trying.
            if task.oldparams is None and self.branchid_counter == 0:
                newparams = self.parameters.next(task.newparams, task.freeindex)
                if newparams is not None:
                    newtask = DeflationTask(taskid=self.taskid_counter,
                                            oldparams=task.oldparams,
                                            freeindex=task.freeindex,
                                            branchid=task.branchid,
                                            newparams=newparams)
                    newpriority = float("-inf")
                    self.graph.push(newtask, newpriority)
                    self.taskid_counter += 1
            return

        # OK. So we were successful. But, Before processing the success, we want
        # to make sure that we really want to keep this solution. After all, we
        # might have been running five deflations in parallel; if they discover
        # the same branch, we don't want them all to track it and continue it.
        # So we check to see if this task has been invalidated by an earlier
        # discovery.

        if task in self.invalidated_tasks:
            # * Send the worker the bad news.
            self.log("Task %s has been invalidated" % task)
            responseback = Response(task.taskid, success=False)
            self.send_response(responseback, team)

            # * Remove the task from the invalidated list.
            self.invalidated_tasks.remove(task)

            # * Insert a new task --- this *might* be a dupe, or it might not
            #   be! We need to try it again to make sure. If it is a dupe, it
            #   won't discover anything; if it isn't, hopefully it will discover
            #   the same (distinct) solution again.
            if task.oldparams is not None:
                priority = self.signs[task.freeindex]*task.newparams[task.freeindex]
            else:
                priority = float("-inf")
            self.graph.push(task, priority)

            # The worker is now idle.
            self.idle_team(team)
            return

        # OK, we're good! The search succeeded and nothing has invalidated it.
        # In this case, we want the master to
        # * Do some stats around how many Newton iterations it took to succeed
        self.total_deflation_successes += 1
        self.total_deflation_iterations += response.data["iterations"]
        self.max_deflation = max(self.max_deflation, response.data["iterations"])

        # * Record any currently ongoing searches that this discovery
        #   invalidates.
        for othertask in self.graph.waiting(DeflationTask):
            self.log("Invalidating %s" % othertask)
            self.invalidated_tasks.add(othertask)

        # * Allocate a new branch id for the discovered branch.
        branchid = self.branchid_counter
        self.branchid_counter += 1

        responseback = Response(task.taskid, success=True, data={"branchid": branchid})
        self.send_response(responseback, team)

        # * Record the branch extents
        for freeindex in range(len(self.parameters.labels)):
            self.branch_extent[(branchid, freeindex)] = [task.newparams[freeindex], task.newparams[freeindex]]

        # * Record this new solution in the journal
        self.journal.entry(team, task.oldparams, branchid, task.newparams, response.data["functionals"], False)

        if not self.disable_deflation:
            # * Insert a new deflation task, to seek again with the same settings.
            newtask = DeflationTask(taskid=self.taskid_counter,
                                    oldparams=task.oldparams,
                                    freeindex=task.freeindex,
                                    branchid=task.branchid,
                                    newparams=task.newparams)
            if task.oldparams is not None:
                newpriority = self.signs[task.freeindex]*newtask.newparams[task.freeindex]
            else:
                newpriority = float("-1000000")

            self.graph.push(newtask, newpriority)
            self.taskid_counter += 1

        # * Record that the worker team is now continuing that branch,
        # if there's continuation to be done.
        newparams = self.parameters.next(task.newparams, task.freeindex)
        if newparams is not None:
            conttask = ContinuationTask(taskid=task.taskid,
                                        oldparams=task.newparams,
                                        freeindex=task.freeindex,
                                        branchid=branchid,
                                        newparams=newparams,
                                        direction=+1)
            self.graph.wait(task.taskid, team, conttask)
            self.log("Waiting on response for %s" % conttask)
            # Write to the journal, saying that this team is now doing continuation.
            self.journal.team_job(team, "c", task.newparams, branchid)
            next_known_branches = self.parameter_map[newparams]
        else:
            # It's at the end of the continuation, there's no more continuation
            # to do. Mark the team as idle.
            next_known_branches = []
            self.idle_team(team)

        # * Now let's ask the user if they want to do anything special,
        #   e.g. insert new tasks going in another direction.
        userin = ContinuationTask(taskid=self.taskid_counter,
                                  oldparams=task.newparams,
                                  freeindex=task.freeindex,
                                  branchid=branchid,
                                  newparams=newparams,
                                  direction=+1)
        self.taskid_counter += 1
        self.process_user_tasks(userin)

        # * If we want to continue backwards, well, let's add that task too
        if self.continue_backwards:
            newparams = self.parameters.previous(task.newparams, task.freeindex)
            back_branchid = self.branchid_counter
            self.branchid_counter += 1

            if newparams is not None:
                bconttask = ContinuationTask(taskid=self.taskid_counter,
                                            oldparams=task.newparams,
                                            freeindex=task.freeindex,
                                            branchid=back_branchid,
                                            newparams=newparams,
                                            direction=-1)
                bconttask.source_branchid = branchid
                newpriority = self.signs[task.freeindex]*newparams[task.freeindex]
                self.graph.push(bconttask, newpriority)
                self.taskid_counter += 1

        if not self.disable_deflation:
            # * If we've found new solutions for the initial parameters, try all of
            #   the initial guesses again (otherwise the behaviour is different
            #   between one worker and many workers if you have multiple initial
            #   guesses)
            if task.newparams == self.initialparams and self.used_initial_guesses:
                nguesses = self.problem.number_initial_guesses(self.initialparams)
                for guess in range(nguesses):
                    # skip the one we've already inserted an extra task for
                    if task.oldparams is None and task.branchid == guess: continue

                    newtask = DeflationTask(taskid=self.taskid_counter,
                                            oldparams=None,
                                            freeindex=task.freeindex,
                                            branchid=guess,
                                            newparams=self.initialparams)
                    self.graph.push(newtask, float("-inf"))
                    self.taskid_counter += 1

        # We'll also make sure that any other DeflationTasks in the queue
        # that have these parameters know about the existence of this branch.
        old_parameter_map = self.parameter_map[task.newparams]
        self.parameter_map[task.newparams] = old_parameter_map + [branchid]

        # If the user wants us to compute stabilities, then let's
        # do that.
        if self.compute_stability:
            stabtask = StabilityTask(taskid=self.taskid_counter,
                                     oldparams=task.newparams,
                                     freeindex=task.freeindex,
                                     branchid=branchid,
                                     direction=+1,
                                     hint=None)
            newpriority = self.signs[task.freeindex]*stabtask.oldparams[task.freeindex]
            self.graph.push(stabtask, newpriority)
            self.taskid_counter += 1

            if self.continue_backwards and newparams is not None:
                stabtask = StabilityTask(taskid=self.taskid_counter,
                                        oldparams=newparams,
                                        freeindex=task.freeindex,
                                        branchid=back_branchid,
                                        direction=-1,
                                        hint=None)
                newpriority = self.signs[task.freeindex]*stabtask.oldparams[task.freeindex]
                self.graph.push(stabtask, newpriority)
                self.taskid_counter += 1

        # Phew! What a lot of bookkeeping. The only thing left to do is wait until the
        # worker has finished his I/O.
        response = Response(taskid=task.taskid, success=True, data={"ensure_branches": next_known_branches})
        self.send_response(response, team, block=True)
        return

    def continuation_task(self, task, team, response):
        if not response.success:
            # We tried to continue a branch, but the continuation died. Oh well.
            # The team is now idle.
            self.log("Continuation task of team %d on branch %d failed at parameters %s." % (team, task.branchid, task.newparams), warning=True)
            self.idle_team(team)
            return

        # Before doing anything else, send the message to the worker
        if task.direction > 0:
            newparams = self.parameters.next(task.newparams, task.freeindex)
        else:
            newparams = self.parameters.previous(task.newparams, task.freeindex)

        if newparams is None:
            # No more continuation to do, the team is now idle.
            self.idle_team(team)
        else:
            next_known_branches = self.parameter_map[newparams]
            responseback = Response(taskid=task.taskid, success=True, data={"ensure_branches": next_known_branches})
            self.send_response(responseback, team)

        # Record success.
        self.journal.entry(team, task.oldparams, task.branchid, task.newparams, response.data["functionals"], True)

        # Update the parameter -> branchid map
        old_parameter_map = self.parameter_map[task.newparams]
        self.parameter_map[task.newparams] = [task.branchid] + old_parameter_map

        # Update the branch extent.
        if (task.branchid, task.freeindex) not in self.branch_extent:
            if task.direction > 0:
                self.branch_extent[(task.branchid, task.freeindex)] = [task.oldparams[task.freeindex], task.newparams[task.freeindex]]
            else:
                self.branch_extent[(task.branchid, task.freeindex)] = [task.newparams[task.freeindex], task.oldparams[task.freeindex]]
        else:
            if task.direction > 0:
                self.branch_extent[(task.branchid, task.freeindex)][1] = task.newparams[task.freeindex]
            else:
                self.branch_extent[(task.branchid, task.freeindex)][0] = task.newparams[task.freeindex]

        # The worker will keep continuing, record that fact

        if newparams is not None:
            conttask = ContinuationTask(taskid=task.taskid,
                                        oldparams=task.newparams,
                                        freeindex=task.freeindex,
                                        branchid=task.branchid,
                                        newparams=newparams,
                                        direction=task.direction)
            self.graph.wait(task.taskid, team, conttask)
            self.log("Waiting on response for %s" % conttask)
            self.journal.team_job(team, task_to_code(conttask), newparams, task.branchid)

        # If the worker has instructed us to insert a continuation task
        # going backwards, then do it. This arises if the worker thinks
        # the solutions have changed too much -- we may have inadvertently
        # jumped from one (mathematical) branch to another.

        if response.data["go_backwards"] and self.continue_backwards:
            back_branchid = self.branchid_counter
            self.branchid_counter += 1
            backtask = ContinuationTask(taskid=self.taskid_counter,
                                        oldparams=task.newparams,
                                        freeindex=task.freeindex,
                                        branchid=back_branchid,
                                        newparams=task.oldparams,
                                        direction=-1*task.direction)
            backtask.source_branchid = task.branchid
            self.taskid_counter += 1
            backpriority = self.signs[task.freeindex]*backtask.newparams[task.freeindex]
            self.graph.push(backtask, backpriority)

            if self.compute_stability:
                backstabtask = StabilityTask(taskid=self.taskid_counter,
                                        oldparams=task.oldparams,
                                        freeindex=task.freeindex,
                                        branchid=back_branchid,
                                        direction=-1*task.direction,
                                        hint=None)
                backstabpriority = self.signs[task.freeindex]*backstabtask.oldparams[task.freeindex]
                self.graph.push(backstabtask, backstabpriority)
                self.taskid_counter += 1

        if not self.disable_deflation:
            # If we've found new solutions for the initial parameters, try all of
            # the initial guesses again (otherwise the behaviour is different
            # between one worker and many workers if you have multiple initial
            # guesses)
            if task.newparams == self.initialparams and self.used_initial_guesses:
                nguesses = self.problem.number_initial_guesses(self.initialparams)
                for guess in range(nguesses):
                    # skip the one we've already inserted an extra task for
                    if task.oldparams is None and task.branchid == guess: continue

                    newtask = DeflationTask(taskid=self.taskid_counter,
                                            oldparams=None,
                                            freeindex=task.freeindex,
                                            branchid=guess,
                                            newparams=self.initialparams)
                    self.graph.push(newtask, float("-inf"))
                    self.taskid_counter += 1

        # Now let's ask the user if they want to do anything special,
        # e.g. insert new tasks going in another direction.
        userin = ContinuationTask(taskid=self.taskid_counter,
                                  oldparams=task.newparams,
                                  freeindex=task.freeindex,
                                  branchid=task.branchid,
                                  newparams=newparams,
                                  direction=+1)
        self.taskid_counter += 1
        self.process_user_tasks(userin)

        if not self.disable_deflation:
            # Whether there is another continuation task to insert or not,
            # we have a deflation task to insert.
            if hasattr(task, 'source_branchid'):
                new_branchid = task.source_branchid
            else:
                new_branchid = task.branchid
            newtask = DeflationTask(taskid=self.taskid_counter,
                                    oldparams=task.oldparams,
                                    freeindex=task.freeindex,
                                    branchid=new_branchid,
                                    newparams=task.newparams)
            self.taskid_counter += 1
            newpriority = self.signs[task.freeindex]*newtask.newparams[task.freeindex]
            self.graph.push(newtask, newpriority)

    def stability_task(self, task, team, response):
        if not response.success:
            self.idle_team(team)
            return

        # Check if this is the end of the known data: if it is,
        # don't continue
        proceed = True
        sign = self.signs[task.freeindex]
        if task.direction > 0:
            if sign*task.oldparams[task.freeindex] >= sign*task.extent[1]:
                proceed = False
        else:
            if sign*task.oldparams[task.freeindex] <= sign*task.extent[0]:
                proceed = False

        if not proceed:
            # We've told the worker to stop. The team is now idle.
            self.idle_team(team)

            # If the continuation is still ongoing, we'll insert another stability
            # task into the queue.
            continuation_ongoing = False

            if task.extent != self.branch_extent[(task.branchid, task.freeindex)]:
                continuation_ongoing = True

            if not continuation_ongoing:
                for currenttask in self.graph.waiting(ContinuationTask):
                    if currenttask.branchid == task.branchid:
                        continuation_ongoing = True
                        break

            if not continuation_ongoing:
                for currenttask in self.graph.executable(ContinuationTask):
                    if currenttask.branchid == task.branchid:
                        continuation_ongoing = True
                        break

            if continuation_ongoing:
                self.log("Stability task has finished early. Inserting another.")
                # Insert another StabilityTask into the queue.
                if task.direction > 0:
                    newparams = self.parameters.next(task.oldparams, task.freeindex)
                else:
                    newparams = self.parameters.previous(task.oldparams, task.freeindex)

                if newparams is not None:
                    nexttask = StabilityTask(taskid=task.taskid,
                                             branchid=task.branchid,
                                             freeindex=task.freeindex,
                                             oldparams=newparams,
                                             direction=task.direction,
                                             hint=None)
                    newpriority = self.signs[task.freeindex]*nexttask.oldparams[task.freeindex]
                    self.graph.push(nexttask, newpriority)
            else:
                self.log("Stability task has finished and continuation not ongoing. Not inserting another.")
            return

        # The worker will keep continuing, record that fact
        if task.direction > 0:
            newparams = self.parameters.next(task.oldparams, task.freeindex)
        else:
            newparams = self.parameters.previous(task.oldparams, task.freeindex)

        if newparams is not None:
            nexttask = StabilityTask(taskid=task.taskid,
                                     branchid=task.branchid,
                                     freeindex=task.freeindex,
                                     oldparams=newparams,
                                     direction=task.direction,
                                     hint=None)
            nexttask.set_extent(task.extent)
            self.graph.wait(task.taskid, team, nexttask)
            self.log("Waiting on response for %s" % nexttask)
            self.journal.team_job(team, task_to_code(nexttask), nexttask.oldparams, task.branchid)
        else:
            self.idle_team(team)

    def insert_continuation_task(self, oldparams, freeindex, branchid, priority):
        newparams = self.parameters.next(oldparams, freeindex)
        branchid  = int(branchid)
        if newparams is not None:
            task = ContinuationTask(taskid=self.taskid_counter,
                                    oldparams=oldparams,
                                    freeindex=freeindex,
                                    branchid=branchid,
                                    newparams=newparams,
                                    direction=+1)
            self.graph.push(task, priority)
            self.taskid_counter += 1

            if self.compute_stability:
                stabtask = StabilityTask(taskid=self.taskid_counter,
                                         oldparams=oldparams,
                                         freeindex=freeindex,
                                         branchid=branchid,
                                         direction=+1,
                                         hint=None)
                newpriority = self.signs[freeindex]*stabtask.oldparams[freeindex]
                self.graph.push(stabtask, newpriority)
                self.taskid_counter += 1

            if self.continue_backwards:
                newparams = self.parameters.previous(oldparams, freeindex)
                back_branchid = self.branchid_counter
                self.branchid_counter += 2 # +2 instead of +2 to maintain sign convention
                                           # that even is advancing in parameter, odd is going backwards

                if newparams is not None:
                    task = ContinuationTask(taskid=self.taskid_counter,
                                            oldparams=oldparams,
                                            freeindex=freeindex,
                                            branchid=back_branchid,
                                            newparams=newparams,
                                            direction=-1)
                    task.source_branchid = branchid
                    self.log("Scheduling task: %s" % task)
                    self.graph.push(task, priority)
                    self.taskid_counter += 1

                    if self.compute_stability:
                        stabtask = StabilityTask(taskid=self.taskid_counter,
                                                 oldparams=newparams,
                                                 freeindex=freeindex,
                                                 branchid=back_branchid,
                                                 direction=-1,
                                                 hint=None)
                        newpriority = self.signs[freeindex]*stabtask.oldparams[freeindex]
                        self.graph.push(stabtask, newpriority)
                        self.taskid_counter += 1

    def idle_team(self, team):
        self.idle_teams.append(team)
        self.journal.team_job(team, "i")

    def compute_sweep(self, freeindex, freeparam):
        minparams = self.graph.sweep(self.minvals[freeindex], freeindex)
        if minparams is not None:
            prevparams = self.parameters.previous(minparams, freeindex)
            if prevparams is not None:
                minwait = prevparams[freeindex]

                tot_solutions = self.problem.number_solutions(minparams)
                if isinf(tot_solutions): tot_solutions = '?'
                num_solutions = len(self.io.known_branches(minparams))
                self.log("Deflation sweep completed <= %14.12e (%s/%s solutions)." % (minwait, num_solutions, tot_solutions))
                # Write to the journal saying where we've completed our sweep up to.
                self.journal.sweep(minwait)

        elif len(self.graph.executable(StabilityTask)) > 0:
            # We have only stability tasks to do.
            minwait = self.parameters.values[freeparam][-1]
            self.journal.sweep(minwait)

    def process_user_tasks(self, userin):
        user_tasks = self.problem.branch_found(userin)
        for (j, user_task) in enumerate(user_tasks):
            assert user_task.taskid == userin.taskid + j + 1

            if isinstance(user_task, StabilityTask):
                self.log("Warning: disregarding user-inserted %s. Stability tasks are inserted automatically." % user_task, warning=True)
                continue

            if hasattr(user_task, 'newparams'):
                # Set the new parameter values to be investigated
                if user_task.direction == +1:
                    user_task.newparams = self.parameters.next(user_task.oldparams, user_task.freeindex)
                else:
                    user_task.newparams = self.parameters.previous(user_task.oldparams, user_task.freeindex)

                if user_task.newparams is None:
                    self.log("Warning: disregarding user-inserted task %s" % user_task, warning=True)
                    continue

            if hasattr(user_task, 'branchid'):
                # Give a new branchid
                user_task.source_branchid = user_task.branchid
                user_task.branchid = self.branchid_counter
                self.branchid_counter += 1

            priority = user_task.oldparams[user_task.freeindex]
            self.log("Registering user-inserted task %s" % user_task)
            self.graph.push(user_task, priority)
            self.taskid_counter += 1

            if isinstance(user_task, ContinuationTask) and self.compute_stability:
                stab_task = StabilityTask(taskid=self.taskid_counter,
                                          oldparams=user_task.newparams,
                                          freeindex=user_task.freeindex,
                                          branchid=user_task.branchid,
                                          direction=user_task.direction,
                                          hint=None)
                self.taskid_counter += 1
                self.log("Registering automatically generated task %s" % stab_task)
                self.graph.push(stab_task, priority)

