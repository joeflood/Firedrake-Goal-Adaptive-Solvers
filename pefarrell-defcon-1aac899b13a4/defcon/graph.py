from __future__ import absolute_import, print_function

from petsc4py import PETSc # for logging
import six

from heapq import heappush, heappop

from defcon.tasks import ContinuationTask, DeflationTask, ArclengthTask, StabilityTask


class DefconGraph(object):
    """
    A task graph for defcon.

    FIXME: This design is too defcon-specific. It would be nice to have
    something a little more generic. But it works for now.
    """

    def __init__(self):
        # Data structures for lists of tasks in various states
        self.new_tasks       = [] # tasks yet to be dispatched
        self.deferred_tasks  = [] # tasks we cannot dispatch yet because we're expecting more info
        self.wait_tasks      = {} # tasks dispatched, waiting to hear back
        self.stability_tasks = [] # stability tasks, kept with a lower priority than others

    def push(self, task, priority):
        if isinstance(task, StabilityTask):
            queue = self.stability_tasks
        else:
            queue = self.new_tasks

        heappush(queue, (priority, task))

    def wait(self, taskid, team, task):
        self.wait_tasks[taskid] = (task, team)

    def finish(self, taskid):
        (task, team) = self.wait_tasks[taskid]
        del self.wait_tasks[taskid]
        return (task, team)

    def pop(self):
        if len(self.stability_tasks) > 0:
            (priority, task) = heappop(self.stability_tasks)
        elif len(self.new_tasks) > 0:
            (priority, task) = heappop(self.new_tasks)
        else:
            raise IndexError

        return (task, priority)

    def defer(self, task, priority):
        queue = self.deferred_tasks
        heappush(queue, (priority, task))

    def executable_tasks(self):
        return len(self.new_tasks) + len(self.stability_tasks)

    def all_tasks(self):
        return len(self.new_tasks) + len(self.stability_tasks) + len(self.wait_tasks) + len(self.deferred_tasks)

    def waiting(self, cls=None):
        """
        Get the waiting tasks of type cls.
        """
        if cls is None:
            out = [task for (task, team) in self.wait_tasks.values()]
        else:
            out = [task for (task, team) in self.wait_tasks.values() if isinstance(task, cls)]
        return out

    def executable(self, cls=None):
        """
        Get the executable tasks of type cls.
        """
        if cls is None:
            out = [task for (priority, task) in self.new_tasks + self.stability_tasks]
        else:
            out = [task for (priority, task) in self.new_tasks + self.stability_tasks if isinstance(task, cls)]
        return out


    def reschedule(self, N):
        """
        Reschedule N deferred tasks.

        Maybe we deferred some deflation tasks because we didn't have enough 
        information to judge if they were worthwhile. Now we must reschedule.
        """
        for i in range(N):
            try:
                (priority, task) = heappop(self.deferred_tasks)
                if isinstance(task, StabilityTask):
                    heappush(self.stability_tasks, (priority, task))
                else:
                    heappush(self.new_tasks, (priority, task))
            except IndexError: break
        return

    def debug(self, log):
        log("DEBUG: new_tasks = %s" % [(priority, str(x)) for (priority, x) in self.new_tasks])
        log("DEBUG: wait_tasks = %s" % [(key, str(self.wait_tasks[key][0]), self.wait_tasks[key][1]) for key in self.wait_tasks])
        log("DEBUG: deferred_tasks = %s" % [(priority, str(x)) for (priority, x) in self.deferred_tasks])
        log("DEBUG: stability_tasks = %s" % [(priority, str(x)) for (priority, x) in self.stability_tasks])

    def sweep(self, minvals, freeindex):
        waiting_values = [wtask[0].oldparams for wtask in self.wait_tasks.values() if isinstance(wtask[0], DeflationTask)]
        newtask_values = [ntask[1].oldparams for ntask in self.new_tasks if isinstance(ntask[1], DeflationTask)]
        deferred_values = [dtask[1].oldparams for dtask in self.deferred_tasks if isinstance(dtask[1], DeflationTask)]
        all_values = list(six.moves.filter(lambda x: x is not None, waiting_values + newtask_values + deferred_values))

        if len(all_values) > 0:
            minparams = minvals(all_values, key=lambda x: x[freeindex])
        else:
            minparams = None

        return minparams

class ProfiledDefconGraph(DefconGraph):
    def __init__(self, nteams):
        DefconGraph.__init__(self)
        self.nteams = nteams
        PETSc.Log.begin()

        ev = PETSc.Log.Event("graph time")
        ev.begin()

        for team in range(self.nteams):
            ev = PETSc.Log.Event("%5d: idle" % team)
            ev.begin()

        self.codes = {ContinuationTask: 'continuation',
                      DeflationTask:    'deflation',
                      StabilityTask:    'stability',
                      ArclengthTask:    'arclength'}

    def wait(self, taskid, team, task):
        DefconGraph.wait(self, taskid, team, task)

        ev = PETSc.Log.Event("%5d: idle" % team)
        ev.end()

        ev = PETSc.Log.Event("%5d: %s" % (team, self.codes[task.__class__]))
        ev.begin()

    def finish(self, taskid):
        (task, team) = self.wait_tasks[taskid]
        del self.wait_tasks[taskid]

        ev = PETSc.Log.Event("%5d: %s" % (team, self.codes[task.__class__]))
        ev.end()

        ev = PETSc.Log.Event("%5d: idle" % team)
        ev.begin()

        return (task, team)

    def report_profile(self):
        print("-" * 80)
        print("| Profiling statistics collected by master" + " "*37 + "|")
        print("-" * 80)
        for team in range(self.nteams):
            ev = PETSc.Log.Event("%5d: idle" % team)
            ev.end()

        ev = PETSc.Log.Event("graph time")
        ev.end()
        total_time = ev.getPerfInfo()['time']
        print("Total time: ", total_time)
        print()

        for team in range(self.nteams):
            print(" " + "*"*12)
            print(" * Team %3d *" % team)
            print(" " + "*"*12)

            idle_time = PETSc.Log.Event("%5d: idle" % team).getPerfInfo()['time']
            cont_time = PETSc.Log.Event("%5d: continuation" % team).getPerfInfo()['time']
            defl_time = PETSc.Log.Event("%5d: deflation" % team).getPerfInfo()['time']
            stab_time = PETSc.Log.Event("%5d: stability" % team).getPerfInfo()['time']
            arcl_time = PETSc.Log.Event("%5d: arclength" % team).getPerfInfo()['time']

            print("     idle:         %12.4f s (%05.2f%%)" % (idle_time, 100*idle_time/total_time))
            print("     continuation: %12.4f s (%05.2f%%)" % (cont_time, 100*cont_time/total_time))
            print("     deflation:    %12.4f s (%05.2f%%)" % (defl_time, 100*defl_time/total_time))
            print("     stability:    %12.4f s (%05.2f%%)" % (stab_time, 100*stab_time/total_time))
            print("     arclength:    %12.4f s (%05.2f%%)" % (arcl_time, 100*arcl_time/total_time))

            print()

