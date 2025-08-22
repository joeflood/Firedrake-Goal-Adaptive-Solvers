from __future__ import absolute_import

from mpi4py import MPI
import six

import sys

from defcon.parametertools import Parameters
from defcon.thread import DefconThread
from defcon.worker import DefconWorker
from defcon.master import DefconMaster


class DeflatedContinuation(object):
    """
    This class is the main driver. It passes most of the work off
    to the DefconWorker and DefconMaster classes.
    """
    def __init__(self, problem, **kwargs):
        """
        Constructor.

        *Arguments*
          problem (:py:class:`defcon.BifurcationProblem`)
            A class representing the bifurcation problem to be solved.
          deflation (:py:class:`defcon.DeflationOperator`)
            A class defining a deflation operator.
          teamsize (:py:class:`int`)
            How many processors should coordinate to solve any individual PDE.
          verbose (:py:class:`bool`)
            Activate verbose output.
          debug (:py:class:`bool`)
            Activate debugging output.
          logfiles (:py:class:`bool`)
            Whether defcon should remap stdout/stderr to logfiles (useful for many processes).
          continue_backwards (+1 or -1)
            Whether defcon should also continue backwards when it finds a new branch with deflation.
          clear_output (:py:class:`bool`)
            Whether defcon should first clear any old output.
          gc_frequency (:py:class:`int`)
            How many solves should pass before we call the garbage collector.
            Set to a small value (e.g. 1) for very large problems, and a large value (e.g. 100)
            for very small problems. We try to do something sensible by default.
          sleep_time (:py:class:`double`)
            How long in seconds master sleeps between repeated attempts when
            probing workers for response. Negative value means busy waiting,
            positive value saves up to one core of CPU time. Unspecified or
            None results in adaptive value given as 5 percent of last response
            time but at most 1.0 second.
          profile (:py:class:`bool`)
            Whether profiling statistics should be collected.
          comm (MPI.Comm)
            The communicator that gathers all processes involved in this computation
          disable_deflation (:py:class:`bool`)
            Whether defcon should continue the existing branches instead of doing deflation.
        """

        worldcomm = kwargs.get("comm", MPI.COMM_WORLD).Dup()
        kwargs["comm"] = worldcomm

        self.problem = problem

        # Set up I/O
        io = problem.io()
        clear_output = kwargs.get("clear_output", False)
        if worldcomm.rank == 0 and clear_output:
            io.clear()

        io.construct(worldcomm)

        if worldcomm.rank == 0:
            self.thread = DefconMaster(problem, **kwargs)
        else:
            self.thread = DefconWorker(problem, **kwargs)

    def run(self, values, freeparam=None):
        """
        The main execution routine.

        *Arguments*
          values (:py:class:`dict`)
            A dictionary mapping ASCII name of parameter to list of parameter values.
            Use a list with one element for parameters you wish to fix.
          freeparam (:py:class:`str`)
            The ASCII name of the parameter on which to start the continuation.
            Additional continuation runs can be executed by overloading the `branch_found`
            routine.
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

        # Next, check parameters.

        problem_parameters = self.problem.parameters()
        assert len(problem_parameters) == len(values)

        # Apply list to concretely instantiate the values
        for param in values:
            if isinstance(values[param], (float, int)):
                values[param] = [values[param]]
            else:
                values[param] = list(values[param])

        if freeparam is None:
            if not (len(values) == 1 or max([len(values[key]) for key in values]) == 1):
                self.thread.log("Must set freeparam in this case.", warning=True)
                assert False
            freeparam = sorted(values.keys())[0]

        parameters = Parameters(problem_parameters, values)

        # If we only have one value for the parameter, don't bother continuing backwards
        if len(values[freeparam]) == 1:
            self.thread.continue_backwards = False

        # Aaaand .. run.

        self.thread.run(parameters, freeparam)

    def bifurcation_diagram(self, functional, fixed={}, style="ok", unstablestyle=None, branches=None, decide_stability=None, **kwargs):
        if self.thread.rank != 0:
            return

        parameters = self.problem.parameters()
        functionals = self.problem.functionals()
        io = self.problem.io()
        io.setup(parameters, functionals, None)

        import matplotlib.pyplot as plt
        if "linewidth" not in kwargs: kwargs["linewidth"] = 2
        if "markersize" not in kwargs: kwargs["markersize"] = 1

        # Find the functional index.
        funcindex = None
        for (i, functionaldata) in enumerate(functionals):
            if functionaldata[1] == functional:
                funcindex = i
                break
        assert funcindex is not None

        # And find the free variable index -- the one that doesn't show up in fixed.
        freeindices = list(range(len(parameters)))
        for (i, param) in enumerate(parameters):
            if param[1] in fixed:
                freeindices.remove(i)
        assert len(freeindices) == 1
        freeindex = freeindices[0]

        class DefaultExtents(object):
            def __getitem__(self, key):
                return [-float("inf"), +float("inf")]

        if branches is None:
            branches = six.moves.xrange(io.max_branch() + 1)
            extents = DefaultExtents()
        if isinstance(branches, list):
            extents = DefaultExtents()
        elif isinstance(branches, dict):
            extents = branches

        for branchid in branches:
            extent = extents[branchid]
            params = io.known_parameters(fixed, branchid)
            funcs = io.fetch_functionals(params, branchid)

            if unstablestyle is None: # don't want to plot stable or unstable points differently
                xs = []
                ys = []
                for i in six.moves.xrange(0, len(params)):
                    param = params[i]
                    func = funcs[i]

                    if extent[0] <= param[freeindex] <= extent[1]:
                        xs.append(param[freeindex])
                        ys.append(func[funcindex])

                    plt.plot(xs, ys, style, **kwargs)
            else:
                # Need to partition xs and ys into stable and unstable points
                stable_xs = []
                stable_ys = []
                unstable_xs = []
                unstable_ys = []
                for i in six.moves.xrange(0, len(params)):
                    param = params[i]
                    func = funcs[i]
                    stab = io.fetch_stability(param, [branchid])[0]

                    if extent[0] <= param[freeindex] <= extent[1]:
                        if decide_stability(stab["stable"]):
                            stable_xs.append(param[freeindex])
                            stable_ys.append(func[funcindex])
                        else:
                            unstable_xs.append(param[freeindex])
                            unstable_ys.append(func[funcindex])

                plt.plot(stable_xs, stable_ys, style, **kwargs)
                plt.plot(unstable_xs, unstable_ys, unstablestyle, **kwargs)

        plt.grid()
        plt.xlabel(parameters[freeindex][2])
        plt.ylabel(functionals[funcindex][2])

