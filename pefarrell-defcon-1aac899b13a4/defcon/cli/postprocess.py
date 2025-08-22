from __future__ import absolute_import, print_function

import sys
from ast import literal_eval

from defcon import backend
from defcon.cli.common import fetch_bifurcation_problem
import six
import gc

def usage(executable):
    sys.exit("""A script that applies the postprocess routine to all solutions, or to the solutions at a specified parameter.

Use like

%s /path/to/my/problem.py /path/to/output [params]
""" % (executable,))


def main(args):
    if len(args) < 3:
        usage(args[0] if len(args) > 0 else "defcon postprocess")

    probpath = args[1]
    outputdir = args[2]
    if outputdir.endswith("/"): outputdir = outputdir[:-1]

    problem = fetch_bifurcation_problem(probpath)
    if problem is None:
        usage(args[0])

    io = problem.io(outputdir)

    mesh = problem.mesh(backend.comm_world)
    Z = problem.function_space(mesh)
    functionals = problem.functionals()
    params = problem.parameters()
    consts = [x[0] for x in params]

    io.setup(params, functionals, Z)
    params = consts

    if len(args) == 4:
        _values = literal_eval(args[3])
        def allvalues(branchid):
            return [_values]

        def allbranches():
            return io.known_branches(_values)
    else:
        def allvalues(branchid):
            return io.known_parameters(fixed={}, branchid=branchid)

        def allbranches():
            max_branch = io.max_branch() + 1
            return six.moves.xrange(max_branch)

    for branchid in allbranches():
        for values in allvalues(branchid):
            # Read solution
            try:
                solution = io.fetch_solutions(values, [branchid])[0]
            except IOError:
                continue

            # Assign constants
            for (const, val) in zip(consts, values):
                const.assign(val)

            problem.postprocess(solution, values, branchid, None)
            gc.collect()
