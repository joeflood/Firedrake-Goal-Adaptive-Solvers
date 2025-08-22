from __future__ import absolute_import, print_function

import sys
import gc

from defcon import backend
from defcon.cli.common import fetch_bifurcation_problem


def usage(executable):
    sys.exit("""A script that computes the functionals of all solutions.

Use like

%s /path/to/my/problem.py /path/to/output
""" % (executable,))


def main(args):
    if len(args) != 3:
        usage(args[0] if len(args) > 0 else "defcon recompute-functionals")

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

    for branchid in range(io.max_branch()+1):
        for values in io.known_parameters(fixed={}, branchid=branchid):
            # Read solution
            solution = io.fetch_solutions(values, [branchid])[0]

            # Assign constants
            for (const, val) in zip(consts, values):
                const.assign(val)

            # Compute functionals
            funcs = []
            for functional in functionals:
                func = functional[0]
                j = func(solution, consts)
                assert isinstance(j, float)
                funcs.append(j)

            # Save to disk again
            print("Saving values: %s, branchid: %d, functionals %s" % (values, branchid, funcs))
            io.save_solution(solution, funcs, values, branchid)
            gc.collect()
