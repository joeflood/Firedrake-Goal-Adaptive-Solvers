from __future__ import absolute_import, print_function

import sys
import os
from ast import literal_eval

from defcon import backend
from defcon.cli.common import fetch_bifurcation_problem
from defcon.cli.recomputeknownbranches import main as recompute_known_branches


def usage(executable):
    sys.exit("""A script that makes a new output directory with a subset of the data.
This is useful for picking up from a previous run.
Use like
%s /path/to/my/problem.py /path/to/output/directory values /path/to/subset/directory
e.g.
%s /path/to/my/problem.py /path/to/output/directory "(0, 0)" /path/to/subset/directory
""" % (executable, executable))


def main(args):
    if len(args) != 5:
        usage(args[0] if len(args) > 0 else "defcon subset-output")

    probpath = args[1]
    outputdir = args[2]
    if outputdir.endswith("/"): outputdir = outputdir[:-1]
    values = literal_eval(args[3])
    if isinstance(values, float): values = (values,)
    subsetdir = args[4]
    if subsetdir.endswith("/"): subsetdir = subsetdir[:-1]

    problem = fetch_bifurcation_problem(probpath)
    if problem is None:
        usage(args[0])

    io = problem.io(outputdir)
    newio = problem.io(subsetdir)

    mesh = problem.mesh(backend.comm_world)
    Z = problem.function_space(mesh)
    functionals = problem.functionals()
    params = problem.parameters()
    consts = [x[0] for x in params]

    io.setup(params, functionals, Z)
    newio.setup(params, functionals, Z)
    params = consts

    branches = io.known_branches(values)
    print("Known branches at %s: %s" % (values, branches))
    solutions = io.fetch_solutions(values, branches)
    try:
        stabilities = io.fetch_stability(values, branches)
    except:
        stabilities = [None]*len(branches)

    for (solution, stability, branch) in zip(solutions, stabilities, branches):
        functionals = io.fetch_functionals([values], branch)[0]
        newio.save_solution(solution, functionals, values, branch)

        if stability is not None:
            newio.save_stability(stability, [], [], values, branch)

    newargs = [args[0], args[1], args[4]]
    recompute_known_branches(newargs)

