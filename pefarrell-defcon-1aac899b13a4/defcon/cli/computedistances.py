from __future__ import absolute_import, print_function

import sys
import os
import glob
import gc

from petsc4py import PETSc

from defcon import backend, StabilityTask
from defcon.cli.common import fetch_bifurcation_problem
import defcon.parametertools
from math import sqrt

import six
import pprint
from ast import literal_eval


def usage(executable):
    sys.exit("""A script that computes ||solution - (other solution)|| for a fixed branchid at a fixed parameter value.

Use like

%s /path/to/my/problem.py /path/to/output values branchid
""" % (executable,))

def main(args):
    if len(args) != 5:
        usage(args[0] if len(args) > 0 else "defcon compute-distances")

    probpath = args[1]
    outputdir = args[2]
    if outputdir.endswith("/"): outputdir = outputdir[:-1]
    values = literal_eval(args[3])
    if isinstance(values, float): values = (values,)
    baseid = int(args[4])

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

    distances = {}
    base = io.fetch_solutions(values, [baseid])[0]

    for branchid in io.known_branches(values):
        solution = io.fetch_solutions(values, [branchid])[0]
        distance = sqrt(backend.assemble(problem.squared_norm(base, solution, values)))
        distances[branchid] = distance
        gc.collect()

    print("Distances:")
    pprint.pprint(distances)
