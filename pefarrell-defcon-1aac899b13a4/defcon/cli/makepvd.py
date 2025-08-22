from __future__ import absolute_import, print_function

import sys
import os
from ast import literal_eval

from defcon import backend
from defcon.cli.common import fetch_bifurcation_problem


def usage(executable):
    sys.exit("""A script that converts saved data to PVD.
Use like
%s /path/to/my/problem.py /path/to/output/directory values [branchids] [pvdname]
e.g.
%s /path/to/my/problem.py /path/to/output/directory "(0, 0)"
to fetch all branches at parameters (0, 0) and save them to a default PVD filename.
""" % (executable, executable))


def main(args):
    if len(args) < 4:
        usage(args[0] if len(args) > 0 else "defcon make-pvd")

    probpath = args[1]
    outputdir = args[2]
    if outputdir.endswith("/"): outputdir = outputdir[:-1]
    values = literal_eval(args[3])
    if isinstance(values, float): values = (values,)
    if not isinstance(values, list): values = [values]

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

    if len(args) > 5:
        filename = args[5]
    else:
        filename = os.path.join(outputdir, "viz", "values-%s.pvd" % (args[3],))

    if backend.__name__ == "firedrake":
        if os.path.isfile(filename) and len(args) > 5:
            # set whatever's necessary to re-use the same PVD
            mode = "a"
        else:
            mode = "w"
        pvd = backend.File(filename, mode=mode)
    else:
        pvd = backend.File(filename)

    for value in values:
        if len(args) > 4:
            branches = literal_eval(args[4])
        else:
            branches = io.known_branches(value)

        print("Known branches at %s: %s" % (value, branches))
        solutions = io.fetch_solutions(value, branches)
        for solution in solutions:
            solution.rename("Solution", "Solution")
            problem.save_pvd(solution, pvd, value)

        print("Wrote to " + filename)
