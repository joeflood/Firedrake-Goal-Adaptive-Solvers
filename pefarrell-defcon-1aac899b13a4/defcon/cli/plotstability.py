from __future__ import absolute_import, print_function

import sys
import os
from ast import literal_eval

from defcon import backend, __default_matplotlib_backend
from defcon.cli.common import fetch_bifurcation_problem

import matplotlib


def usage(executable):
    sys.exit("""A script that plots stability data for a branch.
Use like
%s /path/to/my/problem.py /path/to/output/directory [branchids] [pngname]
e.g.
%s /path/to/my/problem.py /path/to/output/directory [0, 2]
""" % (executable, executable))


def main(args):
    if len(args) < 3:
        usage(args[0] if len(args) > 0 else "defcon plot-stability")

    probpath = args[1]
    outputdir = args[2]
    if outputdir.endswith("/"): outputdir = outputdir[:-1]
    branchids = literal_eval(args[3])
    if isinstance(branchids, int): branchids = [branchids]
    if not isinstance(branchids, list): branchids = list(branchids)

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

    if len(args) > 4:
        filename = args[4]
    else:
        filename = None

    if filename is None:
        matplotlib.use(__default_matplotlib_backend)

    import matplotlib.pyplot as plt
    plt.clf()

    allparams = set()
    for branchid in branchids:
        knownparams = io.known_parameters(fixed={}, branchid=branchid, stability=True)
        allparams = allparams.union(set(knownparams))
    print(allparams)

    evals_real = []
    evals_imag = []
    for params in sorted(allparams):
        stabs = io.fetch_stability(params, branchids, fetch_eigenfunctions=False)
        for (stab, branch) in zip(stabs, branchids):
            print("Eigenvalues for branch %s at parameters %s:" % (branch, str(params)))
            for eval_ in stab["eigenvalues"]:
                print("  %s" % eval_)

            evals = list(map(complex, stab["eigenvalues"]))
            evals_real += [l.real for l in evals]
            evals_imag += [l.imag for l in evals]

    plt.plot(evals_real, evals_imag, 'bo')
    plt.title("Eigenvalues for %s %s" % ("branch" if len(branchids) == 1 else "branches", str(args[3])))
    plt.xlabel("Real component")
    plt.ylabel("Imaginary component")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
