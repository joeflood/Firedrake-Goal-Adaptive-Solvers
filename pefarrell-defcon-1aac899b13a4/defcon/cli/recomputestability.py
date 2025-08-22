from __future__ import absolute_import, print_function

import sys
import os
import glob
import gc

from petsc4py import PETSc

from defcon import backend, StabilityTask
from defcon.cli.common import fetch_bifurcation_problem
import defcon.parametertools

import six


def usage(executable):
    sys.exit("""A script that computes the stability of all solutions.

Use like

%s /path/to/my/problem.py /path/to/output
""" % (executable,))

def main(args):
    if len(args) != 3:
        usage(args[0] if len(args) > 0 else "defcon recompute-stability")

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

    z = backend.Function(Z)

    io.setup(params, functionals, Z)

    opts = PETSc.Options()

    for branchid in six.moves.xrange(io.max_branch()+1):
        params = io.known_parameters(fixed={}, branchid=branchid)
        for param in params:
            consts = [backend.Constant(p) for p in param]
            floats = [float(p) for p in param]

            solver_parameters = problem.solver_parameters(floats, StabilityTask)
            if "snes_linesearch_type" not in solver_parameters:
                opts["snes_linesearch_type"] = "basic"
            if "snes_divergence_tolerance" not in solver_parameters:
                opts["snes_divergence_tolerance"] = -1.0
            if "snes_stol" not in solver_parameters:
                opts["snes_stol"] = 0.0

            opts["pc_mg_galerkin"] = None
            opts["pc_asm_dm_subdomains"] = None

            # set the petsc options from the solver_parameters
            for k in solver_parameters:
                opts[k] = solver_parameters[k]

            solution = io.fetch_solutions(floats, [branchid])[0]
            print("Computing stability for parameters %s" % floats)
            d = problem.compute_stability(consts, branchid, solution)
            io.save_stability(d["stable"], d.get("eigenvalues", []), d.get("eigenfunctions", []), floats, branchid)
            print("parameters/branchid %s/%s: stability: %s" % (floats, branchid, d["stable"]))
            gc.collect()
