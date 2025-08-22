from __future__ import absolute_import, print_function

import sys
import os
import glob
import h5py
import gc

from defcon import backend
from defcon.cli.common import fetch_bifurcation_problem
import defcon.branchio


def usage(executable):
    sys.exit("""A script that regenerates the known_parameters map in case it is deleted/unavailable.
Use like
%s /path/to/my/problem.py /path/to/output/directory
""" % (executable,))


def main(args):
    if len(args) != 3:
        usage(args[0] if len(args) > 0 else "defcon recompute-known-branches")

    probpath = args[1]
    outputdir = args[2]

    problem = fetch_bifurcation_problem(probpath)
    if problem is None:
        usage(args[0])

    io = problem.io(outputdir)

    if not isinstance(io, defcon.branchio.BranchIO):
        print("Only relevant for BranchIO.")
        return 1

    mesh = problem.mesh(backend.comm_world)
    Z = problem.function_space(mesh)
    functionals = problem.functionals()
    params = problem.parameters()
    io.setup(params, functionals, Z)
    pm = io.parameter_map()

    h5s = glob.glob(os.path.join(outputdir, "*.h5"))

    for h5 in h5s:
        basename = os.path.basename(h5)
        branchid = int(basename[9:-3])

        if backend.comm_world.size > 1:
            f = h5py.File(h5, "r", driver="mpio", comm=backend.comm_world.tompi4py())
        else:
            f = h5py.File(h5, "r")

        for key in f.keys():
            params = defcon.branchio.keytoparams(key)
            old_pm = pm[params]
            pm[params] = [branchid] + old_pm
        f.close()
        gc.collect()

    io.close_parameter_map()
