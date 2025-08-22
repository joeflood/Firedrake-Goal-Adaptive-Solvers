from __future__ import absolute_import, print_function

import sys
import glob
import gc

from defcon import backend
from defcon.cli.common import fetch_bifurcation_problem
import defcon.parametertools
import defcon.newton


def usage(executable):
    sys.exit("""A script that merges two or more output directories (e.g. from
two passes back and forth of a defcon run).

Use like

%s /path/to/my/problem.py /path/to/output1 /path/to/output2
""" % (executable,))


def main(args):
    if len(args) < 3:
        usage(args[0] if len(args) > 0 else "defcon make-pvd")

    probpath = args[1]
    outputdirs = args[2:]

    problem = fetch_bifurcation_problem(probpath)
    if problem is None:
        usage(args[0])

    mesh = problem.mesh(backend.comm_world)
    Z = problem.function_space(mesh)
    params = problem.parameters()
    consts = [x[0] for x in params]

    z = backend.Function(Z)
    w = backend.TestFunction(Z)
    F = problem.residual(z, consts, w)
    bcs = problem.boundary_conditions(Z, params)
    functionals = problem.functionals()
    io = problem.io()

    io.setup(params, functionals, Z)
    params = consts

    # First get the directories to loop over
    dirnames = set()
    for outputdir in outputdirs:
        if outputdir.endswith("/"): outputdir = outputdir[:-1]
        assert outputdir != io.directory

        thisdirs = glob.glob(outputdir + "/*=*")
        uniques  = [x.replace(outputdir + "/", "") for x in thisdirs]
        dirnames = dirnames.union(uniques)
    dirnames = sorted(list(dirnames))

    for dirname in dirnames:
        # Load the appropriate parameter values
        defcon.parametertools.parametersfromstring(params, dirname)

        solutionid = 0
        knownroots = []

        for outputdir in outputdirs:
            solutions = glob.glob(outputdir + "/" + dirname + "/*h5")
            for solution in solutions:
                gc.collect()

                # Load from disk
                with backend.HDF5File(backend.comm_world, solution, "r") as h5:
                    h5.read(z, "/solution")

                # Verify distance to other known solutions
                skip = False
                for otherroot in knownroots:
                    d = backend.assemble(problem.squared_norm(otherroot, z, params))
                    if abs(d) < 1.0e-10:
                        skip = True
                        break
                if skip:
                    print("Skipping ", solution)
                    continue

                # OK, it's a new solution. Let's solve the equations (just in case)
                # and add it to the known solutions.
                problemclass = problem.assembler
                solverclass  = problem.solver
                success = defcon.newton.newton(F, z, bcs, problemclass, solverclass, 0)
                if not success:
                    print("Failed to converge from ", solution)
                    continue

                # Add it to known solutions
                knownroots.append(z.copy(deepcopy=True))

                # Compute the functionals
                funcs = []
                for functional in functionals:
                    func = functional[0]
                    j = func(z, params)
                    assert isinstance(j, float)
                    funcs.append(j)

                # Save and increment
                io.save_solution(z, funcs, [float(x) for x in params], solutionid)
                solutionid += 1

                print("Saved ", solution)
