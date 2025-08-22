# This postprocessing script takes in a directory and saves the
# solutions to PVD. Example:

# python makepvd.py output/delta=4.00000000000000008326672684688674053177237510681e-02
# paraview output/delta=4.00000000000000008326672684688674053177237510681e-02/solutions.pvd

import sys
import glob

# I want to call my file allen-cahn.py, damnit!
allencahn = __import__("allen-cahn")
globals().update(vars(allencahn))

problem = AllenCahnProblem()
mesh = problem.mesh(mpi_comm_world())
V    = problem.function_space(mesh)

dir = sys.argv[1]
pvd = File(dir + "/solutions.pvd")
for soln in sorted(glob.glob(dir + "/*xml.gz")):
    y = Function(V, soln, name="Solution")
    pvd << y

print("Created ", (dir + "/solutions.pvd"))
