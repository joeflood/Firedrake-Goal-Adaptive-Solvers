from bratu import BratuProblem
from defcon import *
import matplotlib.pyplot as plt
import time

ac = ArclengthContinuation(problem=BratuProblem(), teamsize=1, verbose=True, logfiles=False)
ac.run(params=(0.5,), free="lambda", ds=0.1, sign=+1, bounds=(0.01, 3.6), branchids=[0])

ac.bifurcation_diagram("eval", "lambda", branchids=[0])
plt.title("The Bratu problem")
plt.savefig("arclength.pdf")
