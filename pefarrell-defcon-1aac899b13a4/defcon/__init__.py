from __future__ import absolute_import

__version__ = "1.0"

import mpi4py.rc
if hasattr(mpi4py.rc, "threaded"):
    mpi4py.rc.threaded = False
elif hasattr(mpi4py.rc, "threads"):
    mpi4py.rc.threads = False
from mpi4py import MPI

try:
    import matplotlib
    __default_matplotlib_backend = matplotlib.get_backend()
    matplotlib.use('PDF')
except ImportError:
    pass

from numpy import arange, linspace

from defcon.backendimporter import backend
from defcon.bifurcationproblem import BifurcationProblem
from defcon.deflatedcontinuation import DeflatedContinuation
from defcon.arclength import ArclengthContinuation
from defcon.iomodule import IO, SolutionIO
from defcon.tasks import DeflationTask, ContinuationTask, StabilityTask, ArclengthTask, TangentPredictionTask
from defcon.operatordeflation import *
from defcon.variationalinequalities import ComplementarityProblem
from defcon.prediction import tangent, secant
from defcon.dwr import estimate_error_dwr

# This might fail because h5py is missing.
try:
    from defconf.branchio import BranchIO
except ImportError:
    pass

if backend.__name__ == "dolfin":
    from defcon.nonlinearsolver import SNUFLSolver

    # Lots of arbitrary interface changes
    try:
        backend.comm_world = backend.mpi_comm_world()
    except AttributeError:
        backend.comm_world = backend.MPI.comm_world

    def vec(x):
        if isinstance(x, backend.Function):
            x = x.vector()
        return backend.as_backend_type(x).vec()

    def mat(x):
        return backend.as_backend_type(x).mat()

elif backend.__name__ == "firedrake":
    backend.comm_world = MPI.COMM_WORLD

from defcon.backend import HDF5File
from defcon.dcsolve import dcsolve

# We have to disable the GC (this is a general thing with running DOLFIN in parallel).
# By default, each Python process decides completely by itself whether to do a
# garbage collection or not. Now suppose some object (e.g. an LU factorisation
# computed by MUMPS) is shared between two processes in a team. It can occur that
# one member of the team decides to clean up, and calls a collective operation,
# deadlocking the entire team. This is not good.
#
# Thus, we disable the GC here and put explicit calls to gc.collect in various places
# in defcon, for both the master and workers.
import gc
gc.disable()
