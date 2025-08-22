from __future__ import absolute_import, print_function, division

from mpi4py import MPI

import gc
import math
import time
import sys
import os
import resource

from defcon.iomodule import remap_c_streams
from defcon.parallellayout import ranktoteamno
from petsc4py import PETSc


class DefconThread(object):
    """
    The base class for DefconWorker/DefconMaster.
    """
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.functionals = problem.functionals()

        self.deflation = kwargs.get("deflation", None)
        self.teamsize  = kwargs.get("teamsize", 1)
        self.verbose   = kwargs.get("verbose", True)
        self.debug     = kwargs.get("debug", False)
        self.logfiles  = kwargs.get("logfiles", False)
        self.continue_backwards = kwargs.get("continue_backwards", True)
        self.worldcomm = kwargs["comm"]

        self.configure_comms()
        self.configure_logs()

        # How many tasks to do before calling the garbage collector
        # Set to small (e.g. 1) if the problem size is very large
        # Set to large (e.g. 100) if the problem size is very small
        # Subclasses may override this value
        self.gc_frequency = kwargs.get("gc_frequency", 10)
        self.collect_call = 0 # counter for the garbage collector

    def log(self, msg, master=False, warning=False):
        if not self.verbose and warning is False: return
        if master is False and self.teamrank != 0: return

        if warning:
            fmt = RED = "\033[1;37;31m%s\033[0m"
        else:
            if master:
                fmt = BLUE = "\033[1;37;34m%s\033[0m"
            else:
                fmt = GREEN = "\033[1;37;32m%s\033[0m"

        if master:
            header = "MASTER:   "
        else:
            header = "TEAM %3d: " % self.teamno

        timestamp = time.strftime("[%d-%m-%y @ %H:%M:%S] ")

        print(fmt % (timestamp + header + msg), flush=True)
        sys.stdout.flush()

    def collect(self):
        """
        Garbage collection.
        """
        self.collect_call += 1
        if self.collect_call % self.gc_frequency == 0:
            gc.collect()
            if hasattr(PETSc, "garbage_cleanup"):
                PETSc.garbage_cleanup(self.teamcomm)
                if hasattr(self, "mesh"):
                    PETSc.garbage_cleanup(self.mesh.mpi_comm())
                self.log("Called PETSc garbage collection.")
            else:
                self.log("Called normal garbage collection.")

    def configure_io(self, parameters):
        """
        parameters is a parametertools.Parameters object.
        """
        io = self.problem.io()
        # Don't construct the FunctionSpace on the master, it's a waste of memory
        if self.rank == 0:
            io.setup(parameters.parameters, self.functionals, None)
        else:
            io.setup(parameters.parameters, self.functionals, self.function_space)

        self.io = io

    def configure_comms(self):
        # Create a unique context, so as not to confuse my messages with other
        # libraries
        self.rank = self.worldcomm.rank

        # Assert even divisibility of team sizes
        if not (self.worldcomm.size-1) % self.teamsize == 0:
            DefconThread.log(self, msg="Need to use integer * teamsize + 1 processes", master=True, warning=True)
            raise ValueError("Incompatible nprocs = %d, teamsize = %d" % (self.worldcomm.size, self.teamsize))

        self.nteams = (self.worldcomm.size-1) // self.teamsize

        # Create local communicator for the team I will join
        self.teamno = ranktoteamno(self.rank, self.teamsize)
        self.teamcomm = self.worldcomm.Split(self.teamno, key=0)
        self.teamrank = self.teamcomm.rank

        # An MPI tag to indicate response messages
        self.responsetag = 121

        # We also need to create a communicator for rank 0 to talk to each
        # team (except for team 0, which it already has, as it is a member)
        if self.rank == 0:
            self.teamcomms = []
            for teamno in range(0, self.nteams):
                teamcommpluszero = self.worldcomm.Split(teamno, key=0)
                self.teamcomms.append(teamcommpluszero)
        else:
            for teamno in range(0, self.nteams):
                if teamno == self.teamno:
                    self.mastercomm = self.worldcomm.Split(self.teamno, key=0)
                else:
                    self.worldcomm.Split(MPI.UNDEFINED, key=0)

    def configure_logs(self):
        # If instructed, create logfiles for each team
        if self.logfiles:
            if self.rank == 0:
                stdout_filename = "defcon.log.master"
                stderr_filename = "defcon.err.master"
            else:
                # Zero pad filenames to ensure they line up.
                nzeros = int(math.floor(math.log(self.nteams, 10))) + 1
                pattern = "%%0%dd" % nzeros # e.g. "%05d" to make five zeros before the teamno
                if self.teamrank == 0:
                    stdout_filename = ("defcon.log." + pattern) % self.teamno
                    stderr_filename = ("defcon.err." + pattern) % self.teamno
                else:
                    stdout_filename = os.devnull
                    stderr_filename = os.devnull

            remap_c_streams(stdout_filename, stderr_filename)

