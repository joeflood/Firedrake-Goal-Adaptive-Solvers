from __future__ import absolute_import, print_function

"""
A module that implements the I/O backend for deflated continuation.

FIXME: I've tried to write this in a modular way so that it is possible to
implement more efficient/scalable backends at a later time.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import json
import tempfile
import shutil
import sys
import os
import glob
import time
from ast import literal_eval
import atexit
import collections

from defcon.parametertools import parameters_to_string
import defcon.backend as backend
from defcon.backend import HDF5File, Function, File


class IO(object):
    """
    Base class for I/O implementations.
    """

    def __init__(self, directory, comm):
        self.directory = directory
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            pass

        tmpdir = os.path.abspath("tmp")
        self.made_tmp = False
        try:
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)
                self.made_tmp = True
        except OSError:
            pass
        self.tmpdir = tmpdir
        self.worldcomm = comm

        atexit.register(self.cleanup)

    def cleanup(self):
        if self.made_tmp:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def log(self, msg, warning=False):
        if hasattr(self, "pcomm"):
            if self.pcomm.rank != 0:
                return

        if warning:
            fmt = RED = "\033[1;37;31m%s\033[0m\n"
        else:
            fmt = GREEN = "\033[1;37;32m%s\033[0m\n"

        timestamp = "[%s] " % time.strftime("%H:%M:%S")

        if warning:
            sys.stderr.write(fmt % (timestamp + msg))
            sys.stderr.flush()
        else:
            sys.stdout.write(fmt % (timestamp + msg))
            sys.stdout.flush()

    def parameter_map(self):
        """
        Return an object that stores the map of parameter tuple -> list of known branches.
        """
        return collections.defaultdict(list)

    def close_parameter_map(self):
        pass

    def construct(self, worldcomm):
        self.worldcomm = worldcomm

    def setup(self, parameters, functionals, function_space):
        self.parameters = parameters
        self.functionals = functionals
        self.function_space = function_space

        # Argh, why do we need two communicators, from two libraries,
        # written by the same person ... ?
        if function_space is not None:
            # petsc4py comm
            self.pcomm = PETSc.Comm(function_space.mesh().mpi_comm())
            self.mcomm = self.pcomm.tompi4py()
        else:
            self.mcomm = MPI.COMM_SELF
            self.pcomm = PETSc.Comm(self.mcomm)

    def clear(self):
        if os.path.exists(self.directory):
            tmpd = tempfile.mkdtemp(dir=os.getcwd())
            shutil.move(self.directory, tmpd + os.path.sep + self.directory)
            shutil.rmtree(tmpd, ignore_errors=True)

    def save_solution(self, solution, funcs, params, branchid, save_dir=None):
        raise NotImplementedError

    def fetch_solutions(self, params, branchids, fetch_dir=None):
        raise NotImplementedError

    def fetch_functionals(self, params, branchids):
        raise NotImplementedError

    def known_branches(self, params):
        raise NotImplementedError

    def known_parameters(self, fixed, branchid, stability=False):
        raise NotImplementedError

    def max_branch(self):
        raise NotImplementedError

    def save_arclength(self, params, branchid, ds, data):
        raise NotImplementedError

    def save_stability(self, stable, eigenvalues, eigenfunctions, params, branchid):
        raise NotImplementedError

    def fetch_stability(self, params, branchids):
        raise NotImplementedError

if backend.__name__ == "firedrake":
    from defcon.backend import CheckpointFile
    class SolutionIO(IO):
        """An I/O class that saves one CheckpointFile per solution found."""
        def dir(self, params):
            return self.directory + os.path.sep + parameters_to_string(self.parameters, params) + os.path.sep

        def save_solution(self, solution, funcs, params, branchid, save_dir=None):

            if save_dir is None:
                save_dir = self.dir(params)
            else:
                save_dir = save_dir + os.path.sep

            try:
                os.makedirs(save_dir)
            except OSError:
                pass

            tmpname = os.path.join(self.tmpdir, 'solution-%d.h5' % branchid)
            with CheckpointFile(tmpname, 'w', comm=self.function_space.mesh().mpi_comm()) as f:
                f.save_mesh(solution.function_space().mesh())
                f.save_function(solution, name="solution")
            if self.pcomm.rank == 0:
                os.rename(tmpname, save_dir + "solution-%d.h5" % branchid)

                with open(save_dir + "functional-%d.txt" % branchid, "w") as f:
                    s = parameters_to_string(self.functionals, funcs).replace('@', '\n') + '\n'
                    f.write(s)

            self.pcomm.Barrier()
            assert os.path.exists(save_dir + "solution-%d.h5" % branchid)

        def fetch_solutions(self, params, branchids, fetch_dir=None):
            if fetch_dir is None:
                fetch_dir = self.dir(params)
            else:
                fetch_dir = fetch_dir + os.path.sep

            solns = []
            for branchid in branchids:
                filename = fetch_dir + "solution-%d.h5" % branchid
                failcount = 0
                while True:
                    try:
                        with CheckpointFile(filename, 'r', comm=self.function_space.mesh().mpi_comm()) as f:
                            # What a chore this all is
                            saved_mesh = f.load_mesh()
                            soln_saved = f.load_function(saved_mesh, name="solution")
                            # Firedrake does not allow you to read data from one mesh to another, even if
                            # the mesh is created identically. To work around this limitation we change
                            # the set in the solution to be the set in the current mesh, which we are
                            # assuming to be the same. This is normally very dangerous, but since
                            # we know what we are doing* we do it anyway.
                            soln_saved.dat.dataset.set = self.function_space.dof_dset.set
                            soln = Function(self.function_space, val=soln_saved.dat)
                        break
                    except Exception:
                        print("Loading file %s failed. Sleeping for 10 seconds and trying again." % filename)
                        failcount += 1
                        if failcount == 10:
                            print("Tried 10 times to load file %s. Raising exception." % filename)
                            raise
                        time.sleep(10)

                solns.append(soln)
            return solns

        def fetch_functionals(self, params, branchid):
            funcs = []
            for param in params:
                with open(self.dir(param) + "functional-%d.txt" % branchid, "r") as f:
                    func = []
                    for line in f:
                        func.append(float(line.split('=')[-1]))
                funcs.append(func)
            return funcs

        def known_branches(self, params):
            filenames = sorted(glob.glob(self.dir(params) + "solution-*.h5"))
            branches = [int(filename.split('-')[-1][:-3]) for filename in filenames]
            return set(branches)

        def known_parameters(self, fixed, branchid, stability=False):
            fixed_indices = []
            fixed_values = []
            for key in fixed:
                fixed_values.append(fixed[key])
                # find the index
                for (i, param) in enumerate(self.parameters):
                    if param[1] == key:
                        fixed_indices.append(i)
                        break

            seen = set()
            if not stability:
                filenames = glob.glob(self.directory + "/*/solution-%d.h5" % branchid)
            else:
                filenames = glob.glob(self.directory + "/*/eigenfunctions-%d.h5" % branchid)
            saved_params = [tuple([float(x.split('=')[-1]) for x in filename.split('/')[-2].split('@')]) for filename in filenames]

            for param in saved_params:
                should_add = True
                for (index, value) in zip(fixed_indices, fixed_values):
                    if param[index] != float("%.15e" % value):
                        should_add = False
                        break

                if should_add:
                    seen.add(param)

            return sorted(list(seen))

        def max_branch(self):
            filenames = glob.glob(self.directory + "/*/solution-*.h5")
            branches = [int(filename.split('-')[-1][:-3]) for filename in filenames]
            return max(branches)

        def save_stability(self, stable, eigenvalues, eigenfunctions, params, branchid):
            assert len(eigenvalues) == len(eigenfunctions)

            if len(eigenvalues) > 0:
                filename = self.dir(params) + "eigenfunctions-%d.h5" % branchid
                if not os.path.exists(self.dir(params)):
                   os.makedirs(self.dir(params), exist_ok=True)
                with CheckpointFile(filename, 'w', comm=self.function_space.mesh().mpi_comm()) as f:
                    f.require_group("/defcon")
                    f.save_mesh(eigenfunctions[0].function_space().mesh())
                    for (i, (eigval, eigfun)) in enumerate(zip(eigenvalues, eigenfunctions)):
                        f.save_function(eigfun, name="eigenfunction-%d" % i)
                        f.set_attr("/defcon", "eigenvalue-%d" % i, eigval)

                    f.set_attr("/defcon", "number_eigenvalues", len(eigenvalues))

                # wait for the file to be written
                size = 0
                while True:
                    try:
                        size = os.stat(self.dir(params) + "eigenfunctions-%d.h5" % branchid).st_size
                    except OSError:
                        pass
                    if size > 0: break
                    #print("Waiting for %s to have nonzero size" % (self.dir(params) + "solution-%d.xml.gz" % branchid))
                    time.sleep(0.1)

            # Trick from Lawrence Mitchell: POSIX guarantees that mv is atomic
            f = tempfile.NamedTemporaryFile("w", delete=False, dir=self.tmpdir)
            s = str(stable)
            f.write(s)
            os.fsync(f.file.fileno())
            f.close()
            filename = self.dir(params) + "stability-%d.txt" % branchid
            if self.pcomm.rank == 0:
                os.rename(f.name, filename)

        def save_arclength(self, params, freeindex, branchid, ds, sign, data):
            dir_name = self.directory + os.path.sep + "arclength" + os.path.sep + "params-%s-freeindex-%s-branchid-%s-ds-%.14e-sign-%d" % (parameters_to_string(self.parameters, params), freeindex, branchid, ds, sign)

            try:
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
            except OSError:
                pass

            # Trick from Lawrence Mitchell: POSIX guarantees that mv is atomic
            f = tempfile.NamedTemporaryFile("w", delete=False, dir=self.tmpdir)
            json.dump(data, f.file, indent=4)
            f.file.flush()
            os.fsync(f.file.fileno())
            f.close()
            if self.pcomm.rank == 0:
                os.rename(f.name, dir_name + os.path.sep + "params-%s-freeindex-%s-branchid-%s-ds-%.14e-sign-%d.json" % (parameters_to_string(self.parameters, params), freeindex, branchid, ds, sign))

        def fetch_stability(self, params, branchids, fetch_eigenfunctions=False):
            stabs = []
            for branchid in branchids:
                stab = {}
                with open(self.dir(params) + "stability-%d.txt" % branchid, "r") as fs:
                    is_stable = literal_eval(fs.read())
                stab["stable"] =  is_stable

                filename = self.dir(params) + "eigenfunctions-%d.h5" % branchid
                if os.path.isfile(filename):
                    evals = []
                    eigfs = []

                    with CheckpointFile(filename, 'r', comm=self.function_space.mesh().mpi_comm()) as f:
                        # get Number of Eigenvalues
                        num_evals = f.get_attr("/defcon", "number_eigenvalues")
                        saved_mesh = f.load_mesh()

                        # Iterate through each eigenvalues and obtain corresponding eigenfunction
                        for i in range(num_evals):
                            eigval = f.get_attr("/defcon", "eigenvalue-%d" % i)
                            evals.append(eigval)

                            if fetch_eigenfunctions:
                                efunc_saved = f.load_function(saved_mesh, name="eigenfunction-%d" % i)
                                efunc = Function(self.function_space, val=efunc_saved.dat)
                                eigfs.append(efunc)

                    stab["eigenvalues"] = evals
                    stab["eigenfunctions"] = eigfs

                # else:
                # we couldn't find any eigenfunction checkpoint files,
                # but this is OK

                stabs.append(stab)
            return stabs
else:
    class SolutionIO(IO):
        """An I/O class that saves one HDF5File per solution found."""
        def dir(self, params):
            return self.directory + os.path.sep + parameters_to_string(self.parameters, params) + os.path.sep

        def save_solution(self, solution, funcs, params, branchid, save_dir=None):

            if save_dir is None:
                save_dir = self.dir(params)
            else:
                save_dir = save_dir + os.path.sep

            try:
                os.makedirs(save_dir)
            except OSError:
                pass

            tmpname = os.path.join(self.tmpdir, 'solution-%d.h5' % branchid)
            with HDF5File(comm=self.function_space.mesh().mpi_comm(), filename=tmpname, file_mode='w') as f:
                f.write(solution, "/solution")
            if self.pcomm.rank == 0:
                os.rename(tmpname, save_dir + "solution-%d.h5" % branchid)

                with open(save_dir + "functional-%d.txt" % branchid, "w") as f:
                    s = parameters_to_string(self.functionals, funcs).replace('@', '\n') + '\n'
                    f.write(s)

            self.pcomm.Barrier()
            assert os.path.exists(save_dir + "solution-%d.h5" % branchid)

        def fetch_solutions(self, params, branchids, fetch_dir=None):
            if fetch_dir is None:
                fetch_dir = self.dir(params)
            else:
                fetch_dir = fetch_dir + os.path.sep

            solns = []
            for branchid in branchids:
                filename = fetch_dir + "solution-%d.h5" % branchid
                failcount = 0
                while True:
                    try:
                        with HDF5File(comm=self.function_space.mesh().mpi_comm(), filename=filename, file_mode='r') as f:
                            soln = Function(self.function_space)
                            f.read(soln, "/solution")
                            f.flush()
                        break
                    except Exception:
                        print("Loading file %s failed. Sleeping for 10 seconds and trying again." % filename)
                        failcount += 1
                        if failcount == 10:
                            print("Tried 10 times to load file %s. Raising exception." % filename)
                            raise
                        time.sleep(10)

                solns.append(soln)
            return solns

        def fetch_functionals(self, params, branchid):
            funcs = []
            for param in params:
                with open(self.dir(param) + "functional-%d.txt" % branchid, "r") as f:
                    func = []
                    for line in f:
                        func.append(float(line.split('=')[-1]))
                funcs.append(func)
            return funcs

        def known_branches(self, params):
            filenames = sorted(glob.glob(self.dir(params) + "solution-*.h5"))
            branches = [int(filename.split('-')[-1][:-3]) for filename in filenames]
            return set(branches)

        def known_parameters(self, fixed, branchid, stability=False):
            fixed_indices = []
            fixed_values = []
            for key in fixed:
                fixed_values.append(fixed[key])
                # find the index
                for (i, param) in enumerate(self.parameters):
                    if param[1] == key:
                        fixed_indices.append(i)
                        break

            seen = set()
            if not stability:
                filenames = glob.glob(self.directory + "/*/solution-%d.h5" % branchid)
            else:
                filenames = glob.glob(self.directory + "/*/eigenfunctions-%d.h5" % branchid)
            saved_params = [tuple([float(x.split('=')[-1]) for x in filename.split('/')[-2].split('@')]) for filename in filenames]

            for param in saved_params:
                should_add = True
                for (index, value) in zip(fixed_indices, fixed_values):
                    if param[index] != float("%.15e" % value):
                        should_add = False
                        break

                if should_add:
                    seen.add(param)

            return sorted(list(seen))

        def max_branch(self):
            filenames = glob.glob(self.directory + "/*/solution-*.h5")
            branches = [int(filename.split('-')[-1][:-3]) for filename in filenames]
            return max(branches)

        def save_stability(self, stable, eigenvalues, eigenfunctions, params, branchid):
            assert len(eigenvalues) == len(eigenfunctions)

            if len(eigenvalues) > 0:
                with HDF5File(comm=self.function_space.mesh().mpi_comm(), filename=(self.dir(params) + "eigenfunctions-%d.h5" % branchid), file_mode='w') as f:
                    for (i, (eigval, eigfun)) in enumerate(zip(eigenvalues, eigenfunctions)):
                        f.write(eigfun, "/eigenfunction-%d" % i)
                        f.attributes("/eigenfunction-%d" % i)['eigenvalue'] = eigval

                    f.attributes('/eigenfunction-0')['number_eigenvalues'] = len(eigenvalues)

                # wait for the file to be written
                size = 0
                while True:
                    try:
                        size = os.stat(self.dir(params) + "eigenfunctions-%d.h5" % branchid).st_size
                    except OSError:
                        pass
                    if size > 0: break
                    #print("Waiting for %s to have nonzero size" % (self.dir(params) + "solution-%d.xml.gz" % branchid))
                    time.sleep(0.1)

            # Trick from Lawrence Mitchell: POSIX guarantees that mv is atomic
            f = tempfile.NamedTemporaryFile("w", delete=False, dir=self.tmpdir)
            s = str(stable)
            f.write(s)
            os.fsync(f.file.fileno())
            f.close()
            filename = self.dir(params) + "stability-%d.txt" % branchid
            if self.pcomm.rank == 0:
                os.rename(f.name, filename)

        def save_arclength(self, params, freeindex, branchid, ds, sign, data):
            dir_name = self.directory + os.path.sep + "arclength" + os.path.sep + "params-%s-freeindex-%s-branchid-%s-ds-%.14e-sign-%d" % (parameters_to_string(self.parameters, params), freeindex, branchid, ds, sign)

            try:
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
            except OSError:
                pass

            # Trick from Lawrence Mitchell: POSIX guarantees that mv is atomic
            f = tempfile.NamedTemporaryFile("w", delete=False, dir=self.tmpdir)
            json.dump(data, f.file, indent=4)
            f.file.flush()
            os.fsync(f.file.fileno())
            f.close()
            if self.pcomm.rank == 0:
                os.rename(f.name, dir_name + os.path.sep + "params-%s-freeindex-%s-branchid-%s-ds-%.14e-sign-%d.json" % (parameters_to_string(self.parameters, params), freeindex, branchid, ds, sign))

        def fetch_stability(self, params, branchids, fetch_eigenfunctions=False):
            stabs = []
            for branchid in branchids:
                stab = {}
                with open(self.dir(params) + "stability-%d.txt" % branchid, "r") as fs:
                    is_stable = literal_eval(fs.read())
                stab["stable"] =  is_stable

                try:
                    filename = self.dir(params) + "eigenfunctions-%d.h5" % branchid
                    evals = []
                    eigfs = []

                    with HDF5File(comm=self.function_space.mesh().mpi_comm(), filename=filename, file_mode='r') as f:
                        # get Number of Eigenvalues
                        num_evals= f.attributes('/eigenfunction-0')['number_eigenvalues']

                        # Iterate through each eigenvalues and obtain corresponding eigenfunction
                        for i in range(num_evals):
                            eigval = f.attributes("/eigenfunction-%d" % i)['eigenvalue']
                            evals.append(eigval)

                            if fetch_eigenfunctions:
                                efunc = Function(self.function_space)
                                f.read(efunc, "/eigenfunction-%d" % i)
                                f.flush()
                                eigfs.append(efunc)

                    stab["eigenvalues"] = evals
                    stab["eigenfunctions"] = eigfs
                except Exception:
                    # Couldn't find any eigenfunctions, OK
                    pass

                stabs.append(stab)
            return stabs

# Some code to remap C- and Python-level stdout/stderr
def remap_c_streams(stdout_filename, stderr_filename):
    import sys
    import os

    if os.path.isfile(stdout_filename):
        os.remove(stdout_filename)
    if os.path.isfile(stderr_filename):
        os.remove(stderr_filename)

    new_stdout = open(stdout_filename, "a+")
    new_stderr = open(stderr_filename, "a+")

    os.dup2(new_stdout.fileno(), sys.stdout.fileno())
    os.dup2(new_stderr.fileno(), sys.stderr.fileno())

    sys.stdout = new_stdout
    sys.stderr = new_stderr
