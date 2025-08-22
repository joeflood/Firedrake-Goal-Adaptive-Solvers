from __future__ import absolute_import
import sys

def import_backend():
    """Import a backend module, tweak its parameters and
    return it; currently either dolfin or firedrake."""

    use_dolfin = True
    use_firedrake = False

    if "dolfin" in sys.modules and "firedrake" not in sys.modules:
        use_dolfin = True

    elif "firedrake" in sys.modules and "dolfin" not in sys.modules:
        use_dolfin = False
        use_firedrake = True

    elif "firedrake" in sys.modules and "dolfin" in sys.modules:
        # both loaded, don't know what to do
        raise ImportError("Import exactly one of dolfin or firedrake before importing defcon.")

    else: # nothing loaded, try importing
        try:
            import dolfin
            use_dolfin = True
            use_firedrake = False
        except ImportError:
            try:
                import firedrake
                use_dolfin = False
                use_firedrake = True
            except ImportError:
                raise ImportError("Must have one of dolfin or firedrake installed.")


    if use_dolfin:
        import dolfin
        assert dolfin.has_petsc4py()
        from petsc4py import PETSc


        try:
            dolfin.set_log_level(dolfin.ERROR)
        except AttributeError:
            dolfin.set_log_level(dolfin.LogLevel.ERROR)

        dolfin.parameters["form_compiler"]["representation"] = "uflacs"
        dolfin.parameters["form_compiler"]["optimize"]     = True
        dolfin.parameters["form_compiler"]["cpp_optimize"] = True
        dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

        # I have to *force* DOLFIN to initialise PETSc.
        # Otherwise, it will do it in the workers, using COMM_WORLD,
        # and deadlock. Yikes.
        dolfin.SubSystemsManager.init_petsc()

        # PETSc has recently implemented a new divergence tolerance,
        # which regularly breaks my deflation code. Disable it.
        dolfin.PETScOptions.set("snes_divergence_tolerance", -1)

        return (dolfin, PETSc)

    elif use_firedrake:
        # firedrake imported, no dolfin
        import firedrake

        firedrake.parameters["pyop2_options"]["lazy_evaluation"] = False

        from firedrake.petsc import PETSc
        opts = PETSc.Options()
        opts.setValue("snes_divergence_tolerance", -1)

        return (firedrake, PETSc)


(backend, PETSc) = import_backend()

# Monkey-patch modules so that user can import from a backend
sys.modules['defcon.backend'] = backend

# More code to deal with PETSc incompatibilities in API
if PETSc.Sys.getVersion()[0:2] <= (3, 7) and PETSc.Sys.getVersionInfo()['release']:

    def get_deep_submat(mat, isrow, iscol=None, submat=None):
        """Get deep submatrix of mat"""
        return mat.getSubMatrix(isrow, iscol, submat=submat)

    def get_shallow_submat(mat, isrow, iscol=None):
        """Get shallow submatrix of mat"""
        submat = PETSc.Mat().create(mat.comm)
        return submat.createSubMatrix(mat, isrow, iscol)

else:

    def get_deep_submat(mat, isrow, iscol=None, submat=None):
        """Get deep submatrix of mat"""
        return mat.createSubMatrix(isrow, iscol, submat=submat)

    def get_shallow_submat(mat, isrow, iscol=None):
        """Get shallow submatrix of mat"""
        submat = PETSc.Mat().create(mat.comm)
        return submat.createSubMatrixVirtual(mat, isrow, iscol)
