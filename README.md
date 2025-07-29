Goal based adaptive meshing for dissertation.

If it does not run, include this prior to running:
export PYTHONPATH=/home/joefl/projects:$PYTHONPATH

FIREDRAKE EDITS:
1.
In venv-firedrake\lib\python3.10\site-packages\tsfc\driver.py, lines 100-101
Comment out the following lines:

#if integral_type.startswith("interior_facet") and diagonal and any(a.function_space().finat_element.is_dg() for a in arguments):
#raise NotImplementedError("Sorry, we can't assemble the diagonal of a form for interior facet integrals") # EDITED BY JF

2. 
In functionspace.py, TensorFunctionSpace:

if shape is None:
    shape = (mesh.geometric_dimension(),) * 2
