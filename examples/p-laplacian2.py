from firedrake import *
from netgen.occ import *
import sys
from goal_adaptivity import GoalAdaptiveNonlinearVariationalSolver

# --- two half-squares that touch at x = 0 ---
square = WorkPlane().MoveTo(-1, -1).Rectangle(2, 2).Face()
line = WorkPlane().MoveTo(0, -1).Line(1).Rotate(90)

# Keep coincident edges distinct (no fuse), so we can glue only what we want:
shape = square - line

# Tag only the interface edges; glue the top, leave the bottom open
for e in shape.edges:
    mp = e.center
    if abs(mp.x) < 1e-1:
        if mp.y < 0.1:        
            e.name = "crack"
    else:
        e.name = "exterior"

# Mesh and glue only the top interface
geo = OCCGeometry(shape, dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=0.2))

for f in shape.edges: # Assign face labels
    print(f.name)

VTKFile("p-laplcian_mesh.pvd").write(mesh)

# Define solver parameters ---------------------
solver_parameters = {
    "max_iterations": 30,
    "output_dir": "output",
    #"uniform_refinement": True
    #"use_adjoint_residual": True
}

#tolerance = 0.000001

#problem = NonlinearVariationalProblem(F, u, bcs)
#GoalAdaptiveNonlinearVariationalSolver(problem, J, tolerance, solver_parameters, u_exact).solve()


