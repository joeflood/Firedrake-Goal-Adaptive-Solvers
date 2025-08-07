from firedrake import *
from netgen.occ import *
from algorithm import *
from functools import singledispatch
from firedrake.mg.ufl_utils import coarsen
from adaptive_mg.adaptive import AdaptiveMeshHierarchy
from adaptive_mg.adaptive_transfer_manager import AdaptiveTransferManager

class GoalAdaptiveNonlinearVariationalSolver():
    r"""Solves a :class:`NonlinearVariationalProblem`."""
    def __init__(self, problem: NonlinearVariationalProblem, solver_parameters: dict, goal_functional, tolerance: float):
        self.problem = problem
        self.solver_parameters = solver_parameters
        self.goal = goal_functional
        self.tolerance = tolerance

        self.V = problem.u.function_space()
        self.u = problem.u
        self.bcs = problem.bcs
        # We also need other things
        self.element = self.V.ufl_element()
        self.test = TestFunction(self.V)
        mesh = self.V.mesh()
        self.meshctx = MeshCtx(mesh)

    def reconstruct_problem(self):
        mesh
        new_mesh = self.meshctx.mesh
        amh = AdaptiveMeshHierarchy([mesh])
        atm = AdaptiveTransferManager()
        amh.add_mesh(new_mesh)
        coef_map = {}
        F_new = coarsen(F, coarsen, coefficient_mapping=coef_map)
        bcs_new = coarsen(bcs, coarsen, coefficient_mapping=coef_map)
        t_new = coarsen(t, coarsen, coefficient_mapping=coef_map)
        return F_new, bcs_new, t_new


    def solve(self):
        
