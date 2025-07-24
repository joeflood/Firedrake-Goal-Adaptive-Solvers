from firedrake import *
from netgen.occ import *
from dataclasses import dataclass, field
from functools import cached_property

class SolverCtx:
    def __init__(self, config: dict):
        self.degree = config["degree"]
        self.dual_solve_method = config["dual_solve_method"]
        self.residual_solve_method = config["residual_solve_method"]
        self.dorfler_alpha = config["dorfler_alpha"]
        self.tolerance = config["goal_tolerance"]
        self.max_iterations = config["max_iterations"]
        self.output_dir = config["output_dir"]

        context = {"degree": self.degree}
        self.dual_solve_degree = eval(config.get("dual_solve_degree", "degree + 1"), {}, context)
        self.residual_degree = eval(config.get("residual_degree", "degree"), {}, context)

    # Solver parameters
    sp_chol = {"pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps"}
    sp_pmg = {"snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-10,
            "ksp_max_it": 1,
            "ksp_convergence_test": "skip",
            "ksp_monitor": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.P1PC",
            "pmg_mg_coarse": {
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled_pc_type": "cholesky",
            },
            "pmg_mg_levels": {
                "ksp_max_it": 10,
                "ksp_type": "chebyshev",
                "pc_type": "python",
                "pc_python_type": "firedrake.ASMStarPC",
                "pc_star_mat_ordering_type": "metisnd",
                "pc_star_sub_sub_pc_type": "cholesky",
            }
        }
    sp_star = {"snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_rtol": 1.0e-10,
            "ksp_max_it": 5,
            "ksp_convergence_test": "skip",
            "ksp_monitor": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMStarPC",
            "pc_star_mat_ordering_type": "metisnd",
            "pc_star_sub_sub_pc_type": "cholesky",
            }
    residual_sp = {"snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "hypre",
            "pc_hypre_type": "pilut"}