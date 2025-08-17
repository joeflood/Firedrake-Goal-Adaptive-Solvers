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
        self.max_iterations = config["max_iterations"]
        self.output_dir = config["output_dir"]

        context = {"degree": self.degree}
        self.dual_solve_degree = eval(config.get("dual_solve_degree", "degree + 1"), {}, context)
        self.residual_degree = eval(config.get("residual_degree", "degree"), {}, context)

        # (Optional) Parameters, Required for GoalAdaptionStabilized 
        self.parameter_init = config.get("parameter_init", 1)
        self.parameter_final = config.get("parameter_final", 1)
        self.parameter_iterations = config.get("parameter_iterations", 1)
        self.write_at_iteration = config.get("write_at_iteration", False)

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
    sp_residual = {"snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "hypre",
            "pc_hypre_type": "pilut"} # Maybe now defunct
    sp_cell     = {"mat_type": "matfree",
               "snes_type": "ksponly",
               "ksp_type": "preonly",
               "pc_type": "python",
               "pc_python_type": "firedrake.PatchPC",
               "patch_pc_patch_save_operators": True,
               "patch_pc_patch_construct_type": "vanka",
               "patch_pc_patch_construct_codim": 0,
               "patch_pc_patch_sub_mat_type": "seqdense",
               "patch_sub_ksp_type": "preonly",
               "patch_sub_pc_type": "lu",
              }
    sp_cell2   = {"mat_type": "matfree",
               "snes_type": "ksponly",
               "ksp_type": "cg",
               "pc_type": "jacobi",
               "pc_hypre_type": "pilut"}
    sp_facet1    = {"mat_type": "matfree",
               "snes_type": "ksponly",
               "ksp_type": "cg",
               "pc_type": "jacobi",
               "pc_hypre_type": "pilut"}
    sp_facet2    = {"snes_type": "ksponly",
               "ksp_type": "preonly",
               "pc_type": "hypre",
               "pc_hypre_type": "pilut"}
    sp_etaT = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}