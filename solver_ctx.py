from firedrake import *
from netgen.occ import *
from dataclasses import dataclass, field
from functools import cached_property

class SolverCtx:
    def __init__(self, config: dict):
        self.manual_indicators = config.get("manual_indicators", False) # Used for manual indicators (only implemented for Poisson but could be overriden)
        self.dorfler_alpha = config.get("dorfler_alpha", 0.5) # Dorfler marking parameter, default 0.5
        self.max_iterations = config.get("max_iterations", 10)
        self.output_dir = config.get("output_dir", "./output")
        self.dual_extra_degree = config.get("dual_extra_degree", 1)
        self.cell_residual_extra_degree = config.get("cell_residual_extra_degree", 0)
        self.facet_residual_extra_degree = config.get("facet_residual_extra_degree", 0)
        self.write_at_iteration = config.get("write_at_iteration", True)
        self.use_adjoint_residual = config.get("use_adjoint_residual", False) # For switching between primal and primal + adjoint residuals
        self.exact_indicators = config.get("exact_indicators", False) # Maybe remove
        self.uniform_refinement = config.get("uniform_refinement", False)
    # Solver parameters
    sp_cell   = {"mat_type": "matfree",
               "snes_type": "ksponly",
               "ksp_type": "cg",
               "pc_type": "jacobi",
               "pc_hypre_type": "pilut"}
    sp_facet    = {"mat_type": "matfree",
               "snes_type": "ksponly",
               "ksp_type": "cg",
               "pc_type": "jacobi",
               "pc_hypre_type": "pilut"}
    
    # EXAMPLE DUAL SOLVE METHODS
    sp_star = {"snes_type": "ksponly",
        "ksp_type": "cg",
        "ksp_rtol": 1.0e-10,
        "ksp_max_it": 20,
        "ksp_convergence_test": "skip",
        "ksp_monitor": None,
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star_mat_ordering_type": "metisnd",
        "pc_star_sub_sub_pc_type": "cholesky"
        }
    sp_vanka = {"snes_type": "ksponly",
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-10,
            "ksp_max_it": 20,
            "ksp_convergence_test": "skip",
            "ksp_monitor": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.ASMVankaPC",
            "pc_vanka_mat_ordering_type": "metisnd",
            "pc_vanka_sub_sub_pc_type": "cholesky",
            "pc_vanka_construct_dim": 0
            }
    sp_chol = {"pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps"}