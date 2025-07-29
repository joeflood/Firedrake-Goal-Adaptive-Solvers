from firedrake import *
from netgen.occ import *
from dataclasses import dataclass
from functools import cached_property
from typing import Any

@dataclass(kw_only=True)
class ProblemCtx:
    V: FunctionSpace
    u: Function
    v: TestFunction
    u_exact: Any
    goal_exact: float
    F: Any
    bcs: list
    parameter: Constant
    J: Any
    f: Any = None
    g: Any = None
    bc_neumann: Any = None

    def __init__(self,
                 space=FunctionSpace,
                 trial=Function,
                 test=TestFunction,
                 exact=None,
                 residual=Any,
                 bcs=list,
                 goal=Any,
                 goal_exact=None,
                 parameter=None,
                 f=None,
                 g=None,
                 bc_neumann=None):
        self.V = space
        self.u = trial
        self.v = test
        self.u_exact = exact
        self.F = residual
        self.bcs = bcs or []
        self.J = goal
        self.goal_exact = goal_exact
        self.f = f
        self.g = g
        self.bc_neumann = bc_neumann
        self.parameter = parameter