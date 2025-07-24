from firedrake import *
from netgen.occ import *
from dataclasses import dataclass
from functools import cached_property
from typing import Any

@dataclass
class ProblemCtx:
    V: FunctionSpace
    u: Function
    v: TestFunction
    u_exact: Any
    F: Any
    bcs: list
    J: Any