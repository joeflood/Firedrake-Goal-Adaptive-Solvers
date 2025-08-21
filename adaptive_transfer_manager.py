from enum import IntEnum
from firedrake import *
from firedrake.mg.embedded import TransferManager
from firedrake.mg.utils import get_level

import time

from mpi4py import MPI


__all__ = ("AdaptiveTransferManager", )


native_families = frozenset(["Lagrange", "Discontinuous Lagrange", "Real", "Q", "DQ", "BrokenElement"])
alfeld_families = frozenset(["Hsieh-Clough-Tocher", "Reduced-Hsieh-Clough-Tocher", "Johnson-Mercier",
                             "Alfeld-Sorokina", "Arnold-Qin", "Reduced-Arnold-Qin", "Christiansen-Hu",
                             "Guzman-Neilan", "Guzman-Neilan Bubble"])
non_native_variants = frozenset(["integral", "fdm", "alfeld"])



class Op(IntEnum):
    PROLONG = 0
    RESTRICT = 1
    INJECT = 2


class AdaptiveTransferManager(TransferManager):
    def __init__(self, *, native_transfers=None, use_averaging=True):  
        super().__init__(native_transfers=native_transfers, use_averaging=use_averaging)
        self.tm = TransferManager()
        self.weight_cache = {}

    def generic_transfer(self, source, target, transfer_op):
        # determine which meshes to iterate over
        amh, source_level = get_level(source.function_space().mesh())
        _, target_level = get_level(target.function_space().mesh())
            
        # decide order of iteration depending on coarse -> fine or fine -> coarse
        order = 1
        if target_level < source_level: order = -1

        curr_source = source
        if source_level == target_level: 
            target.assign(source)
            return

        for level in range(source_level, target_level, order):
            if level  + order == target_level:
                curr_target = target
            else:
                target_mesh = amh.meshes[level + order]
                curr_space = curr_source.function_space()
                target_space = curr_space.reconstruct(mesh=target_mesh)
                if isinstance(curr_source, Function):
                    curr_target = Function(target_space)
                if isinstance(curr_source, Cofunction):
                    curr_target = Cofunction(target_space)

            if transfer_op == self.tm.restrict:
                if (level, level + order) not in self.weight_cache:
                    w = amh.use_weight(curr_source, child=True)
                    self.weight_cache[(source_level, target_level)] = Function(curr_source.function_space()).assign(w)
                else:
                    w = self.weight_cache[(source_level, target_level)]

                wsource = Function(curr_source.function_space())
                with curr_source.dat.vec as svec, w.dat.vec as wvec, wsource.dat.vec as wsvec:
                    wsvec.pointwiseDivide(svec, wvec)
                curr_source = wsource

            if order == 1:
                source_function_splits = amh.split_function(curr_source, child=False)
                target_function_splits = amh.split_function(curr_target, child=True)
            else:
                source_function_splits = amh.split_function(curr_source, child=True)
                target_function_splits = amh.split_function(curr_target, child=False)

            for split_label, _ in source_function_splits.items():
                transfer_op(source_function_splits[split_label], target_function_splits[split_label]) 
                
            amh.recombine(target_function_splits, curr_target, child=order+1)
            curr_source = curr_target

    def prolong(self, uc, uf):
        self.generic_transfer(uc, uf, transfer_op=self.tm.prolong)

    def inject(self, uf, uc):
        self.generic_transfer(uf, uc, transfer_op=self.tm.inject)

    def restrict(self, source, target):
        self.generic_transfer(source, target, transfer_op=self.tm.restrict)
