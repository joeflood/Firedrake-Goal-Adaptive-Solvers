from firedrake import *
from netgen.occ import *
from dataclasses import dataclass
from functools import cached_property

@dataclass
class MeshCtx:
    mesh: Mesh
    
    def __post_init__(self):
        self.boundary_labels()
        
    def boundary_labels(self):
        ngmesh = self.mesh.netgen_mesh
        names = ngmesh.GetRegionNames(codim=1)
        names_to_labels = {}
        for l in names:
            names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
        self.labels = names_to_labels

    @cached_property
    def dim(self):
        return self.mesh.topological_dimension()

    @cached_property
    def cell(self):
        return self.mesh.ufl_cell()

    # --- UFL geometry ---
    @cached_property
    def vol(self):
        return CellVolume(self.mesh)

    @cached_property
    def h(self):
        return CellDiameter(self.mesh)

    @cached_property
    def n(self):
        return FacetNormal(self.mesh)

    def update_mesh(self, new_mesh, new_tags=None):
        # 1) Mutate in place
        self.mesh = new_mesh
        if new_tags is not None:
            self.facet_tags = new_tags

        # 2) Clear any cached_property entries
        for name in ("dim", "cell", "vol", "h", "n"):
            self.__dict__.pop(name, None)

        # 3) Rebuild metadata
        self.boundary_labels()
    
    