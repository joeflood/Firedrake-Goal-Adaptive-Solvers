from netgen.occ import *

def getlabels(mesh):
    ngmesh = mesh.netgen_mesh
    names = ngmesh.GetRegionNames(codim=2)
    print(names)
    names_to_labels = {}
    for l in names:
        names_to_labels[l] = tuple(i+1 for i, name in enumerate(names) if name == l)
        print(names_to_labels[l])
    return names_to_labels