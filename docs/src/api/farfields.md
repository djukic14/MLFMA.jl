# Far fields

Leaf far-field patterns and the collections that store them.

A leaf's far field is its radiated (or received) pattern sampled on that leaf's own direction grid,
referenced to its own box centre. The representation a given operator needs is chosen by
`MLFMA.farfieldtype`; the storage a collection uses is a separate choice, and the compressed
harmonic storages have their own pages under Harmonics.

```@autodocs
Modules = [MLFMA.FarFields]
Filter = obj -> farfieldarea(obj) == :fields
```
