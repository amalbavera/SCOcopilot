# SCOcopilot

[![Build Status][build-button]][build]
[![Coverage Status][codecov-button]][codecov]

[build-button]: https://github.com/Python-Markdown/markdown/actions/workflows/tox.yml/badge.svg
[build]: https://github.com/Python-Markdown/markdown/actions/workflows/tox.yml
[codecov-button]: https://codecov.io/gh/Python-Markdown/markdown/branch/master/graph/badge.svg
[codecov]: https://codecov.io/gh/Python-Markdown/markdown

Spin-Crossover copilot with Equivariant Graph Convolutional Neural Networks

This library is based on the work published in [J. Chem. Theory Comput.] **21** 3913–3921 (2025). Please ensure that the following external libraries are installed beforehand:

+ [pyTorch]
+ [NumPy]
+ [ASE]
+ [RDKit]
+ [PyG]
+ [Matplotlib]

[pyTorch]: https://pytorch.org/
[NumPy]: https://numpy.org/
[ASE]: https://wiki.fysik.dtu.dk/ase/
[RDKit]: https://www.rdkit.org/docs/GettingStartedInPython.html
[PyG]: https://pytorch-geometric.readthedocs.io/en/latest/
[Matplotlib]: https://matplotlib.org/
[J. Chem. Theory Comput.]: https://doi.org/10.1021/acs.jctc.4c01690

# Query

The library accepts SMILES strings as follows:

```python
from scocopilot import query

spin_crossover = query()

spin_crossover("[Co](NCCCC)(NCCCC)(SCCC)(SCCC)")
```

Alternatively, a file with Cartesian coordinates may also serve as input:

```python
from scocopilot import query

spin_crossover = query()

spin_crossover("path/to/file.xyz")
```

Default units are eV, but can be changed with the option `units="kJ/mol"` ot `units="kcal/mol"`. It also is possible to deactivate the `verbose` environment:

```python
from scocopilot import query

spin_crossover = query()

spin_gap, standard_deviation = spin_crossover("path/to/file.xyz", units="kJ/mol", verbose=False)
```

Changing the device from CPU (default) to GPU is done during the creation of the object:

```python
from scocopilot import query

spin_crossover = query(device="cpu")
```
