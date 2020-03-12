Setup
=====

Requirements
------------

HeAT is based on PyTorch. Specifially, we are exploiting PyTorch's support for GPUs and MPI parallelism. Therefore, PyTorch must be compiled with MPI support when using HeAT. The instructions to install PyTorch in that way are contained in the script install-torch.sh, which we're also using to install PyTorch in Travis CI.

Installation
------------

Tagged releases of HeAT are made available on the Python Package Index
(PyPI). You can typically install the latest version with::

  $ pip install heat

Optionally, you can enable and install HDF5 and/or NetCDF support by adding the
respective extra requirements::

  $ pip install 'heat[hdf5, netcdf]'

If you want to work with the development version, you can checkout the sources using::

  $ git clone https://github.com/helmholtz-analytics/heat.git

This version can then be installed via::

  $ cd heat
  $ pip install '.[hdf5, netcdf]'
