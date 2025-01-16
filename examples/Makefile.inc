# PYTHIA configuration file.
# Generated on Sat Feb 3 01:35:32 UTC 2024 with the user supplied options:
# --prefix=/opt/homebrew/Cellar/pythia/8.310
# --with-hepmc3=/opt/homebrew/opt/hepmc3
# --with-lhapdf6=/opt/homebrew/opt/lhapdf

# Install directory prefixes.
PREFIX_BIN=/opt/homebrew/Cellar/pythia/8.310/bin
PREFIX_INCLUDE=/opt/homebrew/Cellar/pythia/8.310/include
PREFIX_LIB=/opt/homebrew/Cellar/pythia/8.310/lib
PREFIX_SHARE=/opt/homebrew/Cellar/pythia/8.310/share/Pythia8

# Compilation flags (see ./configure --help for further documentation).
CXX=clang++
CXX_COMMON=-O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread
CXX_SHARED=-dynamiclib
CXX_SONAME=-Wl,-dylib_install_name,@rpath/
LIB_SUFFIX=.dylib
OBJ_COMMON=

EVTGEN_USE=false
EVTGEN_CONFIG=
EVTGEN_BIN=
EVTGEN_INCLUDE=
EVTGEN_LIB=

FASTJET3_USE=false
FASTJET3_CONFIG=
FASTJET3_BIN=
FASTJET3_INCLUDE=
FASTJET3_LIB=

HEPMC2_USE=false
HEPMC2_CONFIG=
HEPMC2_BIN=
HEPMC2_INCLUDE=
HEPMC2_LIB=

HEPMC3_USE=true
HEPMC3_CONFIG=HepMC3-config
HEPMC3_BIN=/opt/homebrew/Cellar/hepmc3/3.2.7/bin/
HEPMC3_INCLUDE=-I/opt/homebrew/Cellar/hepmc3/3.2.7/include
HEPMC3_LIB=-L/opt/homebrew/Cellar/hepmc3/3.2.7/lib -Wl,-rpath,/opt/homebrew/Cellar/hepmc3/3.2.7/lib -lHepMC3

LHAPDF5_USE=false
LHAPDF5_CONFIG=
LHAPDF5_BIN=
LHAPDF5_INCLUDE=
LHAPDF5_LIB=

LHAPDF6_USE=true
LHAPDF6_CONFIG=lhapdf-config
LHAPDF6_BIN=/opt/homebrew/Cellar/lhapdf/6.5.4/bin/
LHAPDF6_INCLUDE=-I/opt/homebrew/Cellar/lhapdf/6.5.4/include
LHAPDF6_LIB=-L/opt/homebrew/Cellar/lhapdf/6.5.4/lib -Wl,-rpath,/opt/homebrew/Cellar/lhapdf/6.5.4/lib -lLHAPDF

POWHEG_USE=false
POWHEG_CONFIG=
POWHEG_BIN=
POWHEG_INCLUDE=
POWHEG_LIB=

RIVET_USE=false
RIVET_CONFIG=
RIVET_BIN=
RIVET_INCLUDE=
RIVET_LIB=

ROOT_USE=false
ROOT_CONFIG=
ROOT_BIN=
ROOT_INCLUDE=
ROOT_LIB=

GZIP_USE=false
GZIP_CONFIG=
GZIP_BIN=
GZIP_INCLUDE=
GZIP_LIB=

PYTHON_USE=false
PYTHON_CONFIG=
PYTHON_BIN=
PYTHON_INCLUDE=
PYTHON_LIB=

MG5MES_USE=false
MG5MES_CONFIG=
MG5MES_BIN=
MG5MES_INCLUDE=
MG5MES_LIB=

OPENMP_USE=false
OPENMP_CONFIG=
OPENMP_BIN=
OPENMP_INCLUDE=
OPENMP_LIB=

MPICH_USE=false
MPICH_CONFIG=
MPICH_BIN=
MPICH_INCLUDE=
MPICH_LIB=

HDF5_USE=false
HDF5_CONFIG=
HDF5_BIN=
HDF5_INCLUDE=
HDF5_LIB=

HIGHFIVE_USE=false
HIGHFIVE_CONFIG=
HIGHFIVE_BIN=
HIGHFIVE_INCLUDE=
HIGHFIVE_LIB=
