# LU Factorization with (MKL) ScaLAPACK

## Building and Installing

Make sure the repo is cloned with the `--recursive` flag, e.g.
```
git clone --recursive https://github.com/kabicm/lu
```

To build and run, do the following:

```bash
mkdir build && cd build
CC=gcc-9 CXX=g++-9 cmake .. # or whatever version of gcc compiler you have
make -j
mpiexec -np <num MPI ranks> ./lu -N <global matrix size> -b <block size> --p_grid=<prow>,<pcol> -r <num of repetitions>

# Example:
mpirun -n 4 ./lu -N 1200 -b 128 --p_grid=2,2 -r 2
Warning: using only 4 out of 5 processes.
==========================
    PROBLEM PARAMETERS:
==========================
Matrix size: 1200
Block size: 128
Processor grid: 2 x 2
Number of repetitions: 2
--------------------------
TIMINGS [ms] = 353 186
==========================
```

## Authors:
- Marko Kabic (marko.kabic@cscs.ch)
- Tal Ben Nun (talbn@inf.ethz.ch)
