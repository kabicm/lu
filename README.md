# LU Factorization with (MKL) ScaLAPACK

## Building and Installing

To build and run, do the following:

```bash
mkdir build && cd build
CC=gcc-9 CXX=g++-9 cmake .. # or whatever version of gcc compiler you have
make -j
mpiexec -np 5 ./lu <global matrix size> <block size> <num of repetitions>

# Example:
❯ mpirun -np 5 ./lu 1200 128 2
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
