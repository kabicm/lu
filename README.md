# LU & Cholesky Factorizations with (MKL) ScaLAPACK

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

# Example for LU:
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

# Example for Cholesky (output is structured in the same way):
mpirun -n 4 ./cholesky -N 1200 -b 128 --p_grid=2,2 -r 2
```

## Generating and Running the Scripts on Daint
Enter the params you want to work with into `scripts/params.ini`. Now, move to the source folder and generate the _.sh_ files by running `python3 scripts/generate_launch_files.py`. If you only want to generate scripts for lu, you can pass the argument `algo lu`. For only cholesky on the other hand, pass `algo chol`, If no argument is given, both are generated. After having generated the files, run `python3 scripts/launch_on_daint.py`. It will generate allocate nodes for each processor size, at the moment using the heuristic that we have n nodes with 2n ranks. If you launch very large jobs, perhaps you have to change the runtime in  `python3 scripts/generate_launch_files.py` or in the bash scripts directly (it defaults to 2 hours at the moment).
Each `.sh` file contains all jobs for one specific processor size. The outputs of jobs are written to the ./benchmarks folder (which you might have to create first).

## Authors:
- Marko Kabic (marko.kabic@cscs.ch)
- Tal Ben Nun (talbn@inf.ethz.ch)
- Jens Eirik Saethre (saethrej@ethz.ch)
- Andre Gaillard (andrega@ethz.ch)
