# This files generates bash scripts that launch jobs on piz daint according to params.ini format.

import os
import configparser
import ast
import sys
import numpy as np
import csv
import struct
from numpy import genfromtxt
import configparser
import math
import argparse
from datetime import datetime

path_to_launch = './launch/'
path_to_params = './scripts/params.ini'
cholesky_section = 'cholesky'
lu_section = 'lu'
output_path = 'benchmarks'

def createBashPreface(P, algorithm):
    time = datetime.now().time()
    numNodes = math.ceil(P/2)
    return '#!/bin/bash -l \n\
#SBATCH --job-name=mkl-%s-p%d \n\
#SBATCH --time=02:00:00 \n\
#SBATCH --nodes=%d \n\
#SBATCH --output=%s/mkl-%s-p%d-%s.txt \n\
#SBATCH --constraint=mc \n\
#SBATCH --account=g34 \n\n\
export OMP_NUM_THREADS=18 \n\n' % (algorithm, P, numNodes, output_path, algorithm, P, time)

# parse params.ini
def readConfig(section):
    config = configparser.ConfigParser()
    config.read(path_to_params)
    if not config.has_section(section):
        print("Please add a %s section", (section))
        raise Exception()
    try:
        N = ast.literal_eval(config[section]['N'])
    except:
        print("Please add at least one matrix size N=[] (%s)" %(section))
        raise
    try:
        v = ast.literal_eval(config[section]['b'])
    except:
        print("Please add at least one tile size b=[] (%s)" %(section))
        raise
    
    grids = dict()
    try:
        read_grids = ast.literal_eval(config[section]['p_grids'])
    except:
        print("Please add at least one grid p_grids=[[P, grid]] (%s)" %(section))
        raise

    # we separate it here for later file separation
    for g in read_grids:
        P = g[0]
        grid = g[1:]
        grids[P] = grid

    try:
        reps = ast.literal_eval(config[cholesky_section]['r'])
    except:
        print("No number of repetitions found, using default 5. If you do not want this, add r= and the number of reps")
        reps = 5

    if len(N) ==0 or len(v) == 0 or len(grids) == 0:
        print("One of the arrays in params.ini is empty, please add values")
        raise Exception()
    
    return N, v, grids, reps
    

def generateLaunchFile(N, V, grids, reps, algorithm):
    for grid in grids:
        filename = path_to_launch + 'launch_%s_%d.sh' %(algorithm, grid)
        with open(filename, 'w') as f:
            numNodes = math.ceil(grid/2)
            f.write(createBashPreface(grid, algorithm))
            # next we iterate over all possibilities and write the bash script
            for rectangles in grids[grid]:
                for n in N:
                    for v in V:
                        cmd = 'srun -N %d -n %d ./build/%s -N %d -b %d --p_grid=%s -r %d \n' % (numNodes, grid, algorithm, n, v, rectangles, reps)
                        f.write(cmd)
    return

# We use the convention that we ALWAYS use n nodes and 2n ranks
# We might want to change that in future use
if __name__ == "__main__":

    # create a launch directory if it doesn't exist yet
    os.makedirs("launch", exist_ok=True)

    parser = argparse.ArgumentParser(description='Create sbatch files for launch on Piz Daint. \n \
    For every number of processors in params.ini, exactly one script is created \n \
    After creation, use the launch on daint script to launch all')
    parser.add_argument('--algo', metavar='algo', type=str, required=False,
                    help='lu for LU, chol for Cholesky, both for both',  default='both')
    parser.add_argument('--dir', metavar='dir', type=str, required=False,
                    help='path to the output file', default='benchmarks')
    args = vars(parser.parse_args())

    # parse the output directory path, and make the directories
    if args['dir'] is not None:
        output_path = args['dir']
        if output_path[-1] == '/':
            output_path = output_path[:-1]
        os.makedirs(output_path, exist_ok=True)

    # grids is a dict since for each processor size, we have to create a new launch file
    if args['algo'] == 'both':
        try:
            Ns, V, grids, reps = readConfig(cholesky_section)
            generateLaunchFile(Ns, V, grids, reps, cholesky_section)
            print("successfully generated launch files for cholesky")
        except:
            pass


        try:
            Ns, V, grids, reps = readConfig(lu_section)
            generateLaunchFile(Ns, V, grids, reps, lu_section)
            print("successfully generated launch files for lu")
        except:
            pass
    
    elif args['algo'] == 'chol':
        try:
            Ns, V, grids, reps = readConfig(cholesky_section)
            generateLaunchFile(Ns, V, grids, reps, cholesky_section)
            print("successfully generated launch files for cholesky")
        except:
            pass

    elif args['algo'] == 'lu':
        try:
            Ns, V, grids, reps = readConfig(lu_section)
            generateLaunchFile(Ns, V, grids, reps, lu_section)
            print("successfully generated launch files for lu")
        except:
            pass

    
    else:
        print("Please either use --algo=lu, --algo=chol for the algorithm"
              + " and specify the output directory with --dir=<path/to/output>")
