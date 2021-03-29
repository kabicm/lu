'''
Date: 28.03.2021
Author: Jens Eirik Saethre (saethrej@ethz.ch)
Usage: run from SLATE-ROOT folder via e.g. 
    python3 scripts/analyze_scorep_output.py --library=slate --input=benchmarks/scorep/ --output=benchmarks/scorep/communication.csv
'''

import os
import subprocess
import argparse
import pandas as pd
import numpy as np 
from contextlib import contextmanager

input_path = 'benchmarks/scorep/'
output_path = 'benchmarks/scorep/communication.csv'

cmd = "cube_dump -t aggr -m bytes_sent profile.cubex | grep MPI_ | awk '{print $NF}' | paste -sd+ - | bc"

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def extract_comm_volume(cur_path: str) -> tuple:
    '''
    returns a tuple containing the following info:
    algo, N, P, grid, v, scaling, comm_vol
    '''
    # change to this directory to execute the extraction command
    with cd(cur_path):
        # extract the communication volume
        try:
            comm_vol = int(subprocess.check_output(cmd, shell=True))
        except Exception:
            raise Exception

        # extract the information from the directory name
        name = cur_path.split('/')[-1]
        params = name.split('_')

    return params[1], params[2], params[3], params[5], params[4], params[0], comm_vol

def main(library: str):
    # check whether the specified input directory exists
    if not os.path.isdir(input_path):
        print("invalid input directory, aborting.")
        return

    # list all directories contained in the input directory
    subdirs = [x[0] for x in os.walk(input_path) if x[0] != input_path]
    
    # open a file to write results to
    abs_path = os.path.join(os.getcwd(), output_path)
    with open(output_path, "w") as f:
        f.write("algorithm,library,N,N_base,P,grid,unit,type,value,blocksize\n")
        # loop over all directories, and extract the information
        for dir in subdirs:
            try:
                algo, N, P, grid, v, sca, vol = extract_comm_volume(dir)
                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                    algo, library, N, "-", P, grid, "bytes", sca, vol, v
                ))
            # in case that the directory name does not match the 
            except Exception:
                continue
    
    print("Successfully extracted all information. Exiting.")

def prepare():
    ''' loads the piz daint scripts and score-p '''
    os.system('source scripts/piz_daint_cpu.sh')
    os.system('module load Score-P')
    

if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Parse score-p folder for communication volume')
    parser.add_argument('--library', metavar='library', type=str, required=True,
                    help='name of the library that was profiled')
    parser.add_argument('--input', metavar='input', type=str, required=False,
                    help='path to the input folder (scorep)',  default='benchmarks/scorep')
    parser.add_argument('--output', metavar='output', type=str, required=False,
                    help='path to the output file', default='benchmarks/scorep/communication.csv')
    args = vars(parser.parse_args())

    if args['input'] is not None:
        input_path = args['input']
    if args['output'] is not None:
        output_path = args['output']

    # run only if an algo name was specified
    if args['library'] is not None:
        prepare()
        main(args['library'])

    else:
        print("Please enter the library name")