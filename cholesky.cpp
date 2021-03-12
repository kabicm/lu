// std dependencies
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// mpi
#include <mpi.h>

// mkl
#include <mkl.h>
#include <mkl_scalapack.h>
#include <mkl_cblas.h>

#include "utils.hpp"

/**
 * @brief parses the number of processors into a square grid
 * @param P the number of ranks that are available
 * @returns a pair of <PX, PY>
 */
std::pair<int, int> getProcessorGrid(int P)
{
    int sq_root = (int) std::floor(sqrt(P));
    return {sq_root, sq_root};
}

/**
 * @brief parses arguments from the command line
 * 
 * @param argc the number of arguments on the command line
 * @param argv the values of the arguments
 * @param N the matrix dimension
 * @param v the tile size (defaults to 128 if not provided)
 * @param rep the number of repetitions (defaults to 2 if not provided)
 * 
 * @returns true if the program was called correctly
 */
bool getParameters(int argc, char* argv[], int& N, int& v, int& rep)
{
    if (argc < 2) {
        std::cout << "Not enough arguments!" << std::endl;
        std::cout << "Call it like: " << std::endl;
        std::cout << "mpirun -np 4 ./cholesky <global matrix size> <tile size> <n_repetitions>" << std::endl;
        return false;
    }

    // global matrix size
    N = std::atoi(argv[1]);
    // block size (default = 128)
    v = 128;
    if (argc >= 3) {
        v = std::atoi(argv[2]);
    }
    // number of repetitions
    rep = 2;
    if (argc >= 4) {
        rep = std::atoi(argv[3]);
    }

    if (argc > 4) {
        std::cout << "Too many parameters given, ignoring some of them." << std::endl;
    }
    return true;
}

void printTimings(std::vector<double> &timings, int rank, int N, int v, int PX, int PY)
{
    if (rank == 0) {
            std::cout << "==========================" << std::endl;
            std::cout << "    PROBLEM PARAMETERS:" << std::endl;
            std::cout << "==========================" << std::endl;
            std::cout << "Matrix size: " << N << std::endl;
            std::cout << "Block size: " << v << std::endl;
            std::cout << "Processor grid: " << PX << " x " << PY << std::endl;
            std::cout << "Number of repetitions: " << timings.size() << std::endl;
            std::cout << "--------------------------" << std::endl;
            std::cout << "TIMINGS [ms] = ";
            for (auto &time : timings) {
                std::cout << time << " ";
            }
            std::cout << std::endl;
            std::cout << "==========================" << std::endl;
        }
}

/**
 * @brief runs ScaLAPACK's Cholesky factorization pdpotrf() for the specified
 * parameters N, v, and repetitions, and report the timings.
 * @param argc the number of arguments provided (should be 4)
 * @param argv the values of the arguments
 * 
 * @returns 0 on success, <0 otherwise
 */
int main(int argc, char* argv[])
{
    // initialize MPI environment first
    MPI_Init(&argc, &argv);

    // get input parameters from the command line
    int N, v, rep;
    if (!getParameters(argc, argv, N, v, rep)) {
        // wrong usage, terminate
        MPI_Finalize();
        return -1;
    }

    // obtain the processor grid (we only use square grids)
    int rank, P, PX, PY;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    std::tie(PX, PY) = getProcessorGrid(P);
    int usedP = PX * PY;
    if (rank == 0 && usedP < P) {
        std::cout << "Warning: only " << usedP << "/" << P << "processors ares used" << std::endl;
    }

    // only let processors with rank < usedP participate in the factorization
    if (rank < usedP) {
        // create new subcommunicator
        MPI_Comm world = subcommunicator(usedP, MPI_COMM_WORLD);

        // create a blacs context
        int ctx = Csys2blacs_handle(world);
        const char *order = "Row";
        Cblacs_gridinit(&ctx, order, PX, PY);

        // create the matrix descriptors
        int rsrc = 0; int csrc = 0; int info;
        int desca[9];
        int lld = scalapack::min_leading_dimension(N, v, PY); // local leading dim
        descinit_(desca, &N, &N, &v, &v, &rsrc, &csrc, &ctx, &lld, &info);

        // let root rank check for errors during creation, and handle them
        if (rank == 0 && info != 0) {
            std::cout << "Error in descinit, value of info: " << -info << std::endl;
            Cblacs_gridexit(ctx);
            Cblacs_exit(1);

            MPI_Comm_free(&world);
            MPI_Finalize();
            return -1;
        }

        // initialize matrix data with random values in interval [0, 1]
        int bufferSize = scalapack::local_buffer_size(desca);
        std::vector<double> matrix(bufferSize);
        int seed = 1005 + rank;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto &entry : matrix) {
            entry = dist(gen);
        }

        // we perform Cholesky on the full, lower triangular matrix
        int ia = 1; int ja = 1;
        char uplo = 'L';

        // TODO: strengthen the diagonal. Cholesky factorization requires s.p.d.
        // input matrices. however, our input matrix need not be symmetric because
        // ScaLAPACK implicitly assumes this, and thus never touches the upper
        // triangular part. Assuming symmetry, we can make the matrix s.p.d. by
        // strengthening the diagonal such that 
        //              \sum_{j=1, j!=i} A_{ij} < A_{ii} 
        // holds for all all i (i.e. for all diagonal elements). Since all of our
        // matrices entries are elements in [0, 1], setting elements on the diagonal
        // to e.g. 2N^2 (where N indicates the matrix dimension) will ensure this
        // property

        // create a vector for timings
        std::vector<double> timings;
        timings.reserve(rep);

        // perform the cholesky factorization rep times and time it
        for (size_t i = 0; i < rep; ++i) {
            // measure execution time in the current iteration
            MPI_Barrier(world);
            auto start = std::chrono::high_resolution_clock::now();
            pdpotrf_(&uplo, &N, matrix.data(), &ia, &ja, desca, &info);
            MPI_Barrier(world);
            auto end = std::chrono::high_resolution_clock::now();
            double timeInSec = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
            timeInSec *= 1e9;

            // check for errors that occured, and if so terminate
            if (rank == 0 && info != 0) {
                std::cout << "Error in dptorf, value of info: " << -info << std::endl;

                Cblacs_gridexit(ctx);
                Cblacs_exit(1);
                MPI_Comm_free(&world);
                MPI_Finalize();
                return -1;
            }
            timings.push_back(timeInSec);
        }

        // output the timing values
        printTimings(timings, rank, N, v, PX, PY);

        // finalize everything (except for MPI)
        Cblacs_gridexit(ctx);
        Cblacs_exit(1);
        MPI_Comm_free(&world);
    }

    // finalize MPI environment
    MPI_Finalize();
    return 0;
}