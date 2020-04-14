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

// decomposes P into a 2D grid
std::pair<int, int> processor_grid(int P) {
    int sq_root = (int) std::floor(sqrt(P));
    return {sq_root, sq_root};
}

bool get_parameters(int argc, char** argv, int& N, int& nb, int& n_rep) {
    if (argc < 2) {
        std::cout << "Not enough arguments!" << std::endl;
        std::cout << "Call it like: " << std::endl;
        std::cout << "mpirun -np 4 ./lu <global matrix size> <block size> <n_repetitions>" << std::endl;
        return false;
    }

    // global matrix size
    N = std::atoi(argv[1]);
    // block size (default = 128)
    nb = 128;
    if (argc >= 3) {
        nb = std::atoi(argv[2]);
    }
    // number of repetitions
    n_rep = 2;
    if (argc >= 4) {
        n_rep = std::atoi(argv[3]);
    }

    if (argc > 4) {
        std::cout << "Too many parameters given, ignoring some of them." << std::endl;
    }

    return true;
}

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    //******************************
    // INPUT PARAMETERS
    //******************************
    int N;  // global matrix size
    int nb; // block size
    int n_rep; // number of repetitions

    // get the input parameters
    bool status = get_parameters(argc, argv, N, nb, n_rep);
    if (status != true) {
        MPI_Finalize();
        return 0;
    }

    //******************************
    // COMMUNICATOR
    //******************************
    int rank;
    int P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    //******************************
    // GRID DECOMPOSITION
    //******************************
    int prows, pcols;
    // get a 2D processor decomposition
    std::tie(prows, pcols) = processor_grid(P);
    // if P is not a perfect square then P_used != P 
    // (in fact, it always holds: P_used <= P)
    int P_used = prows * pcols;

    if (P_used < P && rank == 0) {
        std::cout << "Warning: using only " << P_used << " out of " << P << " processes." << std::endl;
    }

    if (rank < P_used) {
        // create a subcommunicator of MPI_COMM_WORLD taking first P_used ranks
        // this is only relevant if P is not a perfect square
        MPI_Comm comm = subcommunicator(P_used, MPI_COMM_WORLD);

        //******************************
        // BLACS CONTEXT CREATION
        //******************************
        // get a grid descriptor corresponding to this communicator
        int ctx = Csys2blacs_handle(comm);
        // we use by default row-major ordering of processors
        std::string order = "Row";
        // initialize a blacs context from the given grid descriptor (ctx)
        Cblacs_gridinit(&ctx, order.c_str(), prows, pcols);

        //******************************
        // MATRIX DESCRIPTORS
        //******************************
        int desca[9];
        int rsrc = 0; 
        int csrc = 0;
        int info;
        // local leading dimension
        int lld = scalapack::max_leading_dimension(N, nb, pcols);

        // create matrix descriptor
        descinit_(desca, &N, &N, &nb, &nb, &rsrc, &csrc, &ctx, &lld, &info);

        // check for the error during creation
        if (rank == 0 && info != 0) {
            std::cout << "error: descinit, argument: " << -info << " has an illegal value!" << std::endl;

            //******************************
            // ERROR: finalize everything
            //******************************
            Cblacs_gridexit(ctx);
            int dont_finalize_mpi = 1;
            Cblacs_exit(dont_finalize_mpi);

            MPI_Comm_free(&comm);
            MPI_Finalize();
            return 0;
        }

        //******************************
        // GENERATING MATRIX DATA
        //******************************
        int buffer_size = scalapack::local_buffer_size(desca);
        std::vector<double> a(buffer_size);
        int seed = 1234 + rank;
        std::mt19937 gen (seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        // set all elements to some random value
        for (auto& el : a) {
            el = dist(gen);
        }

        // we want LU of the full matrix, and not of a submatrix
        int ia = 1; 
        int ja = 1;

        //******************************
        // PIVOTING
        //******************************
        std::vector<int> pivots;
        pivots.reserve(N);
        // it's 1-based
        for (int i = 0; i < N; ++i) {
            pivots.push_back(i+1);
        }

        std::vector<long> timings;
        timings.reserve(n_rep);

        //******************************
        // LU FACTORIZATION + TIMING
        //******************************
        for (int i = 0; i < n_rep; ++i) {
            MPI_Barrier(comm);
            auto start = std::chrono::high_resolution_clock::now();

            pdgetrf_(&N, &N, a.data(), &ia, &ja, desca, pivots.data(), &info);

            MPI_Barrier(comm);
            auto end = std::chrono::high_resolution_clock::now();

            // total time
            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

            if(rank == 0 && info != 0) {
                std::cout << "error: pdgetrf, argument: " << -info << " has an illegal value!" << std::endl;

                //******************************
                // ERROR: finalize everything
                //******************************
                // finalize everything
                Cblacs_gridexit(ctx);
                int dont_finalize_mpi = 1;
                Cblacs_exit(dont_finalize_mpi);

                MPI_Comm_free(&comm);
                MPI_Finalize();
                return 0;
            }

            timings.push_back(time);
        }

        if (rank == 0) {
            //******************************
            // OUTPUT TIMINGS
            //******************************
            if (rank == 0) {
                std::cout << "==========================" << std::endl;
                std::cout << "    PROBLEM PARAMETERS:" << std::endl;
                std::cout << "==========================" << std::endl;
                std::cout << "Matrix size: " << N << std::endl;
                std::cout << "Block size: " << nb << std::endl;
                std::cout << "Processor grid: " << prows << " x " << pcols << std::endl;
                std::cout << "Number of repetitions: " << n_rep << std::endl;
                std::cout << "--------------------------" << std::endl;
                std::cout << "TIMINGS [ms] = ";
                for (auto &time : timings) {
                    std::cout << time << " ";
                }
                std::cout << std::endl;
                std::cout << "==========================" << std::endl;
            }
        }

        //******************************
        // FINALIZING EVERYTHING
        //******************************
        Cblacs_gridexit(ctx);
        int dont_finalize_mpi = 1;
        Cblacs_exit(dont_finalize_mpi);

        MPI_Comm_free(&comm);
    }

    MPI_Finalize();
    return 0;
}
