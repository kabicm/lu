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

// costa
#include <costa/layout.hpp>

// cxxopts
#include <cxxopts.hpp>

int main(int argc, char ** argv)
{
    // **************************************
    //   setup command-line parser
    // **************************************
    cxxopts::Options options("SCALAPACK LU MINIAPP", 
        "A miniapp computing: LU factorization of a random matrix.");

    options.add_options()
        ("N,dim",
            "matrix dimension", 
            cxxopts::value<int>()->default_value("10"))
        ("b,block",
            "block dimension.",
            cxxopts::value<int>()->default_value("2"))
        ("p,p_grid",
            "processor 2D-decomposition.",
             cxxopts::value<std::vector<int>>()->default_value("-1,-1"))
        ("r,n_rep",
            "number of repetitions.",
            cxxopts::value<int>()->default_value("2"))
        ("h,help", "Print usage.")
    ;
    const char** const_argv = (const char**) argv;
    auto result = options.parse(argc, const_argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    MPI_Init(&argc, &argv);

    //******************************
    // INPUT PARAMETERS
    //******************************
    auto N = result["dim"].as<int>();  // global matrix size
    auto nb = result["block"].as<int>(); // block size
    auto n_rep = result["n_rep"].as<int>(); // number of repetitions
    auto p_grid = result["p_grid"].as<std::vector<int>>(); // processor grid

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
    int prows = p_grid[0];
    int pcols = p_grid[1];

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
        char order = 'R';
        // initialize a blacs context from the given grid descriptor (ctx)
        Cblacs_gridinit(&ctx, &order, prows, pcols);

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

        // we want LU of the full matrix, and not of a submatrix
        int ia = 1; 
        int ja = 1;

        auto matrix = costa::block_cyclic_layout<double>(
            N, N,
            nb, nb,
            ia, ja,
            N, N,
            prows, pcols,
            order,
            rsrc, csrc,
            a.data(),
            lld,
            rank);

        // function f(i, j) := value of element (i, j) in the global matrix
        // an arbitrary function
        auto f = [&gen, &dist, N](int i, int j) -> double {
            auto value = dist(gen);
            if (i == j) value += 2*N*N;
            return value;
        };

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
            // reinitialize the matrix before each repetitions
            matrix.initialize(f);

            // start the timing
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
