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
 * @brief prints the timings to the desired stream
 * @param out the desired stream (e.g. std::cout or a file)
 * @param timings the vector containing the timings
 * @param N the matrix dimension
 * @param v the tile size
 * @param PX the number of processors in x-direction on the grid
 * @param PY the number of processors in y-direction on the grid
 */
void printTimings(std::vector<double> &timings, std::ostream &out, int N, int v, int PX, int PY)
{
    out << "==========================" << std::endl;
    out << "    PROBLEM PARAMETERS:" << std::endl;
    out << "==========================" << std::endl;
    out << "Matrix size: " << N << std::endl;
    out << "Block size: " << v << std::endl;
    out << "Processor grid: " << PX << " x " << PY << std::endl;
    out << "Number of repetitions: " << timings.size() << std::endl;
    out << "--------------------------" << std::endl;
    out << "TIMINGS [ms] = ";
    for (auto &time : timings) {
        out << time << " ";
    }
    out << std::endl;
    out << "==========================" << std::endl;
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
    // **************************************
    //   setup command-line parser
    // **************************************
    cxxopts::Options options("SCALAPACK CHOLESKY MINIAPP", 
        "A miniapp computing: Cholesky factorization of a random matrix.");

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
    // initialize MPI environment first
    MPI_Init(&argc, &argv);

    //******************************
    // INPUT PARAMETERS
    //******************************
    auto N = result["dim"].as<int>();  // global matrix size
    auto v = result["block"].as<int>(); // block size
    auto rep = result["n_rep"].as<int>(); // number of repetitions
    auto p_grid = result["p_grid"].as<std::vector<int>>(); // processor grid

    // obtain the processor grid (we only use square grids)
    int rank, P, PX, PY;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    PX = p_grid[0];
    PY = p_grid[1];

    // if not set, PX and PY will be -1 (see cxxopts default values above)
    if (PX < 0 || PY < 0) {
        // compute a reasonable decomposition
        std::tie(PX, PY) = getProcessorGrid(P);
    }

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
        char order = 'R';
        Cblacs_gridinit(&ctx, &order, PX, PY);

        // create the matrix descriptors
        int rsrc = 0; int csrc = 0; int info;
        int desca[9];
        int lld = scalapack::max_leading_dimension(N, v, PY); // local leading dim
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
        std::vector<double> mat(bufferSize);
        int seed = 1005 + rank;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // we perform Cholesky on the full, lower triangular matrix
        int ia = 1; int ja = 1;
        char uplo = 'L';

        // matrix descriptor for COSTA
        auto matrix = costa::block_cyclic_layout<double>(
            N, N,
            v, v,
            ia, ja,
            N, N,
            PX, PY,
            order,
            rsrc, csrc,
            mat.data(),
            lld,
            rank
        );

        // define lambda function to strengthen the diagonal
        auto f = [&gen, &dist, N](int i, int j) -> double {
            auto value = dist(gen);
            if (i == j) value += 2.0*N*N;
            return value;
        };

        // create a vector for timings
        std::vector<double> timings;
        timings.reserve(rep);

        // perform the cholesky factorization rep times and time it
        for (size_t i = 0; i < rep; ++i) {
            // reinitialize the matrix before each repetitions
            matrix.initialize(f);

            // measure execution time in the current iteration
            MPI_Barrier(world);
            auto start = std::chrono::high_resolution_clock::now();
            pdpotrf_(&uplo, &N, mat.data(), &ia, &ja, desca, &info);
            MPI_Barrier(world);
            auto end = std::chrono::high_resolution_clock::now();
            auto timeInMS = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

            // check for errors that occured, and if so terminate
            if (rank == 0 && info != 0) {
                std::cout << "Error in dptorf, value of info: " << -info << std::endl;

                Cblacs_gridexit(ctx);
                Cblacs_exit(1);
                MPI_Comm_free(&world);
                MPI_Finalize();
                return -1;
            }
            timings.push_back(timeInMS);
        }

        // let rank 0 output the timing values
        if (rank == 0) {
            printTimings(timings, std::cout, N, v, PX, PY);

            // if you want to print the results to a file, uncomment the following:
            //std::ofstream output("your-filename.txt", std::ios::out);
            //printTimings(timings, output, N, nb, prows, pcols);
            //output.close();
        }

        // finalize everything (except for MPI)
        Cblacs_gridexit(ctx);
        Cblacs_exit(1);
        MPI_Comm_free(&world);
    }

    // finalize MPI environment
    MPI_Finalize();
    return 0;
}
