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

bool get_parameters(int argc, char** argv, int& N, int& n_rep) {
    if (argc < 2) {
        std::cout << "Not enough arguments!" << std::endl;
        std::cout << "Call it like: " << std::endl;
        std::cout << "./local_lu <global matrix size> <n_repetitions>" << std::endl;
        return false;
    }

    // global matrix size
    N = std::atoi(argv[1]);
    // number of repetitions
    n_rep = 2;
    if (argc >= 3) {
        n_rep = std::atoi(argv[2]);
    }

    if (argc > 3) {
        std::cout << "Too many parameters given, ignoring some of them." << std::endl;
    }

    return true;
}

// prints the matrix in row-major format
template <typename T>
void print_matrix(T* mat, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << mat[i * N + j] << ", ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main(int argc, char ** argv)
{
    //******************************
    // INPUT PARAMETERS
    //******************************
    int N;  // global matrix size
    int n_rep; // number of repetitions

    // get the input parameters
    bool status = get_parameters(argc, argv, N, nb, n_rep);
    if (status != true) {
        return 0;
    }

    //******************************
    // GENERATING MATRIX DATA
    //******************************
    std::vector<double> a(N * N);
    int seed = 1234 + rank;
    std::mt19937 gen (seed);
    std::uniform_real_distribution<double> dist(0.0, 10.0);
    // set all elements to some random value
    for (auto& el : a) {
        el = dist(gen);
    }

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
        auto start = std::chrono::high_resolution_clock::now();

        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, a.data(), N, pivots.data());

        auto end = std::chrono::high_resolution_clock::now();

        // total time
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

        timings.push_back(time);
    }

    //******************************
    // OUTPUT TIMINGS
    //******************************
    std::cout << "==========================" << std::endl;
    std::cout << "    PROBLEM PARAMETERS:" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "Matrix size: " << N << std::endl;
    std::cout << "Number of repetitions: " << n_rep << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "Input Matrix: " << std::endl;
    print_matrix(a.data(), N);
    std::cout << "--------------------------" << std::endl;
    std::cout << "TIMINGS [ms] = ";
    for (auto &time : timings) {
        std::cout << time << " ";
    }
    std::cout << std::endl;
    std::cout << "==========================" << std::endl;

    return 0;
}
