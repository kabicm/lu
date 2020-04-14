cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(lu VERSION 0.1 LANGUAGES CXX)

# Dependencies
#
find_package(MPI REQUIRED)

# MKL BLAS
find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_OMP SCALAPACK_32BIT_OMP)
set(BLAS_TARGET "mkl::blas_32bit_omp")
set(SCALAPACK_TARGET "mkl::scalapack_32bit_omp")

add_executable(lu "lu.cpp")
target_compile_features(lu PUBLIC cxx_std_11)
target_link_libraries(lu PUBLIC MPI::MPI_CXX
                         PRIVATE ${BLAS_TARGET}
                                 ${SCALAPACK_TARGET}
                     )
