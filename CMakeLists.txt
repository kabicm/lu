cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(lu VERSION 0.1 LANGUAGES CXX)
include(cmake/adjust_mpiexec_flags.cmake)

# Dependencies
#
find_package(MPI REQUIRED)
adjust_mpiexec_flags()

# MKL BLAS
find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_OMP SCALAPACK_32BIT_OMP)
set(SCALAPACK_TARGET "mkl::scalapack_${MPI_TYPE}_intel_32bit_omp_dyn")

add_subdirectory(libs/costa)
add_subdirectory(libs/cxxopts)

set(executables "lu" "cholesky")
foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_compile_features(${exec} PUBLIC cxx_std_17)
    target_link_libraries(${exec} PUBLIC MPI::MPI_CXX
                             PRIVATE ${BLAS_TARGET}
                                     ${SCALAPACK_TARGET}
                                     costa
                                     cxxopts
                         )
    target_include_directories(${exec} PUBLIC "${lu_SOURCE_DIR}/libs/cxxopts/include/")
endforeach()
