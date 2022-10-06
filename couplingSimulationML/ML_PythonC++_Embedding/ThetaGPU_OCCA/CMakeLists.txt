cmake_minimum_required(VERSION 3.0)
project(burger VERSION 0.1)

enable_language(C)
enable_language(CXX)
enable_language(Fortran)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(FortranCInterface)

set(dfloat "double")
set(dfloatString "\"double\"")
set(dfloatFormat "\"%lf\"")
set(dlong "long int")
set(dlongString "\"long int\"")
set(dlongFormat="\"%ld\"")
set(hlong "long long int")
set(hlongString "\"long long int\"")
set(hlongFormat "\"%lld\"")
set(MPI_DFLOAT "MPI_DOUBLE")
set(MPI_DLONG "MPI_LONG_INT")
set(MPI_HLONG "MPI_LONG_LONG_INT")

#option(USE_MPI "Use the MPI library." ON)
#if(USE_MPI)
#    find_package(MPI)
#endif()

#option(EXTERNAL_BLASLAPACK "Use an external/vendor-provided Blas/Lapack library" OFF)
#if(EXTERNAL_BLASLAPACK)
#  find_package(BLAS)
#  find_package(LAPACK)
#endif()

option(EXTERNAL_OCCA "Use an external OCCA build" ON)
if(EXTERNAL_OCCA)
  find_library(OCCA_LIBRARY_PATH NAMES occa libocca PATHS $ENV{OCCA_DIR}/lib)
  find_path(OCCA_INCLUDE_PATH NAMES occa.hpp occa.h PATHS $ENV{OCCA_DIR}/include)
  add_library(libocca UNKNOWN IMPORTED)
  set_target_properties(libocca PROPERTIES IMPORTED_LOCATION ${OCCA_LIBRARY_PATH})
  target_include_directories(libocca INTERFACE ${OCCA_INCLUDE_PATH})
endif()

add_compile_definitions(
    "USE_OCCA_MEM_BYTE_ALIGN=64"
    "dfloat=double"
    "dfloatString=\"double\""
    "dfloatFormat=\"%lf\""
    "MPI_DFLOAT=MPI_DOUBLE"
    "dlong=int"
    "dlongString=\"int\""
    "dlongFormat=\"%d\""
    "MPI_DLONG=MPI_INT"
    "hlong=long long int"
    "hlongString=\"long long int\""
    "hlongFormat=\"%lld\""
    "MPI_HLONG=MPI_LONG_LONG_INT"
) 

add_executable(burger main.cpp)

include_directories(/lus/grand/projects/catalyst/world-shared/spatel/occa/src)

include_directories(/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/lib/python3.8/site-packages/tensorflow/include/external/local_config_python/numpy_include)
include_directories(/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/include)
include_directories(/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/include/python3.8)
include_directories(/soft/hpc-sdk/Linux_x86_64/21.7/cuda/include)

find_library(PYTHONLIB python3.8 HINTS /lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/lib)

target_include_directories(burger PUBLIC ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(burger 
  PUBLIC
    libocca
    ${PYTHONLIB}
)

install(TARGETS burger RUNTIME DESTINATION ".")
install(DIRECTORY kernel/ DESTINATION kernel)
install(FILES python_module.py DESTINATION ".")
