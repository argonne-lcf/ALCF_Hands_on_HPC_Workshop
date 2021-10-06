project(FortranLoadData)

cmake_minimum_required(VERSION 3.10)

enable_language(Fortran)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

find_package(MPI)

find_library(SR_LIB smartredis PATHS /projects/SDL_Workshop/sdl_ai_workshop/05_Simulation_ML/ML_Fortran_SmartSim/smartredis-0.2.0/install/lib NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ${MPI_INCLUDE_PATH}
    /projects/SDL_Workshop/sdl_ai_workshop/05_Simulation_ML/ML_Fortran_SmartSim/smartredis-0.2.0/install/include
)

set(ftn_client_src
    /projects/SDL_Workshop/sdl_ai_workshop/05_Simulation_ML/ML_Fortran_SmartSim/smartredis-0.2.0/src/fortran/fortran_c_interop.F90
    /projects/SDL_Workshop/sdl_ai_workshop/05_Simulation_ML/ML_Fortran_SmartSim/smartredis-0.2.0/src/fortran/dataset.F90
    /projects/SDL_Workshop/sdl_ai_workshop/05_Simulation_ML/ML_Fortran_SmartSim/smartredis-0.2.0/src/fortran/client.F90
)

# build executables

add_executable(dataLoaderFtn
	load_data.f
	${ftn_client_src}
)
target_link_libraries(dataLoaderFtn
	MPI::MPI_Fortran
	${SR_LIB}
)

