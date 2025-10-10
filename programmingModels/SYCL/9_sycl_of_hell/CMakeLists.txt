cmake_minimum_required(VERSION 3.22.1)
project(syclOfHell LANGUAGES CXX)

# Append local cmake directory to find CMAKE Modules
if( CMAKE_MODULE_PATH )
  list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
else()
  set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If you installed the oneAPI CMake package, prefer CONFIG mode
find_package(SYCL REQUIRED)

file(GLOB files "*.cpp")
foreach(file_name ${files})
  get_filename_component(exe_name ${file_name} NAME_WE)
  add_executable(${exe_name} ${file_name} argparse.hpp)

  if (TARGET SYCL::SYCL)
    # Preferred: oneAPI provides an imported target that sets the right flags
    target_link_libraries(${exe_name} PRIVATE SYCL::SYCL)
  else()
    # Fallback: manually add flags if no imported target is available
    target_compile_options(${exe_name} PRIVATE -fsycl)
    target_link_options(${exe_name}    PRIVATE -fsycl)
  endif()
endforeach()
