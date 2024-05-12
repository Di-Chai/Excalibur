cmake_minimum_required(VERSION 3.20)

set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 17)

# set the project name 
project(DecFedSVD)

set(OpenMP_CXX_FLAGS "-qopenmp")
set(OpenMP_CXX_LIB_NAMES "libiomp5")
set(OpenMP_libiomp5_LIBRARY libiomp5.so)

include_directories(${MKLROOT}/include)
link_directories(${MKLROOT}/lib/intel64)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(IntelSYCL REQUIRED)
find_package(Threads REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_compile_options(-O3 -mavx2)

aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRC)
add_library(MYLIBs SHARED ${DIR_SRC})

set(MKL_LIB "-lmkl_rt -lpthread -lm -ldl")

# add the executable 
add_executable(main main.cpp)
add_executable(gd generate_data.cpp)

target_link_libraries(main PUBLIC MYLIBs Threads::Threads ${MKL_LIB})
target_link_libraries(gd PUBLIC MYLIBs Threads::Threads ${MKL_LIB})
