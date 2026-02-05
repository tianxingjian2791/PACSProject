cmake_minimum_required(VERSION 3.15)

# Project name
project(PACSProject LANGUAGES C CXX)

# Include directories
include_directories(include)

# Find deal.II
find_package(deal.II 9.5.0
  COMPONENTS PETSc
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../dealii-install $ENV{DEAL_II_DIR}
)

if(NOT deal.II_FOUND)
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
  )
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Initialize deal.II cached variables
deal_ii_initialize_cached_variables()

# Target 1: Original main executable
set(TARGET "PACSProject")
add_executable(${TARGET} src/main.cpp)
deal_ii_setup_target(${TARGET})
target_link_libraries(${TARGET} ${OpenMP_CXX_LIBRARIES})

# Target 2: Test unified data generator (simplified)
set(TARGET_TEST "generate_unified")
add_executable(${TARGET_TEST} src/generate_unified_data_test.cpp)
deal_ii_setup_target(${TARGET_TEST})
target_link_libraries(${TARGET_TEST} ${OpenMP_CXX_LIBRARIES})

# Target 3: Production data generator (configurable sizes)
set(TARGET_PROD "generate_production")
add_executable(${TARGET_PROD} src/generate_production_data.cpp)
deal_ii_setup_target(${TARGET_PROD})
target_link_libraries(${TARGET_PROD} ${OpenMP_CXX_LIBRARIES})

# Target 4: Extra-large data generator (10,000+ samples)
set(TARGET_XLARGE "generate_xlarge")
add_executable(${TARGET_XLARGE} src/generate_xlarge_data.cpp)
deal_ii_setup_target(${TARGET_XLARGE})
target_link_libraries(${TARGET_XLARGE} ${OpenMP_CXX_LIBRARIES})

message(STATUS "")
message(STATUS "Configured targets:")
message(STATUS "  PACSProject - Original data generator")
message(STATUS "  generate_unified - Test unified generator (4 samples)")
message(STATUS "  generate_production - Production generator (configurable)")
message(STATUS "  generate_xlarge - Extra-large generator (10,000+ samples)")
message(STATUS "")
