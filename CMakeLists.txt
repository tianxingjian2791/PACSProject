cmake_minimum_required(VERSION 3.15)

# Project name
set(TARGET "PACSProject")
project(${TARGET} LANGUAGES C CXX)

# Glob source files
file(GLOB_RECURSE TARGET_SRC "src/*.cpp")
set(TARGET_SRC ${TARGET_SRC})

# Deal.II configuration
find_package(deal.II 9.5.0
  COMPONENTS PETSc
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../dealii-install ../../ $ENV{DEAL_II_DIR}
)
if(NOT deal.II_FOUND)
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
  )
endif()

deal_ii_initialize_cached_variables()

# Find HDF5 (C and C++ interfaces)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
if(HDF5_FOUND)
  # Add HDF5 include directories
  list(APPEND DEAL_II_EXTRA_INCLUDES ${HDF5_INCLUDE_DIRS})
  # Add HDF5 libraries
  list(APPEND DEAL_II_EXTRA_LIBRARIES HDF5::HDF5_C HDF5::HDF5_CXX)
else()
  message(FATAL_ERROR "HDF5 not found. Please install HDF5 and retry.")
endif()

# Autopilot generates targets and links with deal.II and extra libraries
deal_ii_invoke_autopilot()