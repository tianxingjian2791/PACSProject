cmake_minimum_required(VERSION 3.15)

# Project name
set(TARGET "PACSProject")
project(${TARGET} LANGUAGES C CXX)

# Glob source files
file(GLOB_RECURSE TARGET_SRC "src/*.cpp")
set(TARGET_SRC ${TARGET_SRC})

# include
include_directories(include)

# Deal.II configuration
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

deal_ii_initialize_cached_variables()

# Autopilot generates targets and links with deal.II and extra libraries
deal_ii_invoke_autopilot()