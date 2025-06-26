# --- PyCSEMRI_clean/CMakeLists.txt ---

cmake_minimum_required(VERSION 3.15)
project(PyCSEMRI CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Add FetchContent to download Eigen ---
include(FetchContent)

# Declare the Eigen project content
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0 # A recent, stable version of Eigen
)

# Make Eigen available to our project. This will download it if needed
# and create an 'eigen' subdirectory in our build folder.
FetchContent_MakeAvailable(eigen)
# ----------------------------------------

# Add our library subdirectories
add_subdirectory(libuwwfs)
add_subdirectory(libgraphCutIterations)
add_subdirectory(libdecomposeGivenFieldMapAndDampings)

