# --- PyCSEMRI_clean/CMakeLists.txt ---

cmake_minimum_required(VERSION 3.15)
project(PyCSEMRI CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Add FetchContent to download Eigen ---
include(FetchContent)

# --- Declare Dependencies ---
FetchContent_Declare(
  eigen
  URL      file://${PROJECT_SOURCE_DIR}/include/minimal-eigen.tar.bz2
)

# --- New Method for Boost (Headers-Only) ---
FetchContent_Declare(
  boost
  # Switch to the .zip archive format
  URL      file://${PROJECT_SOURCE_DIR}/include/minimal_boost.tar.bz2
)

#  Make the content available ---
FetchContent_MakeAvailable(eigen)
FetchContent_MakeAvailable(boost)


add_subdirectory(libuwwfs)
add_subdirectory(libgraphCutIterations)
add_subdirectory(libdecomposeGivenFieldMapAndDampings)

