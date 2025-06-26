# --- PyCSEMRI_clean/CMakeLists.txt ---

# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.15)

# Define the project name
project(PyCSEMRI CXX)

# Set the C++ standard for the whole project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the subdirectories that contain the individual library CMakeLists.txt files.
# The install rules are now inside each of these.
add_subdirectory(libuwwfs)
add_subdirectory(libgraphCutIterations)
add_subdirectory(libdecomposeGivenFieldMapAndDampings)
