# --- libgraphCutIterations/CMakeLists.txt ---

# Define the shared library and ALL of its source files, including those for maxflow
add_library(graphCutIterations SHARED
    graphCutIterations.cpp
    createExpansionGraphVARPRO_fast.cpp
    findLocalMinima.cpp
    ${CMAKE_SOURCE_DIR}/include/maxflow/graph.cpp
    ${CMAKE_SOURCE_DIR}/include/maxflow/maxflow.cpp
)

# Tell this library to use the downloaded Eigen
target_link_libraries(graphCutIterations PRIVATE eigen)

# Add the include directories for both this library's headers and for the maxflow headers
target_include_directories(graphCutIterations PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/maxflow
)

target_compile_options(graphCutIterations PRIVATE -O3 -fPIC -w)

# Install rule for scikit-build-core to pick up the binary
install(TARGETS graphCutIterations
    LIBRARY DESTINATION package
    ARCHIVE DESTINATION package
)

