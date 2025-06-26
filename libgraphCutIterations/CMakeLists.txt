# --- libgraphCutIterations/CMakeLists.txt ---

add_library(graphCutIterations SHARED
    graphCutIterations.cpp
    createExpansionGraphVARPRO_fast.cpp
    findLocalMinima.cpp
    ../include/maxflow/graph.cpp
    ../include/maxflow/maxflow.cpp
)

target_include_directories(graphCutIterations PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include/Eigen
    ${CMAKE_SOURCE_DIR}/include/maxflow
)

target_compile_options(graphCutIterations PRIVATE -O3 -fPIC -w)

# Install rule for scikit-build-core to pick up the binary
install(TARGETS graphCutIterations
    LIBRARY DESTINATION package
    ARCHIVE DESTINATION package
)
