# --- libgraphCutIterations/CMakeLists.txt ---

add_library(graphCutIterations SHARED
    graphCutIterations.cpp
    createExpansionGraphVARPRO_fast.cpp
    findLocalMinima.cpp
    ../include/maxflow/graph.cpp
    ../include/maxflow/maxflow.cpp
)

# Tell this library to use the downloaded Eigen
target_link_libraries(graphCutIterations PRIVATE eigen)

# We still need the other include path for maxflow
target_include_directories(graphCutIterations PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_compile_options(graphCutIterations PRIVATE -O3 -fPIC -w)

install(TARGETS graphCutIterations LIBRARY DESTINATION package ARCHIVE DESTINATION package)

