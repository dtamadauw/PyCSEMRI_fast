# --- libgraphCutIterations/CMakeLists.txt ---

add_library(graphCutIterations SHARED
    graphCutIterations.cpp
    createExpansionGraphVARPRO_fast.cpp
    findLocalMinima.cpp
    ${CMAKE_SOURCE_DIR}/include/maxflow/graph.cpp
    ${CMAKE_SOURCE_DIR}/include/maxflow/maxflow.cpp
)
target_link_libraries(graphCutIterations PRIVATE eigen)
target_include_directories(graphCutIterations PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/maxflow
)
target_compile_options(graphCutIterations PRIVATE -O3 -fPIC -w)

# The destination is now 'pycsemri'
install(TARGETS graphCutIterations
    LIBRARY DESTINATION pycsemri
    ARCHIVE DESTINATION pycsemri
)

