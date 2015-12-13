
# Set the paths to the google benchmark libraries
SET(VCL_GOOGLEBENCHMARK_INCLUDE_DIR CACHE PATH "Include directory of gtest")
SET(VCL_GOOGLEBENCHMARK_LIBRARY_DEBUG CACHE FILEPATH "Library directory containing the gtest debug libraries")
SET(VCL_GOOGLEBENCHMARK_LIBRARY_RELEASE CACHE FILEPATH "Library directory containing the gtest release libraries")
INCLUDE_DIRECTORIES(${VCL_GOOGLEBENCHMARK_INCLUDE_DIR})

# Performance benchmarks for mutexes
SUBDIRS(mutex)

# Performance benchmarks for 3x3 Eigen deompositions
SUBDIRS(eigen33performance)

# Correctness benchmarks for 3x3 Eigen deompositions
SUBDIRS(eigen33correctness)

# Performance benchmarks for 3x3 SVDs
SUBDIRS(svd33performance)

# Correctness benchmarks for 3x3 SVDs
SUBDIRS(svd33correctness)
