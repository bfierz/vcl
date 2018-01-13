
# For clang on Windows globally ignore unused commandline arguments
if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang" AND NOT WIN32)
	# Performance benchmarks for mutexes
	SUBDIRS(mutex)
endif()

# Performance benchmarks for 3x3 Eigen deompositions
SUBDIRS(eigen33performance)

# Correctness benchmarks for 3x3 Eigen deompositions
SUBDIRS(eigen33correctness)

# Performance benchmarks for 3x3 SVDs
SUBDIRS(svd33performance)

# Correctness benchmarks for 3x3 SVDs
SUBDIRS(svd33correctness)

# Performance benchmarks for 3x3 rotation computation
SUBDIRS(rotation33performance)

# Correctness benchmarks for 3x3 rotation computation
SUBDIRS(rotation33correctness)

# Benchmark collection for Vcl Core
SUBDIRS(vcl.core)

# Benchmark collection for Vcl Geometry
SUBDIRS(vcl.geometry)
