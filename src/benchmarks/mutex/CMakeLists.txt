PROJECT(mutexperformance)

# Status message
MESSAGE(STATUS "Configuring 'mutexperformance'")

# Boost dependency
FIND_PACKAGE(Boost 1.55 COMPONENTS chrono date_time system thread REQUIRED)
MESSAGE(STATUS "* Boost Include: ${Boost_INCLUDE_DIR}")
MESSAGE(STATUS "* Boost Libraries: ${Boost_LIBRARY_DIRS}")
MESSAGE(STATUS "* Boost Libraries: ${Boost_LIBRARIES}")

# Instruct CMake to run moc automatically when needed.
SET(CMAKE_AUTOMOC ON)

# Find the QtCore library
FIND_PACKAGE(Qt5Core REQUIRED)

# Do not automatically link Qt executables to qtmain target on Windows.
# Uses console mode only. Not need to link against qtmain.lib.
cmake_policy(SET CMP0020 OLD)

SET(SRC
	main.cpp
)

SOURCE_GROUP("" FILES ${SRC})

# Include dependencies
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

ADD_EXECUTABLE(mutexperformance ${SRC})
SET_TARGET_PROPERTIES(mutexperformance PROPERTIES FOLDER benchmarks)

TARGET_LINK_LIBRARIES(mutexperformance
	vcl_core
	${Boost_LIBRARIES}
	Qt5::Core
)
