#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2015 Basil Fierz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
project(mutexperformance)

# Status message
message(STATUS "Configuring 'mutexperformance'")

# Boost dependency
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.55 COMPONENTS chrono date_time system thread)

# Find the thread library
find_package(Threads REQUIRED)

# Find the QtCore library
find_package(Qt5Core)

if (Boost_FOUND AND Qt5Core_FOUND)
	set(SRC
		main.cpp
	)

	source_group("" FILES ${SRC})

	add_executable(mutexperformance ${SRC})
	set_target_properties(mutexperformance PROPERTIES FOLDER benchmarks)

	target_link_libraries(mutexperformance
		vcl_core
		Boost::thread
		Qt5::Core
		benchmark
	)
endif()
