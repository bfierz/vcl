#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2018 Basil Fierz
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
include(GoogleTest)
include(VCLConfig)

# Create a test executable using Google Test
function(vcl_add_test tgt)

	vcl_check_target(gtest)
	vcl_check_target(gtest_main)
	
    add_executable(${tgt} "")
	
	# Place into VS solution folder 'tests'
	set_target_properties(${tgt} PROPERTIES FOLDER tests)

	# Link against gtest
	target_link_libraries(${tgt}
		gtest
		gtest_main
	)
	
	# Enable static code analysis
	if(VCL_ENABLE_CORE_GUIDELINE_CHECKER)
		enable_vs_guideline_checker(vcl_core_test)
	endif()
	
	# Enable code coverage recording
	if(VCL_CODE_COVERAGE AND CMAKE_COMPILER_IS_GNUCXX)
		setup_target_for_coverage_gcovr_xml(
			NAME ${tgt}.coverage
			EXECUTABLE ${tgt}
			DEPENDENCIES ${tgt}
		)
	endif()
	
	# Register for test auto-discovery
	gtest_discover_tests(${tgt})
endfunction()
