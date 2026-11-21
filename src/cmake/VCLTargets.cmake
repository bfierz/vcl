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

# vcl_remove_compile_flag(target option)
# Removes all compile flags from @target that match the regex in @option.
function(vcl_remove_compile_flag target_ option_)
    get_target_property(compile_flags ${target_} COMPILE_FLAGS)
    string(REGEX REPLACE "${option_} " "" gtest_compile_flags ${compile_flags})
    set_target_properties(${target_} PROPERTIES COMPILE_FLAGS "${compile_flags}")
endfunction()

# vcl_replace_compile_flag(target option remove add)
# Removes all compile flags from @target that match the regex in @option; and adds the new flag given in @add.
function(vcl_replace_compile_flag target_ remove_ add_)
    get_target_property(compile_flags ${target_} COMPILE_FLAGS)
    string(REGEX REPLACE "${remove_} " "" gtest_compile_flags ${compile_flags})
    set_target_properties(${target_} PROPERTIES COMPILE_FLAGS "${compile_flags} ${add_}")
endfunction()
