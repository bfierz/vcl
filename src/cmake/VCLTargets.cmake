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

# Copy the target as aliasing imported targets is not possible (< 3.11)
# https://github.com/conan-io/conan/issues/2125
function(vcl_copy_target dst src)
    add_library(${dst} INTERFACE IMPORTED)
    foreach(name INTERFACE_LINK_LIBRARIES INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS)
        get_property(value TARGET ${src} PROPERTY ${name} )
        set_property(TARGET ${dst} PROPERTY ${name} ${value})
    endforeach()
endfunction()

# vcl_combine_targets_property(out_var target_prop target1 target2 ...)
# Helper function: Collects @target_prop properties (as lists) from @target1, @target2 ..,
# combines these lists into one and store into variable @out_var.
function(vcl_combine_targets_property out_var target_prop)
    set(values) # Resulted list
    foreach(t ${ARGN})
        get_property(v TARGET ${t} PROPERTY ${target_prop})
        list(APPEND values ${v})
    endforeach()
    set(${out_var} ${values} PARENT_SCOPE)
endfunction()

# vcl_combine_targets(dst target1 target2 ...)
# Creates a new target @dst which depends on @target1, @target2, ...
function(vcl_combine_targets dst)
    add_library(${dst} INTERFACE IMPORTED)
    set_property(TARGET ${dst} PROPERTY INTERFACE_LINK_LIBRARIES ${ARGN})
endfunction()

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
