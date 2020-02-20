#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2019 Basil Fierz
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
set(file_contents)

# Process include folders
list(REMOVE_DUPLICATES INCLUDE_DIRS)
foreach(dir ${INCLUDE_DIRS})
    list(APPEND include_dir_param "-I")
    list(APPEND include_dir_param "\"${dir}\"")
endforeach()

list(LENGTH OBJECTS file_count)
math(EXPR file_indexer "${file_count} - 1")

foreach(idx RANGE ${file_indexer})
  list(GET OBJECTS ${idx} obj)
  list(GET SOURCES ${idx} src)
  list(GET OUTPUTS ${idx} out)

  get_filename_component(obj_ext ${obj} EXT)
  get_filename_component(obj_name ${obj} NAME_WE)
  get_filename_component(obj_dir ${obj} DIRECTORY)

  if(obj_ext MATCHES ".ptx")
    set(args ${include_dir_param} -m ${obj} -o ${out} ${src})
    execute_process(COMMAND "${CUI_COMMAND}" ${args}
                    WORKING_DIRECTORY ${WORK_DIR}
                    RESULT_VARIABLE result
                    OUTPUT_VARIABLE output
                    ERROR_VARIABLE error_var
                    )
  endif()
endforeach()
