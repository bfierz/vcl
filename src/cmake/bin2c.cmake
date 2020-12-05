#
# This file is part of the Visual Computing Library (VCL) release under the
# MIT license.
#
# Copyright (c) 2020 Basil Fierz
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

file(WRITE ${OUTPUT_C} "/* Autogenerated by VCL bin2c */\n\n")
file(APPEND ${OUTPUT_C} "#include <cstddef>\n")
file(APPEND ${OUTPUT_C} "#include <cstdint>\n\n")

file(WRITE ${OUTPUT_H} "/* Autogenerated by VCL bin2c */\n\n")
file(APPEND ${OUTPUT_H} "#pragma once\n")
file(APPEND ${OUTPUT_H} "#include <cstddef>\n")
file(APPEND ${OUTPUT_H} "#include <cstdint>\n")
file(APPEND ${OUTPUT_H} "#include <vcl/core/span.h>\n\n")

file(READ ${INPUT_FILE} filedata HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})

file(APPEND ${OUTPUT_C} "uint8_t ${SYMBOL}Data[] = { ${filedata} };\nsize_t ${SYMBOL}Size = sizeof(${SYMBOL}Data);\n")
file(APPEND ${OUTPUT_H} "extern uint8_t ${SYMBOL}Data[];\nextern size_t ${SYMBOL}Size;\nconst stdext::span<const uint8_t> ${SYMBOL}{${SYMBOL}Data, ${SYMBOL}Size};\n")