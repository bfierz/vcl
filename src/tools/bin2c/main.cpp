/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 - 2015 Basil Fierz
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// VCL configuration
#include <vcl/config/global.h>

// C++ Standard Library
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>
#include <utility>

// CxxOpts
#include <vcl/core/3rdparty/cxxopts.hpp>

// Copy entire file to container
// Source: http://cpp.indi.frih.net/blog/2014/09/how-to-read-an-entire-file-into-memory-in-cpp/
template<typename Char, typename Traits, typename Allocator = std::allocator<Char>>
std::basic_string<Char, Traits, Allocator> read_stream_into_string(
	std::basic_istream<Char, Traits>& in,
	Allocator alloc = {})
{
	std::basic_ostringstream<Char, Traits, Allocator> ss(
		std::basic_string<Char, Traits, Allocator>(std::move(alloc)));

	if (!(ss << in.rdbuf()))
		throw std::ios_base::failure{ "error" };

	return ss.str();
}

int main(int argc, char* argv[])
{
	cxxopts::Options options(argv[0], "bin2c - command line options");

	// Output width
	int width = 1;

	// Export symbol name
	std::string export_symbol{ "BinData" };

	// Input file
	std::ifstream ifile;

	// Write the temporary buffer to the output
	std::ofstream ofile;

	try
	{
		options.add_options()
			("help", "Print this help information on this tool.")
			("version", "Print version information on this tool.")
			("group", "Number of bytes written together. Valid values are 1, 2, 4 and 8.", cxxopts::value<int>())
			("symbol", "Specify the symbol name used for the converted data.", cxxopts::value<std::string>())
			("o,output-file", "Specify the output file.", cxxopts::value<std::string>())
			("input-file", "Specify the input file.", cxxopts::value<std::string>())
			;
		options.parse_positional("input-file");

		cxxopts::ParseResult parsed_options = options.parse(argc, argv);

		if (parsed_options.count("help") > 0)
		{
			std::cout << options.help({ "" }) << std::endl;
			return 1;
		}

		if (parsed_options.count("input-file") == 0 || parsed_options.count("output-file") == 0)
		{
			std::cout << options.help({ "" }) << std::endl;
			return 1;
		}

		if (parsed_options.count("group") > 0)
		{
			int group = parsed_options["group"].as<int>();
			if (!(group == 1 || group == 2 || group == 4 || group == 8))
			{
				std::cout << options.help({ "" }) << std::endl;
				return -1;
			}

			width = group;
		}

		if (parsed_options.count("symbol") > 0)
		{
			export_symbol = parsed_options["symbol"].as<std::string>();
		}

		ifile.open(parsed_options["input-file"].as<std::string>(), std::ios_base::binary | std::ios_base::in);
		ofile.open(parsed_options["output-file"].as<std::string>());
	} catch (const cxxopts::OptionException& e)
	{
		std::cout << "Error parsing options: " << e.what() << std::endl;
		return 1;
	}

	std::string width_symbol;
	switch (width)
	{
	case 1:
		width_symbol = "uint8_t";
		break;
	case 2:
		width_symbol = "uint16_t";
		break;
	case 4:
		width_symbol = "uint32_t";
		break;
	case 8:
		width_symbol = "uint64_t";
		break;
	}

	if (ifile.is_open())
	{
		// Copy the file to a temporary buffer
		std::string tmp_buffer = read_stream_into_string(ifile);
		ifile.close();

		// Pad to the requested output width
		for (int i = 0; i < width - 1; i++)
		{
			tmp_buffer.push_back(0);
		}

		if (ofile.is_open())
		{
			// Write header
			ofile << R"(#include <cstddef>)" << "\n";
			ofile << R"(#include <cstdint>)" << "\n";
			ofile << width_symbol << " " << export_symbol << "[] = \n{\n";

			// Write data
			ofile << "\t";
			for (int e = 0; e + width <= (int)tmp_buffer.size(); e += width)
			{
				ofile << "0x";
				for (int i = width - 1; i >= 0; i--)
				{
					unsigned int content = (unsigned int)(unsigned char)tmp_buffer[e + i];
					ofile << std::setfill('0') << std::hex << std::setw(2) << content;
				}

				if (e + 2 * width <= (int)tmp_buffer.size())
				{
					ofile << ", ";

					if ((e + width - 1) % 16 == 15)
						ofile << "\n\t";
				}
			}

			// Write footer
			ofile << "\n};\n";
			ofile << "\nsize_t " << export_symbol << "Size = sizeof(" << export_symbol << ") / sizeof(" << width_symbol << ");\n";
		}
	}

	return 0;
}
