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

// Boost
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv [])
{
	// Declare the supported options.
	po::options_description desc
		("Usage: cuc [options]\n\nOptions");
	desc.add_options()
		("help", "Print this help information on this tool.")
		("version", "Print version information on this tool.")
		("group", po::value<int>(), "Number of bytes written together. Valid values are 1, 2, 4 and 8.")
		("symbol", po::value<std::string>(), "Specify the symbol name used for the converted data.")
		("output-file,o", po::value<std::string>(), "Specify the output file.")
		("input-file", po::value<std::string>(), "Specify the input file.")
		;

	po::positional_options_description p;
	p.add("input-file", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	po::notify(vm);

	// Print the help message
	if (vm.count("help") > 0)
	{
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("input-file") == 0 || vm.count("output-file") == 0)
	{
		std::cout << desc << std::endl;
		return -1;
	}


	// Output width
	int width = 1;
	if (vm.count("group") > 0)
	{
		int group = vm["group"].as<int>();
		if (!(group == 1 || group == 2 || group == 4 || group == 8))
		{
			std::cout << desc << std::endl;
			return -1;
		}

		width = group;
	}

	std::string width_symbol;
	switch (width)
	{
	case 1:
		width_symbol = "uint8_t";
	case 2:
		width_symbol = "uint16_t";
	case 4:
		width_symbol = "uint32_t";
	case 8:
		width_symbol = "uint64_t";
	}

	// Export symbol name
	std::string export_symbol{ "BinData" };
	if (vm.count("symbol") > 0)
	{
		export_symbol = vm["symbol"].as<std::string>();
	}

	std::ifstream ifile{ "main.cpp" };
	if (ifile.is_open())
	{
		// Copy the file to a temporary buffer
		std::istream_iterator<char> bos{ ifile };
		std::istream_iterator<char> eos{};
		std::vector<char> tmp_buffer(bos, eos);

		// Pad to the requested output width
		for (int i = 0; i < width - 1; i++)
		{
			tmp_buffer.push_back(0);
		}

		// Write the temporary buffer to the output
		std::ofstream ofile{ "test.txt" };
		if (ofile.is_open())
		{
			// Write header
			ofile << R"(#include <cstdint>)" << "\n";
			ofile << width_symbol << " " << export_symbol << " = \n{\n";

			// Write data
			ofile << "\t";
			for (int e = 0; e + width <= (int) tmp_buffer.size(); e += width)
			{
				ofile << "0x";
				for (int i = 0; i < width; i++)
				{
					ofile << std::setfill('0') << std::hex << std::setw(2) << (unsigned int) tmp_buffer[e + i];
				}

				if (e + 2*width <= (int) tmp_buffer.size())
				{
					ofile << ", ";

					if ((e + width - 1) % 16 == 15)
						ofile << "\n\t";
				}
			}

			// Write footer
			ofile << "\n};\n";
		}
	}

	return 0;
}
