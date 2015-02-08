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
#include <iostream>
#include <filesystem>
#include <sstream>
#include <vector>

// Boost
#include <boost/program_options.hpp>

// Windows API
#ifdef VCL_ABI_WINAPI
#	include <Windows.h>
#else
VCL_ERROR("No compatible process API found.")
#endif

namespace po = boost::program_options;

namespace Vcl { namespace Tools { namespace Clc
{
	enum class Compiler
	{
		Msvc,
		Clang,
		Gcc,
		Intel
	};

	void displayError(LPCTSTR errorDesc, DWORD errorCode)
	{
		TCHAR errorMessage[1024] = TEXT("");

		DWORD flags = FORMAT_MESSAGE_FROM_SYSTEM
			| FORMAT_MESSAGE_IGNORE_INSERTS
			| FORMAT_MESSAGE_MAX_WIDTH_MASK;

		FormatMessage(flags,
			NULL,
			errorCode,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			errorMessage,
			sizeof(errorMessage) / sizeof(TCHAR),
			NULL);

#ifdef _UNICODE
		std::wcerr << L"Error : " << errorDesc << std::endl;
		std::wcerr << L"Code    = " << errorCode << std::endl;
		std::wcerr << L"Message = " << errorMessage << std::endl;
#else
		std::cerr << "Error : " << errorDesc << std::endl;
		std::cerr << "Code    = " << errorCode << std::endl;
		std::cerr << "Message = " << errorMessage << std::endl;
#endif
	}

	int exec(const char* prg, const char* params = nullptr)
	{
		STARTUPINFO si;
		PROCESS_INFORMATION pi;

		// Initialize memory
		ZeroMemory(&si, sizeof(STARTUPINFO));
		si.cb = sizeof(STARTUPINFO);
		si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
		si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
		si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
		
		// Construct the command line
		const char* separator = " ";
		const char* terminator = "\0";
		std::vector<char> cmd;
		cmd.reserve(strlen(prg) + strlen(params) + 2);

		std::copy(prg, prg + strlen(prg), std::back_inserter(cmd));

		if (params)
		{
			std::copy(separator, separator + 1, std::back_inserter(cmd));
			std::copy(params, params + strlen(params), std::back_inserter(cmd));
		}
		std::copy(terminator, terminator + 1, std::back_inserter(cmd));

		if (CreateProcess(
			nullptr,    //_In_opt_     LPCTSTR lpApplicationName,
			cmd.data(), //_Inout_opt_  LPTSTR lpCommandLine,
			nullptr,    //_In_opt_     LPSECURITY_ATTRIBUTES lpProcessAttributes,
			nullptr,    //_In_opt_     LPSECURITY_ATTRIBUTES lpThreadAttributes,
			FALSE,      //_In_         BOOL bInheritHandles,
			0,          //_In_         DWORD dwCreationFlags,
			nullptr,    //_In_opt_     LPVOID lpEnvironment,
			nullptr,    //_In_opt_     LPCTSTR lpCurrentDirectory,
			&si,        //_In_         LPSTARTUPINFO lpStartupInfo,
			&pi         //_Out_        LPPROCESS_INFORMATION lpProcessInformation
		) == FALSE)
		{
			DWORD err = GetLastError();
			displayError(TEXT("Unable to execute."), err);

			return -1;
		}

		// Wait until the process completed
		WaitForInputIdle(pi.hProcess, INFINITE);

		DWORD exit_code;
		GetExitCodeProcess(pi.hProcess, &exit_code);

		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);

		std::flush(std::cout);

		return exit_code;
	}
}}}

int main(int argc, char* argv [])
{
	using namespace Vcl::Tools::Clc;
	namespace fs = std::tr2::sys;

	// Declare the supported options.
	po::options_description desc
	("Usage: clc [options]\n\nOptions");
	desc.add_options()
		("help", "Print this help information on this tool.")
		("version", "Print version information on this tool.")
		("compiler", po::value<std::string>(), "Compiler providing the preprocessor. Allowed options are clang, gcc, msvc, intel")
		("include,I", po::value<std::vector<std::string>>(), "Additional include directory")
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

#ifdef VCL_COMPILER_MSVC
	std::string compiler = "cl";
	char param_tok = '/';
	Compiler format = Compiler::Msvc;
#endif

	if (vm.count("compiler") > 0)
	{
		if (vm["compiler"].as<std::string>() == "msvc")
		{
			compiler = "cl";
			param_tok = '/';
			format = Compiler::Msvc;
		}
		else if (vm["compiler"].as<std::string>() == "clang")
		{
			compiler = "clang";
			param_tok = '-';
			format = Compiler::Clang;
		}
		else if (vm["compiler"].as<std::string>() == "gcc")
		{
			compiler = "gcc";
			param_tok = '-';
			format = Compiler::Gcc;
		}
		else
		{
			std::cerr << "Invalid compiler string" << std::endl;
			std::cout << desc << std::endl;
			return -1;
		}
	}

	// Generate intermediate file name
	std::string preprocess_file = fs::basename(fs::path{ vm["input-file"].as<std::string>() }) + ".i";

	// Preprocess the source file
	std::stringstream cmd;

	if (format == Compiler::Msvc)
	{
		// Add preprocessing command
		cmd << "/P ";

		// Add the output file
		cmd << "/Fi: " << preprocess_file << " ";
	}

	// Add include directories
	if (vm.count("include"))
	{
		for (auto& inc : vm["include"].as<std::vector<std::string>>())
		{
			cmd << param_tok << "I \"" << inc << "\" ";
		}
	}

	// Add the input file
	cmd << R"(")" << vm["input-file"].as<std::string>() << R"(")";

	// Invoke the preprocessor
	exec(compiler.c_str(), cmd.str().c_str());

	// Load the intermediate file and generate the binary file
	cmd.str("");
	cmd.clear();
	cmd << "--group 4 ";
	cmd << "-o " << vm["output-file"].as<std::string>() << " ";
	cmd << preprocess_file;

	// Invoke the binary file translator
	exec("bin2c", cmd.str().c_str());

	return 0;
}
