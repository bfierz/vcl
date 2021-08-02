/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2019 Basil Fierz
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
#ifdef VCL_ABI_WINAPI
#	if VCL_HAS_STDCXX17
#		include <filesystem>
#	else
#		define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#		include <experimental/filesystem>
#	endif
#elif defined(VCL_ABI_POSIX)
#	include <boost/filesystem.hpp>
#endif
#include <iostream>
#include <vector>

VCL_BEGIN_EXTERNAL_HEADERS
// CxxOpts
#include <vcl/core/3rdparty/cxxopts.hpp>
VCL_END_EXTERNAL_HEADERS

// Local
#include "kernelwrapper.h"

// Windows API
#ifdef VCL_ABI_WINAPI
#	include <Windows.h>
#else
VCL_ERROR("No compatible process API found.")
#endif

// Copy entire file to container
// Source: http://cpp.indi.frih.net/blog/2014/09/how-to-read-an-entire-file-into-memory-in-cpp/
template <typename Char, typename Traits, typename Allocator = std::allocator<Char>>
static std::basic_string<Char, Traits, Allocator> read_stream_into_string
(
	std::basic_istream<Char, Traits>& in,
	Allocator alloc = {}
)
{
	std::basic_ostringstream<Char, Traits, Allocator> ss
	(
		std::basic_string<Char, Traits, Allocator>(std::move(alloc))
	);

	if (!(ss << in.rdbuf()))
		throw std::ios_base::failure{ "error" };

	return ss.str();
}

namespace Vcl { namespace Tools { namespace Cui {
	void displayError(LPCTSTR errorDesc, DWORD errorCode)
	{
		TCHAR errorMessage[1024] = TEXT("");

		DWORD flags = FORMAT_MESSAGE_FROM_SYSTEM
			| FORMAT_MESSAGE_IGNORE_INSERTS
			| FORMAT_MESSAGE_MAX_WIDTH_MASK;

		FormatMessage(flags,
			nullptr,
			errorCode,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			errorMessage,
			sizeof(errorMessage) / sizeof(TCHAR),
			nullptr);

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

	void createIoPipe(HANDLE& hRead, HANDLE& hWrite)
	{
		SECURITY_ATTRIBUTES saAttr;

		// Set the bInheritHandle flag so pipe handles are inherited.
		saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
		saAttr.bInheritHandle = TRUE;
		saAttr.lpSecurityDescriptor = nullptr;

		// Create a pipe for the child process's IO
		if (!CreatePipe(&hRead, &hWrite, &saAttr, 0))
			return;

		// Ensure the read handle to the pipe for IO is not inherited.
		if (!SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0))
			return;
	}

	void readFromPipe(HANDLE hProcess, HANDLE hRead)
	{
		DWORD dwAvail, dwRead, dwWritten;
		CHAR chBuf[1024];
		BOOL bSuccess = FALSE;
		HANDLE hParentStdOut = GetStdHandle(STD_OUTPUT_HANDLE);

		for (;;)
		{
			DWORD exit_code;
			GetExitCodeProcess(hProcess, &exit_code); //while the process is running
			if (exit_code != STILL_ACTIVE)
				break;

			PeekNamedPipe(hRead, chBuf, 1024, &dwRead, &dwAvail, nullptr);
			if (dwAvail == 0)
				continue;

			bSuccess = ReadFile(hRead, chBuf, 1024, &dwRead, nullptr);
			if (!bSuccess || dwRead == 0)
				break;

			bSuccess = WriteFile(hParentStdOut, chBuf, dwRead, &dwWritten, nullptr);
			if (!bSuccess)
				break;
		}
	}

	int exec(const char* prg, const char* params = nullptr)
	{
		STARTUPINFO si;
		PROCESS_INFORMATION pi;

		// Create the IO pipe
		HANDLE hWrite, hRead;
		createIoPipe(hRead, hWrite);

		// Initialize memory
		ZeroMemory(&si, sizeof(STARTUPINFO));
		si.cb = sizeof(STARTUPINFO);
		si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
		si.hStdOutput = hWrite; //GetStdHandle(STD_OUTPUT_HANDLE);
		si.hStdError = hWrite;  //GetStdHandle(STD_ERROR_HANDLE);
		si.dwFlags |= STARTF_USESTDHANDLES;

		// Construct the command line
		const char* separator = " ";
		const char* terminator = "\0";
		std::vector<char> cmd;
		cmd.reserve(strlen(prg) + (params ? strlen(params) : 0) + 2);

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
			TRUE,       //_In_         BOOL bInheritHandles,
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

		// Read all the output
		readFromPipe(pi.hProcess, hRead);

		// Successfully created the process.  Wait for it to finish.
		WaitForSingleObject(pi.hProcess, INFINITE);

		DWORD exit_code;
		GetExitCodeProcess(pi.hProcess, &exit_code);

		CloseHandle(pi.hThread);
		CloseHandle(pi.hProcess);
		CloseHandle(hRead);
		CloseHandle(hWrite);

		std::flush(std::cout);

		return exit_code;
	}
}}}

int main(int argc, char* argv[])
{
	using namespace Vcl::Tools::Cui;

#ifdef VCL_ABI_WINAPI
#	if VCL_HAS_STDCXX17
	namespace fs = std::filesystem;
#	else
	namespace fs = std::experimental::filesystem;
#	endif
#elif defined(VCL_ABI_POSIX)
	namespace fs = boost::filesystem;
#endif

	cxxopts::Options options(argv[0], "cui - command line options");

	try
	{
		options.add_options()
			("help", "Print this help information on this tool.")
			("version", "Print version information on this tool.")
			("nvcc", "Specify which nvcc should be used.", cxxopts::value<std::string>())
			("I,include", "Additional include directory", cxxopts::value<std::vector<std::string>>())
			("m,module-file", "Specify the module file to pack.", cxxopts::value<std::string>())
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

		fs::path nvcc_bin_path;
		if (parsed_options.count("nvcc"))
		{
			fs::path exe_path = parsed_options["nvcc"].as<std::string>();
			nvcc_bin_path = exe_path.parent_path();
		} else
		{
			nvcc_bin_path = getenv("CUDA_PATH");
			nvcc_bin_path.append("bin");
		}

		if (parsed_options.count("input-file") == 0)
		{
			std::cout << options.help({ "" }) << std::endl;
			return -1;
		}
		if (parsed_options.count("module-file") == 0)
		{
			std::cout << options.help({ "" }) << std::endl;
			return -2;
		}
		if (parsed_options.count("output-file") == 0)
		{
			std::cout << options.help({ "" }) << std::endl;
			return -3;
		}

		std::vector<std::string> params;
		if (parsed_options.count("include"))
		{
			for (auto& inc : parsed_options["include"].as<std::vector<std::string>>())
			{
				params.emplace_back("-I");
				params.emplace_back("\"" + inc + "\"");
			}
		}
		params.emplace_back(parsed_options["input-file"].as<std::string>());
		const auto kernels = parseCudaKernels(nvcc_bin_path.parent_path().string(), params);

		// Read the binary module
		std::ifstream fatbin_reader;
		fatbin_reader.open(parsed_options["module-file"].as<std::string>(), std::ios_base::binary);
		if (!fatbin_reader.is_open())
		{
			std::cout << "Couldn't open module file: " << parsed_options["module-file"].as<std::string>() << std::endl;
			return -1;
		}
		std::string module_buffer = read_stream_into_string(fatbin_reader);
		fatbin_reader.close();

		// Open the output file
		std::ofstream ww;
		ww.open(parsed_options["output-file"].as<std::string>(), std::ios_base::binary);
		createWrappers(ww, kernels, module_buffer);
		ww.close();
	} catch (const cxxopts::OptionException& e)
	{
		std::cout << "Error parsing options: " << e.what() << std::endl;
		return 1;
	}
	return 0;
}
