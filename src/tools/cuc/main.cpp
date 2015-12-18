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
#ifdef VCL_ABI_WINAPI
#include <filesystem>
#elif defined(VCL_ABI_POSIX)
#include <boost/filesystem.hpp>
#endif
#include <iostream>
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

namespace Vcl { namespace Tools { namespace Cuc
{
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
			GetExitCodeProcess(hProcess, &exit_code);      //while the process is running
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
		si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
		si.hStdOutput = hWrite; //GetStdHandle(STD_OUTPUT_HANDLE);
		si.hStdError  = hWrite; //GetStdHandle(STD_ERROR_HANDLE);
		si.dwFlags |= STARTF_USESTDHANDLES;
		
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

int main(int argc, char* argv [])
{
	using namespace Vcl::Tools::Cuc;

#ifdef VCL_ABI_WINAPI
	namespace fs = std::tr2::sys;
#elif defined(VCL_ABI_POSIX)
	namespace fs = boost::filesystem;
#endif


	// Declare the supported options.
	po::options_description desc
	("Usage: cuc [options]\n\nOptions");
	desc.add_options()
		("help", "Print this help information on this tool.")
		("version", "Print version information on this tool.")
		("m64", "Specify that this should be compiled in 64bit.")
		("profile", po::value<std::vector<std::string>>(), "Target compute architectures (sm_20, sm_30, sm_35, sm_50, compute_20, compute_30, compute_35, compute_50)")
		("include,I", po::value<std::vector<std::string>>(), "Additional include directory")
		("symbol", po::value<std::string>(), "Name of the symbol used for the compiled module")
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

	if (vm.count("symbol") == 0 || vm.count("input-file") == 0 || vm.count("output-file") == 0)
	{
		std::cout << desc << std::endl;
		return -1;
	}

	std::vector<std::string> profiles;
	if (vm.count("profile"))
	{
		profiles = vm["profile"].as<std::vector<std::string>>();
	}

	// Construct the base name for the intermediate files
#if (_MSC_VER < 1900)
	std::string tmp_file_base = fs::basename(fs::path(vm["input-file"].as<std::string>()));
#else
	std::string tmp_file_base = fs::path(vm["input-file"].as<std::string>()).stem().string();
#endif

	// Add the address 
	if (vm.count("m64"))
	{
		tmp_file_base += "_m64";
	}
	else
	{
		tmp_file_base += "_m32";
	}

	// Invoke the cuda compiler for each profile
	std::vector<std::pair<std::string, std::string>> compiled_files;
	compiled_files.reserve(profiles.size());
	for (auto& p : profiles)
	{
		std::stringstream cmd;

		if (vm.count("include"))
		{
			for (auto& inc : vm["include"].as<std::vector<std::string>>())
			{
				cmd << "-I \"" << inc << "\" ";
			}
		}

		cmd << "-gencode=arch=";

		// Generate the output filename for intermediate file
		std::string tmp_file = tmp_file_base + "_";

		auto sm = p.find("sm");
		if (sm != p.npos)
		{
			cmd << "compute" << p.substr(2, p.npos) << ",code=" << p;
			cmd << " -cubin ";
			tmp_file += p + ".cubin";
		}
		else
		{
			cmd << p << ",code=" << p;
			cmd << " -ptx ";
			tmp_file += p + ".ptx";
		}
		compiled_files.emplace_back(p, tmp_file);
		cmd << "-o \"" << tmp_file << "\" \"" << vm["input-file"].as<std::string>() << "\"";

		exec("nvcc.exe", cmd.str().c_str());
	}

	// Create a fat binary from the compiled files 
	std::stringstream fatbin_cmdbuilder;

	// Create a new fatbin
	fatbin_cmdbuilder << R"(--create=")" << tmp_file_base << R"(.fatbin" )";

	// We want to create an embedded file
	//fatbin_cmdbuilder << R"(--embedded-fatbin=")" << tmp_file_base << R"(.fatbin.c" )";

	// Set the bitness
	if (vm.count("m64"))
	{
		fatbin_cmdbuilder << "-64 ";
	}
	else
	{
		fatbin_cmdbuilder << "-32 ";
	}

	// We are compiling cuda
	fatbin_cmdbuilder << R"(--cuda )";

	// Add a hash
	fatbin_cmdbuilder << R"(--key="xxxxxxxxxx" )";
	
	// Add the orignal filename as identifier
	fatbin_cmdbuilder << R"(--ident=")" << vm["symbol"].as<std::string>() << R"(" )";

	// Add all the created files
	for (auto& profile_file : compiled_files)
	{
		fatbin_cmdbuilder << R"("--image=profile=)" << profile_file.first << R"(,file=)" << profile_file.second << R"(" )";
	}

	exec("fatbinary.exe", fatbin_cmdbuilder.str().c_str());

	// Create a source file with the binary 
	std::stringstream bin2c_cmdbuilder;
	bin2c_cmdbuilder.str("");
	bin2c_cmdbuilder.clear();
	bin2c_cmdbuilder << "--group 4 ";

	if (vm.count("symbol"))
	{
		bin2c_cmdbuilder << "--symbol " << vm["symbol"].as<std::string>() << " ";
	}

	bin2c_cmdbuilder << "-o " << vm["output-file"].as<std::string>() << " ";
	bin2c_cmdbuilder << tmp_file_base << R"(.fatbin" )";

	// Invoke the binary file translator
	exec("bin2c", bin2c_cmdbuilder.str().c_str());

	return 0;
}
