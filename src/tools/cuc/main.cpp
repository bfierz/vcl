/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014 Basil Fierz
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
#include <sstream>
#include <vector>

VCL_BEGIN_EXTERNAL_HEADERS
// CxxOpts
#include <vcl/core/3rdparty/cxxopts.hpp>
VCL_END_EXTERNAL_HEADERS

// Windows API
#ifdef VCL_ABI_WINAPI
#	include <Windows.h>
#else
VCL_ERROR("No compatible process API found.")
#endif

namespace Vcl { namespace Tools { namespace Cuc {
	void displayError(LPCTSTR errorDesc, DWORD errorCode)
	{
		TCHAR errorMessage[1024] = TEXT("");

		DWORD flags = FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS | FORMAT_MESSAGE_MAX_WIDTH_MASK;

		FormatMessage(flags, nullptr, errorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), errorMessage, sizeof(errorMessage) / sizeof(TCHAR), nullptr);

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
	using namespace Vcl::Tools::Cuc;

#ifdef VCL_ABI_WINAPI
#	if VCL_HAS_STDCXX17
	namespace fs = std::filesystem;
#	else
	namespace fs = std::experimental::filesystem;
#	endif
#elif defined(VCL_ABI_POSIX)
	namespace fs = boost::filesystem;
#endif

	cxxopts::Options options(argv[0], "cuc - command line options");

	try
	{
		// clang-format off
		options.add_options()
			("help", "Print this help information on this tool.")
			("version", "Print version information on this tool.")
			("nvcc", "Specify which nvcc should be used.", cxxopts::value<std::string>())
			("m64", "Specify that this should be compiled in 64bit.")
			("profile", "Target compute architectures (sm_30, sm_35, sm_50, compute_30, compute_35, compute_50)", cxxopts::value<std::vector<std::string>>())
			("I,include", "Additional include directory", cxxopts::value<std::vector<std::string>>())
			("L,library-path", "Additional library directory (passed to nvcc)", cxxopts::value<std::vector<std::string>>())
			("l,library", "Additional library (passed to nvcc)", cxxopts::value<std::vector<std::string>>())
			("symbol", "Name of the symbol used for the compiled module", cxxopts::value<std::string>())
			("o,output-file", "Specify the output file.", cxxopts::value<std::string>())
			("input-file", "Specify the input file.", cxxopts::value<std::string>())
			;
		// clang-format on
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

		if (parsed_options.count("symbol") == 0 || parsed_options.count("input-file") == 0 || parsed_options.count("output-file") == 0)
		{
			std::cout << options.help({ "" }) << std::endl;
			return -1;
		}

		std::vector<std::string> profiles;
		if (parsed_options.count("profile"))
		{
			profiles = parsed_options["profile"].as<std::vector<std::string>>();
		}

		// Construct the base name for the intermediate files
		std::string tmp_file_base = fs::path(parsed_options["input-file"].as<std::string>()).stem().string();

		// Add the address
		if (parsed_options.count("m64"))
		{
			tmp_file_base += "_m64";
		} else
		{
			tmp_file_base += "_m32";
		}

		// Invoke the cuda compiler for each profile
		std::vector<std::pair<std::string, std::string>> compiled_files;
		compiled_files.reserve(profiles.size());
		for (auto& p : profiles)
		{
			std::stringstream cmd_compile;
			std::stringstream cmd_link;

			// Verbose compiler output
			//cmd_compile << R"(--verbose )";

			// Force a compiler version
			//cmd_compile << R"(--use-local-env --cl-version 2013 )";

			if (parsed_options.count("include"))
			{
				for (auto& inc : parsed_options["include"].as<std::vector<std::string>>())
				{
					cmd_compile << "-I \"" << inc << "\" ";
				}
			}

			cmd_compile << "-gencode=arch=";
			cmd_link << "-arch ";

			// Generate the output filename for intermediate file
			std::string tmp_file = tmp_file_base + "_";
			std::string cc_file = tmp_file_base + "_compiled_";

			auto sm = p.find("sm");
			if (sm != p.npos)
			{
				cmd_compile << "compute" << p.substr(2, p.npos) << ",code=" << p;
				cmd_compile << " -cubin ";
				tmp_file += p + ".cubin";

				cmd_link << p << " ";
				cc_file += p + ".cubin";
			} else
			{
				cmd_compile << p << ",code=" << p;
				cmd_compile << " -ptx ";
				tmp_file += p + ".ptx";

				cmd_link << p << " ";
				cc_file += p + ".ptx";
			}

			// Link against CUDA libraries
			if (parsed_options.count("library-path"))
			{
				for (auto& path : parsed_options["library-path"].as<std::vector<std::string>>())
				{
					cmd_link << "-L\"" << path << "\" ";
				}
			}
			if (parsed_options.count("library"))
			{
				for (auto& lib : parsed_options["library"].as<std::vector<std::string>>())
				{
					cmd_link << "-l\"" << lib << "\" ";
				}
			}
			cmd_compile << "-dc -rdc=true ";

			compiled_files.emplace_back(p, tmp_file);
			cmd_compile << "-o \"" << cc_file << "\" \"" << parsed_options["input-file"].as<std::string>() << "\"";
			cmd_link << "-o \"" << tmp_file << "\" \"" << cc_file << "\"";

			// Compile the code
			exec((nvcc_bin_path / "nvcc.exe").string().c_str(), cmd_compile.str().c_str());

			// Link the code
			exec((nvcc_bin_path / "nvlink.exe").string().c_str(), cmd_link.str().c_str());
		}

		// Create a fat binary from the compiled files
		std::stringstream fatbin_cmdbuilder;

		// Create a new fatbin
		fatbin_cmdbuilder << R"(--create=")" << tmp_file_base << R"(.fatbin" )";

		// We want to create an embedded file
		//fatbin_cmdbuilder << R"(--embedded-fatbin=")" << tmp_file_base << R"(.fatbin.c" )";

		// Set the bitness
		if (parsed_options.count("m64"))
		{
			fatbin_cmdbuilder << "-64 ";
		} else
		{
			fatbin_cmdbuilder << "-32 ";
		}

		// We are compiling cuda (not supported since 10.1), doesn't seem to affect the output
		//fatbin_cmdbuilder << R"(--cuda )";

		// Add the orignal filename as identifier
		fatbin_cmdbuilder << R"(--ident=")" << parsed_options["symbol"].as<std::string>() << R"(" )";

		// Add all the created files
		for (auto& profile_file : compiled_files)
		{
			fatbin_cmdbuilder << R"("--image=profile=)" << profile_file.first << R"(,file=)" << profile_file.second << R"(" )";
		}

		exec((nvcc_bin_path / "fatbinary.exe").string().c_str(), fatbin_cmdbuilder.str().c_str());

		// Create a source file with the binary
		std::stringstream bin2c_cmdbuilder;
		bin2c_cmdbuilder.str("");
		bin2c_cmdbuilder.clear();
		bin2c_cmdbuilder << "--group 4 ";

		if (parsed_options.count("symbol"))
		{
			bin2c_cmdbuilder << "--symbol " << parsed_options["symbol"].as<std::string>() << " ";
		}

		bin2c_cmdbuilder << "-o " << parsed_options["output-file"].as<std::string>() << " ";
		bin2c_cmdbuilder << tmp_file_base << R"(.fatbin)";

		// Invoke the binary file translator
		exec("bin2c", bin2c_cmdbuilder.str().c_str());
	} catch (const cxxopts::OptionException& e)
	{
		std::cout << "Error parsing options: " << e.what() << std::endl;
		return 1;
	}
	return 0;
}
