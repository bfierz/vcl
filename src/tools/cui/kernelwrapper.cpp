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
#include "kernelwrapper.h"

// C++ standard library
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <vector>

// libclang
#include <clang-c/Index.h>

// Templating library
#include "3rdparty/mustache.hpp"

enum class CudaFunctionType
{
	DeviceFunction = 1,
	HostFunction = 2,
	GlobalFunction = 4
};

static std::string readDisplayName(CXCursor cursor)
{
	auto name = clang_getCursorDisplayName(cursor);
	std::string result = clang_getCString(name);
	clang_disposeString(name);

	return result;
}

static std::string readCursorName(CXCursor cursor)
{
	auto name = clang_getCursorSpelling(cursor);
	std::string result = clang_getCString(name);
	clang_disposeString(name);

	return result;
}

static std::string readTypeName(CXType type)
{
	auto name = clang_getTypeSpelling(type);
	std::string result = clang_getCString(name);
	clang_disposeString(name);

	return result;
}

class FunctionDeclaration
{
public:
	FunctionDeclaration(CXCursor c, const std::string& ns)
	: _cursor(c)
	, _cudaType(CudaFunctionType::DeviceFunction)
	{
		// Store the type of the cursor
		_kind = clang_getCursorKind(_cursor);

		// Fetch the type
		_type = clang_getCursorType(_cursor);

		switch (_type.kind)
		{
		case CXType_Invalid:
		{
			break;
		}
		case CXType_Record:
		{
			break;
		}
		case CXType_Unexposed:
		{
			break;
		}
		}

		auto mangledName = clang_getCursorUSR(_cursor);
		_mangledName = clang_getCString(mangledName);
		clang_disposeString(mangledName);

		_name = readCursorName(_cursor);
	}

	const CXCursor& cursor() const
	{
		return _cursor;
	}

	const CXType& type() const
	{
		return _type;
	}

	const std::string& mangledName() const
	{
		return _mangledName;
	}

	const std::string& name() const
	{
		return _name;
	}

	void setCudaType(CudaFunctionType type)
	{
		_cudaType = type;
	}

	void addParameter(Parameter param)
	{
		_parameters.emplace_back(std::move(param));
	}

	bool isGlobalFunction() const
	{
		return (static_cast<int>(_cudaType) & static_cast<int>(CudaFunctionType::GlobalFunction)) != 0;
	}

	std::vector<Parameter> parameters() const { return _parameters; }

private:
	/// Human readable type name
	std::string _name;

	/// Mangled name
	std::string _mangledName;

	/// CUDA function type
	CudaFunctionType _cudaType;

	/// clang type declaration
	CXType _type;

	/// clang declaration cursor
	CXCursor _cursor;

	/// Conviently store the kind of the cursor
	CXCursorKind _kind;

	/// Parameters of the function
	std::vector<Parameter> _parameters;
};

struct TraversalCtx
{
	std::vector<CXCursor> Namespace;
	std::vector<std::unique_ptr<FunctionDeclaration>> Functions;
};

static std::string toString(const std::vector<CXCursor>& ns)
{
	std::stringstream ss;

	for (const auto& c : ns)
	{
		auto name = clang_getCursorSpelling(c);
		ss << "::" << clang_getCString(name);
		clang_disposeString(name);
	}

	return ss.str();
}

static enum CXChildVisitResult parseFunction(CXCursor cursor, CXCursor parent, CXClientData client_data)
{
	auto func_decl = reinterpret_cast<FunctionDeclaration*>(client_data);

	CXCursorKind kind = clang_getCursorKind(cursor);
	switch (kind)
	{
	case CXCursor_AnnotateAttr:
		return CXChildVisit_Recurse;
	case CXCursor_ParmDecl:
	{
		auto type = clang_getCursorType(cursor);

		Parameter param;
		param.Name = readDisplayName(cursor);
		param.TypeName = readTypeName(type);
		param.Alignment = clang_Type_getAlignOf(type);
		param.Size = clang_Type_getSizeOf(type);
		param.IsConst = clang_isConstQualifiedType(type) != 0;
		param.IsRestricted = clang_isRestrictQualifiedType(type) != 0;
		param.IsPointer = type.kind == CXType_Pointer;

		func_decl->addParameter(std::move(param));

		return CXChildVisit_Recurse;
	}
	case CXCursor_CUDADeviceAttr:
	{
		func_decl->setCudaType(CudaFunctionType::DeviceFunction);
		break;
	}
	case CXCursor_CUDAHostAttr:
	{
		func_decl->setCudaType(CudaFunctionType::HostFunction);
		break;
	}
	case CXCursor_CUDAGlobalAttr:
	{
		func_decl->setCudaType(CudaFunctionType::GlobalFunction);
		break;
	}
	}
	return CXChildVisit_Continue;
}

static enum CXChildVisitResult gather(CXCursor cursor, CXCursor parent, CXClientData client_data)
{
	auto ctx = reinterpret_cast<TraversalCtx*>(client_data);
	auto functions = &ctx->Functions;

	CXCursorKind kind = clang_getCursorKind(cursor);
	switch (kind)
	{
	case CXCursor_Namespace:
		ctx->Namespace.push_back(cursor);
		clang_visitChildren(cursor, gather, client_data);
		ctx->Namespace.pop_back();
		break;
	case CXCursor_FunctionDecl:

		functions->emplace_back(std::make_unique<FunctionDeclaration>(cursor, toString(ctx->Namespace)));
		break;
	}

	return CXChildVisit_Recurse;
}

std::vector<Kernel> parseCudaKernels(std::string cuda_toolkit_root, const std::vector<std::string>& params)
{
	// Check minimal version
	auto version = clang_getClangVersion();
	clang_disposeString(version);

	std::vector<const char*> parser_params = {
		"-nocudainc", "-nocudalib",
		"-D__CUDA_LIBDEVICE__",       // Required to have the location specifiers resolved correctly on Windows
		"-include", "cuda_runtime.h", // Default include of CUDA applications
		"-I"                          // Include command for CUDA runtime
	};
	cuda_toolkit_root.append("\\include");
	parser_params.emplace_back(cuda_toolkit_root.c_str());
	std::transform(params.begin(), params.end(), std::back_inserter(parser_params), [](const std::string& param) { return param.c_str(); });

	// Create an index of parsed translation units
	CXIndex index = clang_createIndex(0, 1);

	// Parse a single translation unit
	CXTranslationUnit translation_unit = clang_parseTranslationUnit(index, nullptr, parser_params.data(), static_cast<int>(parser_params.size()), 0, 0, CXTranslationUnit_DetailedPreprocessingRecord | CXTranslationUnit_Incomplete | CXTranslationUnit_SkipFunctionBodies | CXTranslationUnit_KeepGoing | CXTranslationUnit_IncludeAttributedTypes | CXTranslationUnit_VisitImplicitAttributes);

	// Traverse AST and collect types
	TraversalCtx type_ctx;
	clang_visitChildren(clang_getTranslationUnitCursor(translation_unit), gather, &type_ctx);

	// Parse the individual function and collect kernels
	std::vector<Kernel> kernels;
	for (auto& func : type_ctx.Functions)
	{
		clang_visitChildren(func->cursor(), parseFunction, func.get());
		if (func->isGlobalFunction())
		{
			kernels.emplace_back(func->name(), func->parameters());
		}
	}

	clang_disposeTranslationUnit(translation_unit);
	clang_disposeIndex(index);

	return kernels;
}

constexpr char module_template[] = R"(
#include <cuda.h>
#include <memory>
#include <string.h>
#include <tuple>
#include <cuda_runtime.h>
static unsigned char fatbin_data[] = {
	{{module_binary}}
};

static CUmodule loadModule(const void* data)
{
	static const auto module = [](const void* data)
	{
		CUmodule mod = 0;
		CUresult err = cuModuleLoadData(&mod, data);
		std::unique_ptr<struct CUmod_st, CUresult(*)(CUmodule)> module_ptr(mod, cuModuleUnload);
		return module_ptr;
	}(data);

	return module.get();
}
static CUfunction loadFunction(CUmodule mod, const char* name)
{
	CUfunction func;
	cuModuleGetFunction(&func, mod, name);
	return func;
})";

constexpr char function_template[] = R"(
CUresult {{kernel_name}}(dim3 gridDim, dim3 blockDim, unsigned int dynamicSharedMemory, CUstream stream{{#func_params}}, {{{func_param_type}}} {{{func_param_name}}}{{/func_params}})
{
	static const CUmodule module = loadModule(fatbin_data);
	static const CUfunction func = loadFunction(module, "{{kernel_name}}");
	
	unsigned char param_buffer[{{cu_param_set_size}}];
	long long param_buffer_size = sizeof(param_buffer);
	{{#cu_params}}
	memcpy(param_buffer + {{{cu_param_offset}}}, &{{{cu_param_name}}}, {{{cu_param_size}}});
	{{/cu_params}}
	void* params[] =
	{
		CU_LAUNCH_PARAM_BUFFER_POINTER, param_buffer,
		CU_LAUNCH_PARAM_BUFFER_SIZE,    &param_buffer_size,
		CU_LAUNCH_PARAM_END
	};
	return cuLaunchKernel(func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, dynamicSharedMemory, stream, nullptr, params);
}
)";

void createWrappers(std::ofstream& ofs, const std::vector<Kernel>& kernels, const std::string& module)
{
	using namespace kainjow::mustache;

	mustache mod{ module_template };

	std::stringstream ss;
	int cnt = 0;
	for (const auto byte : module)
	{
		ss << "0x" << std::setfill('0') << std::hex << std::setw(2) << (unsigned int)(unsigned char)byte << ", ";
		if (++cnt % 16 == 0)
			ss << "\n\t";
	}
	ofs << mod.render({ "module_binary", ss.str().c_str() });

	for (auto& kernel : kernels)
	{
		mustache func{ function_template };
		data func_data;
		func_data.set("kernel_name", kernel.Name);
		data func_params{ data::type::list };
		for (const auto& param : kernel.Parameters)
		{
			data d;
			d.set("func_param_type", param.TypeName);
			d.set("func_param_name", param.Name);
			func_params << d;
		}
		func_data.set("func_params", func_params);

#define ALIGN_UP(offset, alignment) (offset) = ((offset) + (alignment)-1) & ~((alignment)-1)

		// Compute the size of the kernel parameter buffer
		long long param_set_size = 0;

		data cu_params{ data::type::list };
		for (auto& param : kernel.Parameters)
		{
			// Update alignment for current parameter
			param_set_size = ALIGN_UP(param_set_size, param.Alignment);

			// Write the param to the transfer buffer
			data d;
			d.set("cu_param_offset", std::to_string(param_set_size));
			d.set("cu_param_name", param.Name);
			d.set("cu_param_size", std::to_string(param.Size));
			cu_params << d;

			// Increment to next parameter
			param_set_size += param.Size;
		}
		func_data.set("cu_param_set_size", std::to_string(param_set_size));
		func_data.set("cu_params", cu_params);

		ofs << func.render(func_data);
	}
}
