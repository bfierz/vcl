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
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

// libclang
#include <clang-c/Index.h>

// Test call
// D:\DevTools\LLVM-8.0.0\bin\clang -Xclang -ast-dump -fsyntax-only -nocudainc -nocudalib -include cuda_runtime.h -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include" -I "D:/projects/vcl/src/libs/vcl.math.cuda" -I "D:/projects/vcl/src/libs/vcl.core.cuda" -I "D:/projects/vcl/src/libs/vcl.math" -D__CUDA_LIBDEVICE__ .\jacobisvd33_mcadams.cu > ast2.txt

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
		param.IsConst = clang_isConstQualifiedType(type);
		param.IsRestricted = clang_isRestrictQualifiedType(type);
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

std::vector<Kernel> generateKernelCallWrappers(std::string cuda_toolkit_root, const std::vector<std::string>& params)
{
	// Check minimal version
	auto version = clang_getClangVersion();
	clang_disposeString(version);

	std::vector<const char*> parser_params = {
		"-nocudainc", "-nocudalib",
		"-D__CUDA_LIBDEVICE__", // Required to have the location specifiers resolved correctly on Windows
		"-include", "cuda_runtime.h", // Default include of CUDA applications
		"-I" // Include command for CUDA runtime
	};
	cuda_toolkit_root.append("\\include");
	parser_params.emplace_back(cuda_toolkit_root.c_str());
	std::transform(params.begin(), params.end(), std::back_inserter(parser_params), [](const std::string& param) { return param.c_str(); });

	// Create an index of parsed translation units
	CXIndex index = clang_createIndex(0, 1);

	// Parse a single translation unit
	CXTranslationUnit translation_unit = clang_parseTranslationUnit(index, 0, parser_params.data(), parser_params.size(), 0, 0,
		CXTranslationUnit_DetailedPreprocessingRecord | CXTranslationUnit_Incomplete | CXTranslationUnit_SkipFunctionBodies | CXTranslationUnit_KeepGoing | CXTranslationUnit_IncludeAttributedTypes | CXTranslationUnit_VisitImplicitAttributes);

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
