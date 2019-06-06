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
#include <memory>
#include <sstream>
#include <vector>

// libclang
#include <clang-c/Index.h>

enum class CudaFunctionType
{
	DeviceFunction = 1,
	HostFunction = 2,
	GlobalFunction = 4
};

class FunctionDeclaration
{
public:
	FunctionDeclaration(CXCursor c, const std::string& ns)
		: _cursor(c)
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

		auto name = clang_getCursorSpelling(_cursor);
		_name = ns + "::" + clang_getCString(name);
		clang_disposeString(name);
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

	bool isGlobalFunction() const
	{
		return (static_cast<int>(_cudaType) & static_cast<int>(CudaFunctionType::GlobalFunction)) != 0;
	}

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
		auto _name = clang_getCString(clang_getCursorDisplayName(cursor));
		auto _type = clang_getCursorType(cursor);
		auto _type_name = clang_getCString(clang_getTypeSpelling(_type));

		// Determine qualifiers
		bool _is_const = clang_isConstQualifiedType(_type);
		bool _is_restricted = clang_isRestrictQualifiedType(_type);
		bool _is_volatile = clang_isVolatileQualifiedType(_type);

		bool _is_ptr = _type.kind == CXType_Pointer;
		bool _is_lref = _type.kind == CXType_LValueReference;
		bool _is_rref = _type.kind == CXType_RValueReference;

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
	CXChildVisitResult result = CXChildVisit_Continue;

	auto ctx = reinterpret_cast<TraversalCtx*>(client_data);
	auto functions = &ctx->Functions;

	CXCursorKind kind = clang_getCursorKind(cursor);
	switch (kind)
	{
	case CXCursor_Namespace:
		ctx->Namespace.push_back(cursor);
		clang_visitChildren(cursor, gather, client_data);
		ctx->Namespace.pop_back();
		result = CXChildVisit_Continue;
		break;
	case CXCursor_FunctionDecl:

		functions->emplace_back(std::make_unique<FunctionDeclaration>(cursor, toString(ctx->Namespace)));
		result = CXChildVisit_Continue;
		break;
	}

	return result;
}

std::string generateKernelCallWrappers(std::string cuda_toolkit_root, const std::vector<std::string>& params)
{
	// Check minimal version
	auto version = clang_getClangVersion();
	clang_disposeString(version);

	std::vector<const char*> parser_params = {
		"-nocudainc", "-nocudalib", "-D__CUDA_LIBDEVICE__", "-include", "cuda_runtime.h", "-I"
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

	// Parse the individual types
	for (auto& func : type_ctx.Functions)
	{
		clang_visitChildren(func->cursor(), parseFunction, func.get());
	}

	clang_disposeTranslationUnit(translation_unit);
	clang_disposeIndex(index);

	return {};
}

// "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include" 
