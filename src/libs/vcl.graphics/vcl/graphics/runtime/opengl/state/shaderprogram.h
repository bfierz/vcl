/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2015 Basil Fierz
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
#pragma once

// VCL library
#include <vcl/config/global.h>
#include <vcl/config/opengl.h>

// C++ standard libary
#include <memory>
#include <vector>

// VCL
#include <vcl/graphics/runtime/opengl/resource/resource.h>
#include <vcl/graphics/runtime/opengl/resource/shader.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	struct ShaderProgramDescription
	{
		//! Vertex shader
		Runtime::OpenGL::Shader* VertexShader{ nullptr };

		//! Tessellation Control shader
		Runtime::OpenGL::Shader* TessControlShader{ nullptr };

		//! Tessellation Evaluation shader
		Runtime::OpenGL::Shader* TessEvalShader{ nullptr };

		//! Geometry shader
		Runtime::OpenGL::Shader* GeometryShader{ nullptr };

		//! Fragment shader
		Runtime::OpenGL::Shader* FragmentShader{ nullptr };

		//! Compute shader
		Runtime::OpenGL::Shader* ComputeShader{ nullptr };
	};

	enum class ProgramResourceType
	{
		Float,
		Float2,
		Float3,
		Float4,
		Double,
		Double2,
		Double3,
		Double4,
		Int,
		Int2,
		Int3,
		Int4,
		UnsignedInt,
		UnsignedInt2,
		UnsignedInt3,
		UnsignedInt4,
		Bool,
		Bool2,
		Bool3,
		Bool4,
		FloatMatrix2,
		FloatMatrix2x3,
		FloatMatrix2x4,
		FloatMatrix3,
		FloatMatrix3x2,
		FloatMatrix3x4,
		FloatMatrix4,
		FloatMatrix4x2,
		FloatMatrix4x3,
		DoubleMatrix2,
		DoubleMatrix2x3,
		DoubleMatrix2x4,
		DoubleMatrix3,
		DoubleMatrix3x2,
		DoubleMatrix3x4,
		DoubleMatrix4,
		DoubleMatrix4x2,
		DoubleMatrix4x3,

		Sampler1D,
		Sampler1DArray,
		Sampler1DArrayShadow,
		Sampler1DShadow,
		Sampler2D,
		Sampler2DArray,
		Sampler2DArrayShadow,
		Sampler2DMultisample,
		Sampler2DRect,
		Sampler2DRectShadow,
		Sampler2DShadow,
		Sampler3D,
		SamplerBuffer,
		SamplerCube,
		SamplerCubeArray,
		SamplerCubeArrayShadow,
		SamplerCubeShadow,
		IntSampler1D,
		IntSampler1DArray,
		IntSampler2D,
		IntSampler2DArray,
		IntSampler2DMultisample,
		IntSampler2DRect,
		IntSampler3D,
		IntSamplerBuffer,
		IntSamplerCube,
		IntSamplerCubeArray,
		UnsignedIntSampler1D,
		UnsignedIntSampler1DArray,
		UnsignedIntSampler2D,
		UnsignedIntSampler2DArray,
		UnsignedIntSampler2DMultisample,
		UnsignedIntSampler2DRect,
		UnsignedIntSampler3D,
		UnsignedIntSamplerBuffer,
		UnsignedIntSamplerCube,
		UnsignedIntSamplerCubeArray,

		Invalid = -1
	};

	struct AttributeData
	{
		int Location;
		std::string Name;
		ProgramResourceType Type;
	};

	struct ProgramOutputData
	{
		int Location;
		std::string Name;
		ProgramResourceType Type;
	};

	struct UniformData
	{
		int Location;
		std::string Name;
		ProgramResourceType Type;
		int ArraySize;
	};

	class ProgramResources
	{
	public:
		ProgramResources(GLuint program);

	public:
		const std::vector<AttributeData>& attributes() const { return _attributes; }
		const std::vector<UniformData>& uniforms() const { return _uniforms; }
		
	public:
		static GLenum toGLenum(ProgramResourceType t);
		static ProgramResourceType toResourceType(GLenum type);
		static const char* name(GLenum type);

	private:
		std::vector<AttributeData> _attributes;
		std::vector<ProgramOutputData> _outputs;
		std::vector<UniformData> _uniforms;
	};

	class ShaderProgram : public Resource
	{
	public:
		ShaderProgram(const ShaderProgramDescription& desc);

	public:
		void bind();

	//public:
	//	void setUniform(const std::string& name, float value);
	//	void setUniform(const std::string& name, const Eigen::Vector2f& value);
	//	void setUniform(const std::string& name, const Eigen::Vector3f& value);
	//	void setUniform(const std::string& name, const Eigen::Vector4f& value);
	//	void setUniform(const std::string& name, int value);
	//	void setUniform(const std::string& name, const Eigen::Vector2i& value);
	//	void setUniform(const std::string& name, const Eigen::Vector3i& value);
	//	void setUniform(const std::string& name, const Eigen::Vector4i& value);
	//	void setUniform(const std::string& name, unsigned int value);
	//	void setUniform(const std::string& name, const Eigen::Vector2ui& value);
	//	void setUniform(const std::string& name, const Eigen::Vector3ui& value);
	//	void setUniform(const std::string& name, const Eigen::Vector4ui& value);
	//	void setUniform(const std::string& name, const Eigen::Matrix4f& value);
	//
	//public:
	//	void setUniform(const UniformHandle& handle, float value);
	//	void setUniform(const UniformHandle& handle, const Eigen::Vector2f& value);
	//	void setUniform(const UniformHandle& handle, const Eigen::Vector3f& value);
	//	void setUniform(const UniformHandle& handle, const Eigen::Vector4f& value);
	//	void setUniform(const UniformHandle& handle, int value);
	//	void setUniform(const UniformHandle& handle, const Eigen::Vector2i& value);
	//	void setUniform(const UniformHandle& handle, const Eigen::Vector3i& value);
	//	void setUniform(const UniformHandle& handle, const Eigen::Vector4i& value);
	//	void setUniform(const UniformHandle& handle, unsigned int value);
	//	void setUniform(const UniformHandle& handle, const Eigen::Vector2ui& value);
	//	void setUniform(const UniformHandle& handle, const Eigen::Vector3ui& value);
	//	void setUniform(const UniformHandle& handle, const Eigen::Vector4ui& value);
	//	void setUniform(const UniformHandle& handle, const Eigen::Matrix4f& value);

	private:
		void printInfoLog() const;

	private:
		//! Uniforms and resources
		std::unique_ptr<ProgramResources> _resources;
	};
}}}}
#endif // VCL_OPENGL_SUPPORT

