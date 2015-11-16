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
#include <vcl/graphics/runtime/resource/buffer.h>
#include <vcl/graphics/runtime/resource/texture.h>
#include <vcl/graphics/runtime/state/inputlayout.h>
#include <vcl/graphics/runtime/state/sampler.h>

#ifdef VCL_OPENGL_SUPPORT

namespace Vcl { namespace Graphics { namespace Runtime { namespace OpenGL
{
	struct ShaderProgramDescription
	{
		//! Input layout
		InputLayoutDescription InputLayout;

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

		Image1D,
		Image2D,
		Image3D,
		ImageCube,
		Image2DRect,
		Image1DArray,
		Image2DArray,
		ImageCubeArray,
		ImageBuffer,
		Image2DMS,
		Image2DMSArray,
		IntImage1D,
		IntImage2D,
		IntImage3D,
		IntImageCube,
		IntImage2DRect,
		IntImage1DArray,
		IntImage2DArray,
		IntImageCubeArray,
		IntImageBuffer,
		IntImage2DMS,
		IntImage2DMSArray,
		UnsignedIntImage1D,
		UnsignedIntImage2D,
		UnsignedIntImage3D,
		UnsignedIntImageCube,
		UnsignedIntImage2DRect,
		UnsignedIntImage1DArray,
		UnsignedIntImage2DArray,
		UnsignedIntImageCubeArray,
		UnsignedIntImageBuffer,
		UnsignedIntImage2DMS,
		UnsignedIntImage2DMSArray,

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

		//! Location of the resource this uniform points to
		int ResourceLocation;
	};

	struct UniformBlockData
	{
		UniformBlockData(int loc, std::string name, int res_loc) : Location(loc), Name(std::move(name)), ResourceLocation(res_loc) {}

		//! Uniform location
		int Location;

		//! Name of the uniform block
		std::string Name;

		//! Location of the resource this uniform points to
		int ResourceLocation;
	};

	struct BufferBlockData
	{
		BufferBlockData(int loc, std::string name, int res_loc) : Location(loc), Name(std::move(name)), ResourceLocation(res_loc) {}

		//! Uniform location
		int Location;

		//! Name of the uniform block
		std::string Name;

		//! Location of the resource this uniform points to
		int ResourceLocation;
	};

	class ProgramAttributes
	{
	public:
		ProgramAttributes(GLuint program);

	public:
		const std::vector<AttributeData>& elems() const { return _attributes; }

	public:
		std::vector<AttributeData>::iterator begin() { return _attributes.begin(); }
		std::vector<AttributeData>::const_iterator begin() const { return _attributes.cbegin(); }
		std::vector<AttributeData>::const_iterator cbegin() const { return _attributes.cbegin(); }

		std::vector<AttributeData>::iterator end() { return _attributes.end(); }
		std::vector<AttributeData>::const_iterator end() const { return _attributes.cend(); }
		std::vector<AttributeData>::const_iterator cend() const { return _attributes.cend(); }

	private:
		std::vector<AttributeData> _attributes;
	};

	class ProgramOutput
	{
	public:
		ProgramOutput(GLuint program);

	public:
		const std::vector<ProgramOutputData>& elems() const { return _outputs; }

	public:
		std::vector<ProgramOutputData>::iterator begin() { return _outputs.begin(); }
		std::vector<ProgramOutputData>::const_iterator begin() const { return _outputs.cbegin(); }
		std::vector<ProgramOutputData>::const_iterator cbegin() const { return _outputs.cbegin(); }

		std::vector<ProgramOutputData>::iterator end() { return _outputs.end(); }
		std::vector<ProgramOutputData>::const_iterator end() const { return _outputs.cend(); }
		std::vector<ProgramOutputData>::const_iterator cend() const { return _outputs.cend(); }

	private:
		std::vector<ProgramOutputData> _outputs;
	};

	class ProgramUniforms
	{
	public:
		ProgramUniforms(GLuint program);

	public:
		const std::vector<UniformData>& elems() const { return _uniforms; }

	public:
		std::vector<UniformData>::iterator begin() { return _uniforms.begin(); }
		std::vector<UniformData>::const_iterator begin() const { return _uniforms.cbegin(); }
		std::vector<UniformData>::const_iterator cbegin() const { return _uniforms.cbegin(); }

		std::vector<UniformData>::iterator end() { return _uniforms.end(); }
		std::vector<UniformData>::const_iterator end() const { return _uniforms.cend(); }
		std::vector<UniformData>::const_iterator cend() const { return _uniforms.cend(); }

	private:
		std::vector<UniformData> _uniforms;
	};

	class ProgramUniformBlocks
	{
	public:
		ProgramUniformBlocks(GLuint program);

	public:
		const std::vector<UniformBlockData>& elems() const { return _resources; }

	public:
		std::vector<UniformBlockData>::iterator begin() { return _resources.begin(); }
		std::vector<UniformBlockData>::const_iterator begin() const { return _resources.cbegin(); }
		std::vector<UniformBlockData>::const_iterator cbegin() const { return _resources.cbegin(); }

		std::vector<UniformBlockData>::iterator end() { return _resources.end(); }
		std::vector<UniformBlockData>::const_iterator end() const { return _resources.cend(); }
		std::vector<UniformBlockData>::const_iterator cend() const { return _resources.cend(); }

	private:
		std::vector<UniformBlockData> _resources;
	};

	class ProgramBuffers
	{
	public:
		ProgramBuffers(GLuint program);

	public:
		const std::vector<BufferBlockData>& elems() const { return _resources; }

	public:
		std::vector<BufferBlockData>::iterator begin() { return _resources.begin(); }
		std::vector<BufferBlockData>::const_iterator begin() const { return _resources.cbegin(); }
		std::vector<BufferBlockData>::const_iterator cbegin() const { return _resources.cbegin(); }

		std::vector<BufferBlockData>::iterator end() { return _resources.end(); }
		std::vector<BufferBlockData>::const_iterator end() const { return _resources.cend(); }
		std::vector<BufferBlockData>::const_iterator cend() const { return _resources.cend(); }

	private:
		std::vector<BufferBlockData> _resources;
	};

	class ProgramResources
	{
	public:
		ProgramResources(GLuint program);

	public:
		const std::vector<AttributeData>& attributes() const { return _attributes.elems(); }
		const std::vector<UniformData>& uniforms() const { return _uniforms.elems(); }
		const std::vector<UniformBlockData>& uniformBlocks() const { return _uniformBlocks.elems(); }
		
	public:
		static GLenum toGLenum(ProgramResourceType t);
		static ProgramResourceType toResourceType(GLenum type);
		static const char* name(GLenum type);

	private:
		ProgramAttributes _attributes;
		ProgramOutput _outputs;
		ProgramUniforms _uniforms;
		ProgramUniformBlocks _uniformBlocks;
		ProgramBuffers _buffers;
	};

	class ShaderProgram : public Resource
	{
	public:
		ShaderProgram(const ShaderProgramDescription& desc);

	public:
		void bind();

	public:
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
		void setTexture(const char* name, const Runtime::Texture* tex, const Runtime::Sampler* sampler);
		void setImage(const char* name, const Runtime::Texture* img, bool read, bool write);
		void setConstantBuffer(const char* name, const Runtime::Buffer* buf, size_t offset = 0, size_t size = 0);
		void setBuffer(const char* name, const Runtime::Buffer* buf, size_t offset = 0, size_t size = 0);
	
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
		void linkAttributes(const InputLayoutDescription& layout);

	private:
		//! Print the information of the current program state
		void printInfoLog() const;

	private:
		//! Uniforms and resources
		std::unique_ptr<ProgramResources> _resources;
	};
}}}}
#endif // VCL_OPENGL_SUPPORT

