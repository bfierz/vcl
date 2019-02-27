#include <vcl/math/opencl/jacobisvd33_mcadams.h>

// C++ standard library
#include <array>
#include <iostream>

// VCL
#include <vcl/compute/opencl/buffer.h>
#include <vcl/compute/opencl/kernel.h>
#include <vcl/compute/opencl/module.h>
#include <vcl/math/ceil.h>

// Kernels
extern uint32_t JacobiSVD33McAdamsCL[];
extern size_t JacobiSVD33McAdamsCLSize;

namespace Vcl { namespace Mathematics { namespace OpenCL
{
	JacobiSVD33::JacobiSVD33(Core::ref_ptr<Compute::Context> ctx)
	: _ownerCtx(ctx)
	{
		VclRequire(ctx, "Context is valid.");

		_A = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 16 * 9 * sizeof(float));
		_U = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 16 * 9 * sizeof(float));
		_V = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 16 * 9 * sizeof(float));
		_S = ctx->createBuffer(Vcl::Compute::BufferAccess::ReadWrite, 16 * 3 * sizeof(float));
		
		// Load the module
		_svdModule = ctx->createModuleFromSource(reinterpret_cast<const int8_t*>(JacobiSVD33McAdamsCL), JacobiSVD33McAdamsCLSize * sizeof(uint32_t));

		if (_svdModule)
		{
			// Load the jacobi SVD kernel
			_svdKernel = _svdModule->kernel("JacobiSVD33McAdams");
		}

		VclEnsure(implies(_svdModule, _svdKernel), "SVD kernel is valid.");
	}

	void JacobiSVD33::operator()
	(
		Vcl::Compute::CommandQueue& queue,
		const Vcl::Core::InterleavedArray<float, 3, 3, -1>& inA,
		Vcl::Core::InterleavedArray<float, 3, 3, -1>& outU,
		Vcl::Core::InterleavedArray<float, 3, 3, -1>& outV,
		Vcl::Core::InterleavedArray<float, 3, 1, -1>& outS
	)
	{
		VclRequire(dynamic_cast<Compute::OpenCL::CommandQueue*>(&queue), "Commandqueue is CUDA command queue.");
		VclRequire(_svdKernel, "SVD kernel is loaded.");
		
		auto A = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_A);
		auto U = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_U);
		auto V = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_V);
		auto S = Vcl::Core::dynamic_pointer_cast<Compute::OpenCL::Buffer>(_S);
		
		const size_t capacity = inA.capacity();
		const size_t size = inA.size();
		if (_capacity < capacity)
		{
			_capacity = capacity;
			A->resize(_capacity * 9 * sizeof(float));
			U->resize(_capacity * 9 * sizeof(float));
			V->resize(_capacity * 9 * sizeof(float));
			S->resize(_capacity * 3 * sizeof(float));
		}

		queue.write(A, inA.data());

		// Perform the SVD computation
		std::array<size_t, 3> grid = { ceil<128>(size), 0, 0 };
		std::array<size_t, 3> block = { 128, 0, 0 };
		static_cast<Compute::OpenCL::Kernel*>(_svdKernel.get())->run
		(
			static_cast<Compute::OpenCL::CommandQueue&>(queue),
			1,
			grid,
			block,
			(int) size,
			(int) _capacity,
			(cl_mem) *A,
			(cl_mem) *U,
			(cl_mem) *V,
			(cl_mem) *S
		);

		queue.read(outU.data(), U);
		queue.read(outV.data(), V);
		queue.read(outS.data(), S);
	}
}}}
