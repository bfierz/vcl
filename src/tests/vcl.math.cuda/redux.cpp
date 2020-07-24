/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2017 Basil Fierz
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
#include <random>

// Include the relevant parts from the library
#include <vcl/compute/cuda/buffer.h>
#include <vcl/compute/cuda/commandqueue.h>
#include <vcl/compute/cuda/context.h>
#include <vcl/compute/cuda/device.h>
#include <vcl/compute/cuda/platform.h>
#include <vcl/core/memory/smart_ptr.h>

#include <vcl/math/cuda/conjugategradientscontext.h>
#include <vcl/math/math.h>

// Google test
#include <gtest/gtest.h>

// Mock the CG context
class CgContextMock : public Vcl::Mathematics::Solver::Cuda::ConjugateGradientsContext
{
public:
	using ConjugateGradientsContext::ConjugateGradientsContext;

	void computeInitialResidual() override
	{
	}
			
	void computeQ() override
	{
	}

	void setupProblem()
	{
		// ISO C++ randorm number generator
		std::mt19937 twister;
		std::uniform_real_distribution<float> values;
		
		_input_r.clear();
		_input_d.clear();
		_input_q.clear();
		_input_r.reserve(size());
		_input_d.reserve(size());
		_input_q.reserve(size());

		std::generate_n(std::back_inserter(_input_r), size(), [&twister, &values]() { return values(twister); });
		std::generate_n(std::back_inserter(_input_d), size(), [&twister, &values]() { return values(twister); });
		std::generate_n(std::back_inserter(_input_q), size(), [&twister, &values]() { return values(twister); });

		_queue->write(_devResidual, _input_r.data(), true);
		_queue->write(_devDirection, _input_d.data(), true);
		_queue->write(_devQ, _input_q.data(), true);
	}

	void computeReference(float& l1_r, float& l1_g, float& l1_b, float& l1_a)
	{
		const std::vector<float>& r = _input_r;
		const std::vector<float>& d = _input_d;
		const std::vector<float>& q = _input_q;
		
		l1_r = 0;
		l1_g = 0;
		l1_b = 0;
		l1_a = 0;
		for (size_t i = 0; i < size(); i++)
		{
			l1_r += r[i] * r[i];
			l1_g += d[i] * q[i];
			l1_b += r[i] * q[i];
			l1_a += q[i] * q[i];
		}
	}

	void testResults()
	{
		using Vcl::Mathematics::equal;

		_queue->read(_hostR, Vcl::Compute::BufferView(_reduceBuffersR[0], 0, sizeof(float)));
		_queue->read(_hostG, Vcl::Compute::BufferView(_reduceBuffersG[0], 0, sizeof(float)));
		_queue->read(_hostA, Vcl::Compute::BufferView(_reduceBuffersA[0], 0, sizeof(float)));
		_queue->read(_hostB, Vcl::Compute::BufferView(_reduceBuffersB[0], 0, sizeof(float)));
		_queue->sync();
		
		float l1_r = 0;
		float l1_g = 0;
		float l1_b = 0;
		float l1_a = 0;
		computeReference(l1_r, l1_g, l1_b, l1_a);

		EXPECT_TRUE(equal(*_hostR, l1_r, size()*1e-5f)) << "Ref: " << *_hostR << ", Cuda: " << l1_r;
		EXPECT_TRUE(equal(*_hostG, l1_g, size()*1e-5f)) << "Ref: " << *_hostG << ", Cuda: " << l1_g;
		EXPECT_TRUE(equal(*_hostB, l1_b, size()*1e-5f)) << "Ref: " << *_hostB << ", Cuda: " << l1_b;
		EXPECT_TRUE(equal(*_hostA, l1_a, size()*1e-5f)) << "Ref: " << *_hostA << ", Cuda: " << l1_a;
	}

private:
	std::vector<float> _input_r;
	std::vector<float> _input_d;
	std::vector<float> _input_q;

};

extern Vcl::owner_ptr<Vcl::Compute::Cuda::Context> default_ctx;

TEST(Cuda, Reduction)
{
	using namespace Vcl::Compute::Cuda;

	const size_t problem_size = 400000;

	auto queue = default_ctx->defaultQueue();

	CgContextMock cg_ctx(default_ctx, queue, problem_size);
	cg_ctx.setupProblem();
	cg_ctx.reduceVectors();
	cg_ctx.testResults();
}

