/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2018 Basil Fierz
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
class CgContextUpdateMock : public Vcl::Mathematics::Solver::Cuda::ConjugateGradientsContext
{
public:
	CgContextUpdateMock(Vcl::ref_ptr<Vcl::Compute::Context> ctx, Vcl::ref_ptr<Vcl::Compute::CommandQueue> queue, int size)
	: ConjugateGradientsContext(ctx, queue, size)
	{
		_devX = Vcl::static_pointer_cast<Vcl::Compute::Cuda::Buffer>(_ownerCtx->createBuffer(Vcl::Compute::BufferAccess::None, size * sizeof(float)));
	}

	virtual ~CgContextUpdateMock()
	{
		_ownerCtx->release(_devX);
	}

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

		_input_x.clear();
		_input_r.clear();
		_input_d.clear();
		_input_q.clear();
		_input_x.reserve(size());
		_input_r.reserve(size());
		_input_d.reserve(size());
		_input_q.reserve(size());

		std::generate_n(std::back_inserter(_input_x), size(), [&twister, &values]() { return values(twister); });
		std::generate_n(std::back_inserter(_input_r), size(), [&twister, &values]() { return values(twister); });
		std::generate_n(std::back_inserter(_input_d), size(), [&twister, &values]() { return values(twister); });
		std::generate_n(std::back_inserter(_input_q), size(), [&twister, &values]() { return values(twister); });

		_queue->write(_devX, _input_x.data(), true);
		_queue->write(_devResidual, _input_r.data(), true);
		_queue->write(_devDirection, _input_d.data(), true);
		_queue->write(_devQ, _input_q.data(), true);

		*_hostR = values(twister);
		*_hostG = values(twister);
		*_hostA = values(twister);
		*_hostB = values(twister);

		_queue->write(Vcl::Compute::BufferView(_reduceBuffersR[0], 0, sizeof(float)), _hostR, true);
		_queue->write(Vcl::Compute::BufferView(_reduceBuffersG[0], 0, sizeof(float)), _hostG, true);
		_queue->write(Vcl::Compute::BufferView(_reduceBuffersA[0], 0, sizeof(float)), _hostA, true);
		_queue->write(Vcl::Compute::BufferView(_reduceBuffersB[0], 0, sizeof(float)), _hostB, true);
	}

	void computeReference()
	{
		const float d_r = *_hostR;
		const float d_g = *_hostG;
		const float d_a = *_hostA;
		const float d_b = *_hostB;

		float alpha = 0.0f;
		if (fabs(d_g) > 0.0f)
			alpha = d_r / d_g;

		float beta = d_r - 2.0f * alpha * d_b + alpha * alpha * d_a;
		if (fabs(d_r) > 0.0f)
			beta = beta / d_r;

		Eigen::Map<Eigen::VectorXf> x(_input_x.data(), _input_x.size());
		Eigen::Map<Eigen::VectorXf> r(_input_r.data(), _input_r.size());
		Eigen::Map<Eigen::VectorXf> d(_input_d.data(), _input_d.size());
		Eigen::Map<Eigen::VectorXf> q(_input_q.data(), _input_q.size());

		x += alpha * d;
		r -= alpha * q;
		d = r + beta * d;
	}

	void testResults()
	{
		using Vcl::Mathematics::equal;

		std::vector<float> x(size());
		std::vector<float> r(size());
		std::vector<float> d(size());
		std::vector<float> q(size());
		_queue->read(x.data(), Vcl::Compute::BufferView(_devX));
		_queue->read(r.data(), Vcl::Compute::BufferView(_devResidual));
		_queue->read(d.data(), Vcl::Compute::BufferView(_devDirection));
		_queue->read(q.data(), Vcl::Compute::BufferView(_devQ));
		_queue->sync();

		computeReference();

		compare("x", x, _input_x);
		compare("r", r, _input_r);
		compare("d", d, _input_d);
		compare("q", q, _input_q);
	}

	void compare(const char* marker, const std::vector<float>& sol, const std::vector<float>& ref)
	{
		// Check for the solution
		Eigen::VectorXf err;
		err.setZero(sol.size());
		for (size_t i = 0; i < sol.size(); i++)
		{
			err(i) = fabs(ref[i] - sol[i]);
		}

		Eigen::VectorXf::Index max_err_idx;
		EXPECT_LE(err.maxCoeff(&max_err_idx), 1e-4f) << "Maximum error for " << marker << " at index " << max_err_idx;
	}

private:
	std::vector<float> _input_x;
	std::vector<float> _input_r;
	std::vector<float> _input_d;
	std::vector<float> _input_q;
};

extern Vcl::owner_ptr<Vcl::Compute::Cuda::Context> default_ctx;

TEST(Cuda, CgUpdateVectors)
{
	using namespace Vcl::Compute::Cuda;

	const size_t problem_size = 400000;

	auto queue = default_ctx->defaultQueue();

	CgContextUpdateMock cg_ctx(default_ctx, queue, problem_size);
	cg_ctx.setupProblem();
	cg_ctx.updateVectors();
	cg_ctx.testResults();
}
