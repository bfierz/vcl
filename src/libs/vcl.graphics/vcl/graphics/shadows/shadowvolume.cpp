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
#include <vcl/graphics/shadows/shadowvolume.h>

VCL_BEGIN_EXTERNAL_HEADERS
// FMT
#include <fmt/format.h>
VCL_END_EXTERNAL_HEADERS

// VCL
#include <vcl/graphics/shadows/shadowmap.h>
#include <vcl/math/math.h>

namespace Vcl { namespace Graphics
{
	ShadowMapVolume::~ShadowMapVolume()
	{
	}

	ShadowMapAlgorithm ShadowMapVolume::shadowMapAlgorithm() const
	{
		return _shadowMapAlgorithm;
	}

	void ShadowMapVolume::setShadowMapAlgorithm(ShadowMapAlgorithm algorithm)
	{
		_shadowMapAlgorithm = algorithm;
	}

	ShadowMap* ShadowMapVolume::shadowMap() const
	{
		return _shadowMap.get();
	}

	void ShadowMapVolume::setShadowMap(std::unique_ptr<ShadowMap> shadow_map)
	{
		_shadowMap = std::move(shadow_map);
	}

	PerspectiveShadowMapVolume::PerspectiveShadowMapVolume(std::shared_ptr<MatrixFactory> factory)
	: mFactory(std::move(factory))
	, mPosition(0, 0, 0)
	, mDirection(1, 0, 0)
	, mFOV(45.0f * 3.14f / 180.0f)
	, mNearPlane(1)
	, mFarPlane(1000)
	{
	}

	PerspectiveShadowMapVolume::~PerspectiveShadowMapVolume()
	{
	}

	const Eigen::Vector3f& PerspectiveShadowMapVolume::position() const
	{
		return mPosition;
	}

	void PerspectiveShadowMapVolume::setPosition(const Eigen::Vector3f& pos)
	{
		mPosition = pos;
	}

	const Eigen::Vector3f& PerspectiveShadowMapVolume::direction() const
	{
		return mDirection;
	}

	void PerspectiveShadowMapVolume::setDirection(const Eigen::Vector3f& dir)
	{
		mDirection = dir;
	}

	float PerspectiveShadowMapVolume::fieldOfView() const
	{
		return mFOV;
	}

	void PerspectiveShadowMapVolume::setFieldOfView(float fov)
	{
		mFOV = fov;
	}

	float PerspectiveShadowMapVolume::nearPlane() const
	{
		return mNearPlane;
	}

	void PerspectiveShadowMapVolume::setNearPlane(float near_plane)
	{
		mNearPlane = near_plane;
	}

	float PerspectiveShadowMapVolume::farPlane() const
	{
		return mFarPlane;
	}

	void PerspectiveShadowMapVolume::setFarPlane(float far_plane)
	{
		mFarPlane = far_plane;
	}

	Eigen::Matrix4f PerspectiveShadowMapVolume::lightMatrix() const
	{
		Eigen::Matrix4f bias = Eigen::Matrix4f::Zero();
		bias(0, 0) = 0.5f; bias(1, 1) = 0.5f; bias(2, 2) = 0.5f;
		bias(0, 3) = 0.5f; bias(1, 3) = 0.5f; bias(2, 3) = 0.5f; bias(3, 3) = 1.0f;
		return bias * computeProjectionMatrix() * computeViewMatrix();
	}

	Eigen::Matrix4f PerspectiveShadowMapVolume::computeViewMatrix() const
	{
		return mFactory->createLookAt(mPosition, mDirection, Eigen::Vector3f(0, 1, 0), Handedness::RightHanded);
	}

	Eigen::Matrix4f PerspectiveShadowMapVolume::computeProjectionMatrix() const
	{
		float aspect_ratio = (float) shadowMap()->width() / (float) shadowMap()->height();
		return mFactory->createPerspectiveFov(nearPlane(), farPlane(), aspect_ratio, fieldOfView(), Handedness::RightHanded);
	}

	PerspectiveViewFrustum<float> PerspectiveShadowMapVolume::computeFrustum() const
	{
		Eigen::Vector3f right, up, dir;
		dir = direction();
		right = Eigen::Vector3f(0, 1, 0).cross(dir).normalized();
		up = dir.cross(right).normalized();
		return PerspectiveViewFrustum<float>(1, 1, mFOV, nearPlane(), farPlane(), position(), dir, up, right);
	}

	OrthographicShadowMapVolume::OrthographicShadowMapVolume(std::shared_ptr<MatrixFactory> factory)
	: mFactory(std::move(factory))
	, mPosition(0, 0, 0)
	, mDirection(1, 0, 0)
	, mUp(0, 1, 0)
	, mRight(0, 0, 1)
	, mWidth(-1)
	, mHeight(-1)
	, mNearPlane(-1)
	, mFarPlane(-1)
	{
	}

	OrthographicShadowMapVolume::OrthographicShadowMapVolume
	(
		std::shared_ptr<MatrixFactory> factory,
		const Eigen::Vector3f& position,
		const Eigen::Vector3f& direction,
		const Eigen::Vector3f& right,
		const Eigen::Vector3f& up
	)
	: mFactory(std::move(factory))
	, mPosition(position)
	, mDirection(direction)
	, mUp(up)
	, mRight(right)
	, mWidth(-1)
	, mHeight(-1)
	, mNearPlane(-1)
	, mFarPlane(-1)
	{
	}

	OrthographicShadowMapVolume::~OrthographicShadowMapVolume()
	{
	}

	const Eigen::Vector3f& OrthographicShadowMapVolume::position() const
	{
		return mPosition;
	}

	void OrthographicShadowMapVolume::setPosition(const Eigen::Vector3f& pos)
	{
		mPosition = pos;
	}

	const Eigen::Vector3f& OrthographicShadowMapVolume::direction() const
	{
		return mDirection;
	}

	void OrthographicShadowMapVolume::setDirection(const Eigen::Vector3f& dir)
	{
		mDirection = dir;
	}

	const Eigen::Vector3f& OrthographicShadowMapVolume::right() const
	{
		return mRight;
	}

	void OrthographicShadowMapVolume::setRight(const Eigen::Vector3f& right)
	{
		mRight = right;
	}

	const Eigen::Vector3f& OrthographicShadowMapVolume::up() const
	{
		return mUp;
	}

	void OrthographicShadowMapVolume::setUp(const Eigen::Vector3f& up)
	{
		mUp = up;
	}

	float OrthographicShadowMapVolume::width() const
	{
		return mWidth;
	}

	void OrthographicShadowMapVolume::setWidth(float w)
	{
		mWidth = w;
	}

	float OrthographicShadowMapVolume::height() const
	{
		return mHeight;
	}

	void OrthographicShadowMapVolume::setHeight(float h)
	{
		mHeight = h;
	}

	float OrthographicShadowMapVolume::nearPlane() const
	{
		return mNearPlane;
	}

	void OrthographicShadowMapVolume::setNearPlane(float near_plane)
	{
		mNearPlane = near_plane;
	}

	float OrthographicShadowMapVolume::farPlane() const
	{
		return mFarPlane;
	}

	void OrthographicShadowMapVolume::setFarPlane(float far_plane)
	{
		mFarPlane = far_plane;
	}

	Eigen::Matrix4f OrthographicShadowMapVolume::lightMatrix() const
	{
		Eigen::Matrix4f bias = Eigen::Matrix4f::Zero();
		bias(0, 0) = 0.5f; bias(1, 1) = 0.5f; bias(2, 2) = 0.5f;
		bias(0, 3) = 0.5f; bias(1, 3) = 0.5f; bias(2, 3) = 0.5f; bias(3, 3) = 1.0f;
		return bias * computeProjectionMatrix() * computeViewMatrix();
	}

	Eigen::Matrix4f OrthographicShadowMapVolume::computeViewMatrix() const
	{
		using Vcl::Mathematics::equal;

		VclRequireEx(equal(mDirection.cross(mUp).dot(mRight), 1, 1e-4f), "Frame is orthogonal.", fmt::format("Angle: %f", mDirection.cross(mUp).dot(mRight)));

		return mFactory->createLookAt(mPosition, mDirection, mUp, Handedness::RightHanded);
	}

	Eigen::Matrix4f OrthographicShadowMapVolume::computeProjectionMatrix() const
	{
		VclRequire(mWidth > 0, "Width is valid");
		VclRequire(mHeight > 0, "Height is valid");
		VclRequire(mNearPlane > 0, "Near plane is valid");
		VclRequire(mFarPlane > 0, "Far plane is valid");

		return mFactory->createOrtho(mWidth, mHeight, nearPlane(), farPlane(), Handedness::RightHanded);
	}

	ParallelSplitOrthographicShadowMapVolume::ParallelSplitOrthographicShadowMapVolume
	(
		std::shared_ptr<MatrixFactory> factory,
		int nr_splits,
		float lambda
	)
	: ParallelSplitOrthographicShadowMapVolume(factory, nr_splits, lambda, Eigen::Vector3f(0, -1, 0), nullptr)
	{
	}

	ParallelSplitOrthographicShadowMapVolume::ParallelSplitOrthographicShadowMapVolume
	(
		std::shared_ptr<MatrixFactory> factory,
		int nr_splits,
		float lambda,
		const Eigen::Vector3f& direction,
		const Vcl::Graphics::PerspectiveViewFrustum<float>* frustum /* = nullptr */
	)
	: OrthographicShadowMapVolume(std::move(factory))
	, mSplits(nr_splits + 1)
	, mLambda(lambda)
	{
		VclRequire(nr_splits > 0, "At least 1 split is defined.");

		// Allocate storage
		mOrthoFrustums.resize(nr_splits);

		// Set the main the direction of the enclosing orthographic frustum
		setDirection(direction);

		// Initialize the volume
		if (frustum)
			update(frustum);
	}

	ParallelSplitOrthographicShadowMapVolume::~ParallelSplitOrthographicShadowMapVolume()
	{
	}

	void ParallelSplitOrthographicShadowMapVolume::update(const Vcl::Graphics::PerspectiveViewFrustum<float>* frustum)
	{
		VclCheck(frustum, "frustum pointer is set.");
		if (frustum)
		{
			// Compute enclosing frustum
			auto ef = OrthographicViewFrustum<float>::enclose(*frustum, direction());
			this->setPosition(ef.position());
			this->setDirection(ef.direction());
			this->setRight(ef.right());
			this->setUp(ef.up());
			this->setWidth(ef.width());
			this->setHeight(ef.height());
			this->setNearPlane(ef.nearPlane());
			this->setFarPlane(ef.farPlane());

			// Compute split positions based on
			// GPU Gems 3 - Parallel-Split Shadow Maps on Programmable GPUs
			int nr_splits = (int)mSplits.size();
			float n = frustum->nearPlane();
			float f = frustum->farPlane();

			mSplits.front() = n;
			for (int i = 1; i < nr_splits; i++)
			{
				float uniform_split = n + (f - n) * (float)i / (float)nr_splits;
				float log_split = n * pow(f / n, (float)i / (float)nr_splits);
				float split = mLambda * log_split + (1 - mLambda) * uniform_split;
				mSplits[i] = split;
			}
			mSplits.back() = f;

			// Compute orthographic volumes for each split
			for (size_t i = 0; i < mSplits.size() - 1; i++)
			{
				// Compute the perspective split frustum
				PerspectiveViewFrustum<float> psf
				(
					frustum->width(), frustum->height(), frustum->fieldOfView(),
					mSplits[i], mSplits[i+1],
					frustum->position(), frustum->direction(), frustum->up(), frustum->right()
				);

				mOrthoFrustums[i] = OrthographicViewFrustum<float>::enclose(psf, direction());
			}
		}
	}

	float ParallelSplitOrthographicShadowMapVolume::split(unsigned int idx) const
	{
		VclRequire(idx < mSplits.size() - 1, "Index is valid.");

		return mSplits[idx + 1];
	}

	Eigen::Matrix4f ParallelSplitOrthographicShadowMapVolume::lightMatrix(unsigned int split) const
	{
		Eigen::Matrix4f bias = Eigen::Matrix4f::Zero();
		bias(0, 0) = 0.5f; bias(1, 1) = 0.5f; bias(2, 2) = 0.5f;
		bias(0, 3) = 0.5f; bias(1, 3) = 0.5f; bias(2, 3) = 0.5f; bias(3, 3) = 1.0f;
		return bias * mOrthoFrustums[split].computeProjectionMatrix(*mFactory) * mOrthoFrustums[split].computeViewMatrix(*mFactory);
	}

	Eigen::Matrix4f ParallelSplitOrthographicShadowMapVolume::computeViewMatrix(unsigned int split) const
	{
		return mOrthoFrustums[split].computeViewMatrix(*mFactory);
	}

	Eigen::Matrix4f ParallelSplitOrthographicShadowMapVolume::computeProjectionMatrix(unsigned int split) const
	{
		return mOrthoFrustums[split].computeProjectionMatrix(*mFactory);
	}
}}
