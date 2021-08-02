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

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// VCL
#include <vcl/graphics/shadows/algorithms.h>
#include <vcl/graphics/frustum.h>
#include <vcl/graphics/matrixfactory.h>

// Forward declaration
namespace Vcl { namespace Graphics {
	class ShadowMap;
}}
namespace Vcl { namespace Graphics {
	class Camera;
}}

// Declaration
namespace Vcl { namespace Graphics {
	class ShadowMapVolume
	{
	protected:
		ShadowMapVolume() = default;

	public:
		virtual ~ShadowMapVolume();

	public:
		ShadowMapAlgorithm shadowMapAlgorithm() const;
		void setShadowMapAlgorithm(ShadowMapAlgorithm algorithm);

	public:
		ShadowMap* shadowMap() const;
		void setShadowMap(std::unique_ptr<ShadowMap> shadow_map);

	private:
		ShadowMapAlgorithm _shadowMapAlgorithm = ShadowMapAlgorithm::None;
		std::unique_ptr<ShadowMap> _shadowMap;
	};

	class PerspectiveShadowMapVolume : public ShadowMapVolume
	{
	public:
		PerspectiveShadowMapVolume(std::shared_ptr<MatrixFactory> factory);
		virtual ~PerspectiveShadowMapVolume();

	public:
		std::shared_ptr<MatrixFactory> matrixFactory() const
		{
			return mFactory;
		}

	public:
		const Eigen::Vector3f& position() const;
		void setPosition(const Eigen::Vector3f& pos);

		const Eigen::Vector3f& direction() const;
		void setDirection(const Eigen::Vector3f& dir);

	public:
		float fieldOfView() const;
		void setFieldOfView(float fov);

		float nearPlane() const;
		void setNearPlane(float near_plane);

		float farPlane() const;
		void setFarPlane(float far_plane);

	public:
		Eigen::Matrix4f lightMatrix() const;
		Eigen::Matrix4f computeViewMatrix() const;
		Eigen::Matrix4f computeProjectionMatrix() const;
		PerspectiveViewFrustum<float> computeFrustum() const;

	public: /* Factories */
		std::shared_ptr<MatrixFactory> mFactory;

	private: /* Configuration */
		Eigen::Vector3f mPosition;
		Eigen::Vector3f mDirection;
		float mFOV;
		float mNearPlane;
		float mFarPlane;
	};

	class OrthographicShadowMapVolume : public ShadowMapVolume
	{
	public:
		OrthographicShadowMapVolume(std::shared_ptr<MatrixFactory> factory);
		OrthographicShadowMapVolume
		(
			std::shared_ptr<MatrixFactory> factory,
			const Eigen::Vector3f& position,
			const Eigen::Vector3f& direction,
			const Eigen::Vector3f& right,
			const Eigen::Vector3f& up
		);
		virtual ~OrthographicShadowMapVolume();

	public:
		std::shared_ptr<MatrixFactory> matrixFactory() const
		{
			return mFactory;
		}

	public:
		const Eigen::Vector3f& position() const;
		void setPosition(const Eigen::Vector3f& pos);

		const Eigen::Vector3f& direction() const;
		void setDirection(const Eigen::Vector3f& dir);

		const Eigen::Vector3f& right() const;
		void setRight(const Eigen::Vector3f& pos);

		const Eigen::Vector3f& up() const;
		void setUp(const Eigen::Vector3f& pos);

	public:
		float width() const;
		void setWidth(float w);

		float height() const;
		void setHeight(float h);

		float nearPlane() const;
		void setNearPlane(float near_plane);

		float farPlane() const;
		void setFarPlane(float far_plane);

	public:
		Eigen::Matrix4f lightMatrix() const;
		Eigen::Matrix4f computeViewMatrix() const;
		Eigen::Matrix4f computeProjectionMatrix() const;

	public: /* Factories */
		std::shared_ptr<MatrixFactory> mFactory;

	private: /* Configuration */
		Eigen::Vector3f mPosition;
		Eigen::Vector3f mDirection;
		Eigen::Vector3f mUp;
		Eigen::Vector3f mRight;
		float mWidth;
		float mHeight;
		float mNearPlane;
		float mFarPlane;
	};

	class ParallelSplitOrthographicShadowMapVolume : public OrthographicShadowMapVolume
	{
	public:
		ParallelSplitOrthographicShadowMapVolume
		(
			std::shared_ptr<MatrixFactory> factory,
			int nr_splits,
			float lambda
		);
		ParallelSplitOrthographicShadowMapVolume
		(
			std::shared_ptr<MatrixFactory> factory,
			int nr_splits,
			float lambda,
			const Eigen::Vector3f& direction,
			const Vcl::Graphics::PerspectiveViewFrustum<float>* frustum = nullptr
		);
		virtual ~ParallelSplitOrthographicShadowMapVolume();

	public:
		void update(const Vcl::Graphics::PerspectiveViewFrustum<float>* frustum);

	public:
		int nrSplits() const { return (int)mSplits.size() - 1; }
		float split(unsigned int idx) const;

	public:
		using OrthographicShadowMapVolume::computeProjectionMatrix;
		using OrthographicShadowMapVolume::computeViewMatrix;
		using OrthographicShadowMapVolume::lightMatrix;
		Eigen::Matrix4f lightMatrix(unsigned int split) const;
		Eigen::Matrix4f computeViewMatrix(unsigned int split) const;
		Eigen::Matrix4f computeProjectionMatrix(unsigned int split) const;

	private: /* Configuration */
		std::vector<float> mSplits;
		float mLambda;

		///! Orthographic frustums of the splits
		std::vector<OrthographicViewFrustum<float>, Eigen::aligned_allocator<OrthographicViewFrustum<float>>> mOrthoFrustums;
	};

	//class CascadedOrthographicShadowMapVolume : public OrthographicShadowMapVolume
	//{
	//public:
	//	CascadedOrthographicShadowMapVolume();
	//	virtual ~CascadedOrthographicShadowMapVolume();
	//};
}}
