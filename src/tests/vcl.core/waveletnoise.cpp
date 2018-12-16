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

// C++ standard library
#include <numeric>

// Include the relevant parts from the library
#include <vcl/math/math.h>
#include <vcl/util/waveletnoise.h>
#include <vcl/util/waveletnoise_helpers.h>
#include <vcl/util/waveletnoise_modulo.h>

VCL_BEGIN_EXTERNAL_HEADERS

// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

#include "waveletnoise_rnd.h"

class WaveletNoiseTest : public Vcl::Util::WaveletNoise<32>, public testing::Test
{
public:
	WaveletNoiseTest()
		: WaveletNoise(random_numbers)
	{}
};

bool equal
(
	const std::array<float, 3>& x,
	const std::array<float, 3>& y,
	float tol = 0
)
{
	bool eq = true;
	for (int i = 0; i < 3; i++)
	{
		eq = eq && Vcl::Mathematics::equal(x[i], y[i], tol);
	}

	return eq;
}

using namespace Vcl::Util::Details;

TEST_F(WaveletNoiseTest, Modulo)
{
	using namespace Vcl::Util;

	int m0 = 53 % 47;
	EXPECT_EQ(m0, 6);
	int m1 = FastMath<47>::modulo(53);
	EXPECT_EQ(m1, 6);

	int m2 = -53 % 47;
	EXPECT_EQ(m2, -6);
	int m3 = FastMath<47>::modulo(-53);
	EXPECT_EQ(m3, 41);

	int m4 = -17 % 16;
	EXPECT_EQ(m4, -1);
	int m5 = FastMath<16>::modulo(-17);
	EXPECT_EQ(m5, 15);

	int m6 = -33 % 32;
	EXPECT_EQ(m6, -1);
	int m7 = FastMath<32>::modulo(-33);
	EXPECT_EQ(m7, 31);

	int m8 = -65 % 64;
	EXPECT_EQ(m8, -1);
	int m9 = FastMath<64>::modulo(-65);
	EXPECT_EQ(m9, 63);

	int m10 = -129 % 128;
	EXPECT_EQ(m10, -1);
	int m11 = FastMath<128>::modulo(-129);
	EXPECT_EQ(m11, 127);

	Vcl::int8 m12 = FastMath<64>::modulo(Vcl::int8(-65));
	EXPECT_TRUE(Vcl::all(m12 == Vcl::int8(63)));
}

TEST_F(WaveletNoiseTest, EvalSpline)
{
	Vec3 values0, values1, values2, values3, values4;
	int mid0, mid1, mid2, mid3, mid4;
	evaluateQuadraticSplineBasis(0.0f, values0, mid0);
	EXPECT_TRUE(equal(values0, { 0.125f, 0.75f, 0.125f }, 1e-4f));

	evaluateQuadraticSplineBasis(0.25f, values1, mid1);
	EXPECT_TRUE(equal(values1, { 0.03125f, 0.6875f, 0.28125f }, 1e-4f));

	evaluateQuadraticSplineBasis(0.5f, values2, mid2);
	EXPECT_TRUE(equal(values2, { 0.0f, 0.5f, 0.5f }, 1e-4f));

	evaluateQuadraticSplineBasis(0.75f, values3, mid3);
	EXPECT_TRUE(equal(values3, { 0.28125f, 0.6875f, 0.03125f }, 1e-4f));

	evaluateQuadraticSplineBasis(1.0f, values4, mid4);
	EXPECT_TRUE(equal(values4, { 0.125f, 0.75f, 0.125f }, 1e-4f));
}

TEST_F(WaveletNoiseTest, ACoeffs)
{
	const auto sum = std::accumulate(std::begin(ACoeffs), std::end(ACoeffs), 0.0f);
	EXPECT_TRUE(Vcl::Mathematics::equal(sum, 1.0f, 1e-4f));
}

TEST_F(WaveletNoiseTest, DownsampleIdentity)
{
	std::array<float, 64> values;
	std::generate(std::begin(values), std::end(values), []() { return 1.0f; });

	std::array<float, 32> downsampled;
	downsample<32>(values, downsampled, 64, 1);

	const auto sum = std::accumulate(std::begin(downsampled), std::end(downsampled), 0.0f);
	EXPECT_TRUE(Vcl::Mathematics::equal(sum, 32.0f, 1e-4f)) << "Actual sum: " << sum;
}

TEST_F(WaveletNoiseTest, Upsample)
{
	std::array<float, 32> values;
	std::generate(std::begin(values), std::end(values), []() { return 1.0f; });

	std::array<float, 64> upsampled;
	upsample<32>(values, upsampled, 64, 1);

	const auto sum = std::accumulate(std::begin(upsampled), std::end(upsampled), 0.0f);
	EXPECT_TRUE(Vcl::Mathematics::equal(sum, 64.0f, 1e-4f)) << "Actual sum: " << sum;
}

TEST_F(WaveletNoiseTest, Evaluate)
{
	const std::array<float, 3> offset = { 0.25f, 0.5f, 0.75f };
	const std::array<float, 27> ref = {
		0.692688227f,
		0.662242115f,
		0.564585209f,
		0.902872443f,
		0.845676601f,
		0.697789967f,
		1.003895040f,
		0.934411824f,
		0.756718814f,
		0.625678957f,
		0.609067559f,
		0.546854317f,
		0.868099511f,
		0.779047072f,
		0.619916975f,
		1.015417340f,
		0.865536869f,
		0.626371861f,
		0.491447628f,
		0.483999848f,
		0.456886590f,
		0.769531250f,
		0.649453819f,
		0.484109968f,
		0.973325670f,
		0.749123514f,
		0.458190203f,
	};

	for (int k = 0; k < 3; k++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				using namespace Vcl::Mathematics;
				float noise_value = evaluate({ offset[i], offset[j], offset[k] });
				EXPECT_TRUE(equal(ref[k*9 + j*3 + i], noise_value, 1e-5f));
			}
		}
	}
}

TEST_F(WaveletNoiseTest, EvaluateWithNormal)
{
	const std::array<float, 3> offset = { 0.25f, 0.5f, 0.75f };
	const std::array<float, 8*27> ref = {
		0.692688227f,
		0.662242115f,
		0.564585209f,
		0.902872443f,
		0.845676601f,
		0.697789967f,
		1.00389504f,
		0.934411824f,
		0.756718814f,
		0.625678957f,
		0.609067559f,
		0.546854317f,
		0.868099511f,
		0.779047072f,
		0.619916975f,
		1.01541734f,
		0.865536869f,
		0.626371861f,
		0.491447628f,
		0.483999848f,
		0.456886590f,
		0.769531250f,
		0.649453819f,
		0.484109968f,
		0.973325670f,
		0.749123514f,
		0.458190203f,
		0.864743173f,
		0.721788406f,
		0.546328425f,
		1.05490828f,
		0.886431456f,
		0.678256929f,
		1.08095074f,
		0.915276647f,
		0.705668747f,
		0.921509445f,
		0.779113173f,
		0.599613369f,
		1.05914927f,
		0.911037743f,
		0.719708085f,
		1.05419338f,
		0.924694240f,
		0.749143302f,
		0.872233808f,
		0.737685382f,
		0.566641152f,
		0.970853627f,
		0.850623667f,
		0.687317193f,
		0.958868742f,
		0.871719778f,
		0.738231957f,
		0.646314383f,
		0.601079285f,
		0.455976158f,
		0.767660201f,
		0.785699129f,
		0.684867203f,
		0.806610405f,
		0.896288693f,
		0.855505407f,
		0.703951001f,
		0.593739510f,
		0.413372070f,
		0.819247723f,
		0.743878961f,
		0.582853734f,
		0.856479824f,
		0.820869207f,
		0.690276623f,
		0.706779540f,
		0.533816874f,
		0.323436260f,
		0.809315562f,
		0.640413046f,
		0.423065484f,
		0.844134808f,
		0.680898666f,
		0.462689817f,
		0.717736006f,
		0.454741180f,
		0.102548830f,
		1.13442528f,
		0.885776043f,
		0.535139620f,
		1.40086555f,
		1.23961627f,
		0.910763204f,
		0.707506299f,
		0.466071367f,
		0.207032576f,
		1.09155560f,
		0.819142997f,
		0.523766935f,
		1.35288596f,
		1.13101363f,
		0.815188348f,
		0.614451110f,
		0.400571376f,
		0.245464042f,
		0.955548048f,
		0.665633321f,
		0.433012336f,
		1.20858502f,
		0.930599272f,
		0.634601235f,
		0.596896291f,
		0.542045832f,
		0.427396834f,
		1.13151431f,
		1.01120579f,
		0.779327035f,
		1.54522729f,
		1.37624192f,
		1.05034018f,
		0.610650897f,
		0.551636815f,
		0.443106025f,
		1.10954213f,
		0.956213653f,
		0.704831839f,
		1.48664212f,
		1.25685656f,
		0.886612713f,
		0.572940350f,
		0.515551567f,
		0.422461331f,
		1.02942300f,
		0.852136254f,
		0.594572484f,
		1.37041688f,
		1.09051609f,
		0.691158593f,
		0.692876995f,
		0.669660389f,
		0.522920072f,
		0.898197770f,
		0.880417824f,
		0.754419863f,
		0.961450994f,
		0.967666030f,
		0.893134058f,
		0.703480482f,
		0.714834988f,
		0.648220599f,
		0.893938482f,
		0.866583228f,
		0.778014839f,
		0.941001117f,
		0.888965070f,
		0.801896274f,
		0.624060571f,
		0.649944305f,
		0.628636003f,
		0.808005989f,
		0.777810335f,
		0.695799589f,
		0.860910714f,
		0.783082068f,
		0.657360196f,
		0.894334018f,
		0.719821572f,
		0.446954191f,
		1.12474072f,
		0.926483750f,
		0.606280684f,
		1.16488802f,
		1.00558078f,
		0.713439882f,
		0.732828915f,
		0.551056623f,
		0.300912142f,
		1.06682968f,
		0.834650695f,
		0.502997398f,
		1.21526706f,
		0.986929715f,
		0.639335990f,
		0.468750834f,
		0.295901924f,
		0.0903942436f,
		0.869327605f,
		0.633219779f,
		0.329296201f,
		1.09934533f,
		0.841500342f,
		0.489230365f,
		0.508207798f,
		0.416183472f,
		0.250023574f,
		0.833702147f,
		0.717345834f,
		0.526882470f,
		1.00087154f,
		0.930119395f,
		0.761069655f,
		0.420504987f,
		0.319878995f,
		0.206589520f,
		0.798975587f,
		0.627890468f,
		0.456767023f,
		1.02119780f,
		0.853273869f,
		0.657114565f,
		0.241463572f,
		0.113030255f,
		0.0384803079f,
		0.652849317f,
		0.427654475f,
		0.262900203f,
		0.948867023f,
		0.703019142f,
		0.465040058f,
	};

	for (int z = 0; z < 2; z++)
	{
		for (int y = 0; y < 2; y++)
		{
			for (int x = 0; x < 2; x++)
			{
				for (int k = 0; k < 3; k++)
				{
					for (int j = 0; j < 3; j++)
					{
						for (int i = 0; i < 3; i++)
						{
							using namespace Vcl::Mathematics;
							const auto n = Eigen::Vector3f(x, y, z).normalized();
							const Vec3 normal = { n[0], n[1], n[2] };
							float noise_value = evaluate({ offset[i], offset[j], offset[k] }, normal);
							EXPECT_TRUE(equal(ref[27*(z*4 + y*2 + x) + k * 9 + j * 3 + i], noise_value, 1e-5f));
						}
					}
				}
			}
		}
	}
}

TEST_F(WaveletNoiseTest, DxDyDz)
{
	const std::array<float, 5> offset = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };

	for (int k = 0; k < 5; k++)
	{
		for (int j = 0; j < 5; j++)
		{
			for (int i = 0; i < 5; i++)
			{
				using namespace Vcl::Mathematics;

				const Vec3 p{ offset[i], offset[j], offset[k] };

				Mat33 result;
				dxDyDz(p, result);
				
				const float fx = dx(p);
				const float fy = dy(p);
				const float fz = dz(p);

				EXPECT_TRUE(equal(result[0][1], fy, 1e-5f));
				EXPECT_TRUE(equal(result[0][2], fz, 1e-5f));
				EXPECT_TRUE(equal(result[1][0], fx, 1e-5f));
				EXPECT_TRUE(equal(result[1][2], fz, 1e-5f));
				EXPECT_TRUE(equal(result[2][0], fx, 1e-5f));
				EXPECT_TRUE(equal(result[2][1], fy, 1e-5f));
			}
		}
	}
}
