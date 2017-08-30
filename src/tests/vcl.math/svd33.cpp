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

// VCL configuration
#include <vcl/config/global.h>
#include <vcl/config/eigen.h>

// C++ standard library
#include <random>

// Include the relevant parts from the library
#include <vcl/core/interleavedarray.h>
#include <vcl/math/math.h>
#include <vcl/math/jacobisvd33_mcadams.h>
#include <vcl/math/jacobisvd33_qr.h>
#include <vcl/math/jacobisvd33_twosided.h>

// Google test
#include <gtest/gtest.h>

// Common functions
#define USE_PREGEN_DATA
namespace
{
	float predefinedProblems[] =
	{
		1, 0, 0,0, 1, 0,0, 0, 1,
		1, 0, 0,0, 1, 0,0, 0, 1,
		1, 0, 0,0, 1, 0,0, 0, 1,
		1, 0, 0,0, 1, 0,0, 0, 1,
		1, 0, 0,0, 1, 0,0, 0, 1,
		1, 0, 0,0, 1, 0,0, 0, 1,
		1, 0, 0,0, 1, 0,0, 0, 1,
		1, 0, 0,0, 1, 0,0, 0, 1,
		0.35,    0,    0,   0, 0.35,    0,   0,    0, 0.35,
		0.35,    0,    0,   0, 0.35,    0,   0,    0, 0.35,
		0.35,    0,    0,   0, 0.35,    0,   0,    0, 0.35,
		0.35,    0,    0,   0, 0.35,    0,   0,    0, 0.35,
		0.35,    0,    0,   0, 0.35,    0,   0,    0, 0.35,
		0.35,    0,    0,   0, 0.35,    0,   0,    0, 0.35,
		0.35,    0,    0,   0, 0.35,    0,   0,    0, 0.35,
		0.35,    0,    0,   0, 0.35,    0,   0,    0, 0.35,
		 0.520643, 0.0227124,  0.251318, 0.404902, 0.0192711,  0.946668, 0.710671,   0.25048,  0.786821,
		 0.41937, 0.499774, 0.857077,0.521916, 0.543856, 0.140039,0.561032, 0.274196,  0.34467,
		0.715893,  0.19832, 0.245533,0.164833, 0.910501, 0.320065,0.239277, 0.249168, 0.744281,
		   0.4381,    0.9503,  0.583883, 0.777045,  0.257262,  0.459925,0.0807073,  0.769432,  0.967825,
		0.0309147,   0.90712, 0.0912189, 0.696239,  0.504976, 0.0455399, 0.256359,  0.532405,  0.322289,
		0.924194, 0.285187, 0.297847,0.480969, 0.361325, 0.298767,0.620412,  0.98152, 0.152036,
		 0.423817,  0.759622, 0.0888609, 0.486834,  0.069837,  0.722758, 0.713892,  0.755031,  0.628902,
		  0.21099,  0.987307,   0.43414, 0.225448,  0.222135,  0.623808, 0.272961, 0.0864162,  0.596983,
		 0.362117, 0.0220786,  0.490911, 0.859905,  0.935147,  0.366719, 0.836365,  0.284114,  0.853215,
		 0.487331,  0.687829,  0.980166,  0.11218,  0.909242, 0.0290329, 0.422952,  0.519371,  0.736445,
		0.888796, 0.969531,  0.49149,0.870392, 0.828304, 0.455645,0.528452, 0.588669, 0.735908,
		 0.586928,  0.101607, 0.0324015, 0.759026,  0.548245,  0.118927, 0.343911,  0.436551,  0.138265,
		0.644111, 0.867071, 0.990865,0.953564, 0.605491, 0.875281,0.849987, 0.791824, 0.695038,
		  0.66887,   0.80717,  0.147311, 0.827346,  0.904333, 0.0791418, 0.466998,  0.912689,  0.604654,
		  0.28054, 0.0633181,  0.323224, 0.355726,  0.629748,  0.956864, 0.832426,  0.672333,  0.138648,
		 0.972835,  0.951291,  0.311976, 0.627886,   0.13069,  0.100668,0.0736188, 0.0875155,  0.256984,
		0.894202, 0.748943, 0.746571,0.210137, 0.718358, 0.567179,0.676285,  0.72612, 0.201725,
		 0.868726,   0.71123,  0.281168, 0.365701,  0.611837,  0.285751,0.0265524,  0.654266,  0.116882,
		 0.79339, 0.506296,  0.96275,0.988584, 0.567172, 0.795094,0.595384, 0.887004, 0.122479,
		0.458105, 0.934162, 0.236615,0.811923, 0.341087, 0.831381,0.116003, 0.402188, 0.719541,
		  0.297123,   0.186586, 0.00570613,  0.843262,     0.5898,   0.644393,  0.369665,    0.74979,    0.85286,
		0.888838, 0.128372, 0.279316,0.128813, 0.767029, 0.901857,0.994564, 0.725961, 0.338689,
		0.833823, 0.600164, 0.517831,0.812785, 0.917667, 0.297175,0.527364, 0.400797, 0.805795,
		0.367725, 0.127923,   0.8075,0.266564, 0.467029, 0.826142, 0.59225, 0.329163, 0.477926,
		 0.371176,  0.618904,  0.987139,0.0859536,  0.691681,  0.434842, 0.161771,  0.124018, 0.0195539,
		  0.36401,  0.235788,  0.787104, 0.952125,  0.683944, 0.0994404,0.0368733,  0.158937,  0.236674,
		 0.358345, 0.0861154,  0.692114, 0.697382,  0.552795,  0.200986, 0.998865,  0.934142,  0.362999,
		 0.165289, 0.0989368,  0.387434, 0.152076,  0.519763,  0.472191, 0.943607,   0.09432,  0.286332,
		0.225179, 0.777137, 0.989908,0.931878,  0.31606, 0.544445,0.649411, 0.473452, 0.583937,
		  0.33019,  0.690941,  0.113724, 0.321522,   0.30109,   0.66992, 0.384338, 0.0439639,  0.790953,
		  0.776293,   0.700266,   0.205705,  0.420477,   0.399689,   0.751891,0.00717771,   0.782775,   0.798766,
		0.311554, 0.756531, 0.125406,0.559647,  0.91358, 0.771515,0.748577, 0.170088, 0.141306,
		  0.832282,   0.200414,   0.815662,  0.602823,   0.745415,  0.0951238,0.00443744,   0.370268,   0.725176,
		 0.277405, 0.0941685,  0.523856, 0.590795,  0.573825,  0.960691, 0.372524,  0.129493, 0.0555534,
		  0.50963,  0.187052,  0.367357,0.0742729,  0.613573,  0.414354, 0.104138,  0.569106,  0.915303,
		0.0634439,  0.114006,  0.697087,0.0509637, 0.0561426,  0.521955,0.0152459,  0.775391,  0.428071,
		0.580879, 0.489904, 0.234415,0.527771, 0.293188, 0.636769,0.198587, 0.531008, 0.356116,
		0.242226, 0.740409, 0.937673,0.579444, 0.590133, 0.395352,0.720755, 0.314094, 0.188345,
		0.972132, 0.590951, 0.962899,0.118415, 0.140093, 0.633606,0.207918, 0.113886, 0.519524,
		 0.474761,  0.908874,  0.320076, 0.128101,  0.979988,  0.156734,0.0923789,  0.880038,  0.802339,
		0.654764, 0.728632, 0.568274,0.370185, 0.307231, 0.558256,0.521725, 0.810924, 0.200951,
		0.651893, 0.358894, 0.693481,0.779626, 0.893327, 0.998629,0.911216,   0.7543, 0.311207,
			0.95554,    0.409707,    0.862113,   0.535271,    0.349181,   0.0390328,   0.559026,    0.909052, 0.000327772,
		 0.404979,  0.971619,  0.781072, 0.254931,  0.168979,  0.746042,0.0163063,  0.841278,  0.492911,
		0.487119, 0.532793, 0.900794,0.256963, 0.608284, 0.257456,0.832431, 0.955518, 0.198655,
		 0.401213,  0.359711,  0.102809,0.0781677,  0.249421,  0.581417, 0.878028,  0.596802,  0.790322,
		 0.394662,  0.924365,  0.297382, 0.424487,  0.428222, 0.0504178, 0.740887,  0.596828,  0.871441,
		0.800913, 0.587855, 0.611199,0.463175, 0.880588, 0.547968, 0.66794, 0.685165,  0.57688,
		 0.976944,  0.588149,  0.503932, 0.549357,  0.661278,  0.464039,  0.66517,  0.316831, 0.0303706,
		 0.788752,  0.599136, 0.0909839, 0.851882,  0.481792,  0.052365, 0.903292,  0.570519,  0.550525,
		 0.504267,  0.608395, 0.0637689, 0.505807,  0.193483,  0.870505,0.0221698,  0.496585,  0.111787,
		  0.87611,  0.651191, 0.0857647, 0.924414,  0.974275,  0.276202, 0.466376,  0.222165,  0.578479,
		 0.877945,  0.831223, 0.0450919, 0.214676,  0.263458,  0.618934, 0.457222,  0.246939,   0.13779,
		0.0282853,  0.804414,  0.685024,0.0915762,  0.318291,  0.224083, 0.624383,  0.304855,  0.628726,
		0.0221841,  0.201672,  0.780919, 0.239676,  0.818365, 0.0718277, 0.806745,    0.4538,  0.156149,
		 0.558087,  0.313727,   0.71475, 0.251147,  0.132566,  0.662279,0.0103248,  0.500025,  0.520758,
		0.380525,  0.74988, 0.365758,0.509431, 0.829917, 0.467508,0.887737, 0.232667, 0.227199,
		  0.64082,  0.656539,  0.458976, 0.118464, 0.0237956,  0.486476, 0.235979,  0.224702,  0.542421,
		  0.200237,   0.180076,   0.496747,0.00430347,  0.0600218,   0.993688,  0.226868,   0.637551,   0.589712,
		 0.531601,  0.829321,  0.515012, 0.689118,  0.219158, 0.0940731, 0.198204,  0.618056,  0.329565,
		0.625022, 0.617491, 0.180081,0.348768,  0.82826, 0.805169,0.239331, 0.946657, 0.576843,
		 0.425389,  0.293695,  0.201379,  0.83667,  0.439389, 0.0382358, 0.971805,  0.890589,  0.559929,
		0.485052, 0.226578, 0.812773,0.505264,  0.79858, 0.544856,0.869584, 0.121113, 0.435041,
		 0.650536,   0.85549, 0.0245672, 0.398691,   0.79577,  0.195309,  0.26119,   0.14189,  0.797712,
		 0.579084,  0.413172,  0.858129, 0.939198,  0.557455, 0.0830819, 0.772611,  0.754257,  0.812641,
		0.269266, 0.782017, 0.668374,0.777096, 0.792946, 0.830731,0.793292, 0.166222, 0.100147,
		 0.876922,  0.276656, 0.0987145, 0.281124,  0.198392,  0.175454, 0.986548,  0.503781,  0.614568,
		0.442933, 0.498453,  0.42923,0.148727, 0.564425,  0.82545,0.418755, 0.375626,  0.85107,
		 0.83865, 0.228017, 0.158304,0.462783, 0.315445, 0.461605,0.955961, 0.398656, 0.153426,
		0.929713, 0.964296, 0.949445,0.963796, 0.301274, 0.668396,0.842813, 0.313689, 0.456873,
		  0.426122,   0.849068,    0.49556,  0.614552,   0.922741,  0.0115287,  0.721641,   0.292288, 0.00523035,
		 0.757397,  0.238789, 0.0998681, 0.296046,   0.81877,  0.482061, 0.807112,  0.725405,  0.753475,
		 0.31726,  0.96102, 0.778193,0.515558, 0.574123, 0.881345,0.110429, 0.818204, 0.939234,
		0.987285,  0.96694, 0.373858,0.444595, 0.503499, 0.757264,0.903807, 0.823076, 0.829222,
		  0.31973,  0.799977, 0.0769521, 0.462343,  0.696496,  0.659913, 0.367233,   0.59787,  0.736685,
		0.126635,  0.19819, 0.661052, 0.16328, 0.498686, 0.818349,0.086843, 0.676035, 0.427997,
		 0.856602,  0.456753,  0.829606, 0.827364,  0.398371, 0.0429127, 0.857263, 0.0531367,  0.805169,
		0.622407, 0.811645, 0.310326,0.241334, 0.795906, 0.359206, 0.72657, 0.610263, 0.762972,
		 0.154424,  0.871387, 0.0699211, 0.272157,  0.750185,  0.491876, 0.065452, 0.0996625,  0.935215,
		0.446057, 0.626001, 0.168493,0.732351, 0.371656, 0.278478,0.698676, 0.166497, 0.323832,
		  0.16023,  0.017112,  0.980462, 0.925277,  0.857562, 0.0151172, 0.663758, 0.0700159, 0.0301436,
		 0.830635,  0.075168, 0.0587445, 0.734044, 0.0486698,  0.913913, 0.899506,  0.069936,  0.747241,
		   0.4213,  0.129988,  0.878506, 0.396422,  0.782106,  0.320763,0.0489925, 0.0204567,  0.932468,
		 0.248066,  0.338049,  0.980579,0.0166563,  0.776577,  0.436277, 0.096566,  0.586235,  0.250935,
		 0.252768,  0.367929,  0.832211, 0.568896,  0.931525,  0.387573,0.0715725,  0.621577,  0.645227,
		 0.577878,  0.967671,  0.344483,0.0849481,  0.789609,  0.510477, 0.992265,  0.874716,  0.496346,
		 0.869768,  0.124984,   0.56634, 0.651146,  0.607965,  0.866855,0.0953233,  0.633072,  0.949746,
		0.194975, 0.264466, 0.528521,0.882346, 0.689586, 0.940496, 0.59432, 0.893492, 0.662669,
		 0.452568,   0.13327,   0.52997,  0.30248,  0.677981,  0.664288, 0.195069,  0.369203, 0.0677825,
		0.857862, 0.113165, 0.881989,0.108224,  0.11142, 0.242903,0.923583,  0.66295, 0.386046,
			0.875,  0.917315,  0.183098, 0.411109, 0.0470179,  0.736449,   0.4843, 0.0386098,  0.139198,
		 0.281885,  0.393752,  0.270645, 0.758243,  0.682393,  0.721181,0.0871171, 0.0892393,  0.661774,
		 0.710648,  0.200916,  0.631902, 0.901797, 0.0933125,  0.557962, 0.311426,  0.780639,  0.491221,
		0.972465, 0.449748, 0.868409,0.590519, 0.524292, 0.514068,0.463618, 0.897181, 0.855578,
		 0.966538,  0.328141,  0.227367, 0.103953,  0.936987, 0.0151045, 0.447349,  0.100356,  0.339617,
		 0.420638,  0.856489,  0.321358, 0.614714, 0.0592269,  0.836418, 0.859319,  0.688564,   0.92906,
		0.0900997,  0.845989,   0.93764, 0.467594,  0.608119, 0.0881119, 0.468066, 0.0795336,  0.359514,
		0.513898,   0.3573, 0.619364, 0.29623, 0.855836, 0.128509,0.104001, 0.453015, 0.928457,
		 0.768058,  0.192599,  0.292531,0.0344313,   0.66978,  0.936939, 0.998312,  0.420276,  0.822518,
		 0.116618,  0.159051,  0.185804, 0.380808,   0.33857,  0.504723,0.0881964,  0.854314,  0.684523,
		0.929677, 0.194848, 0.123159,0.857129,  0.35805, 0.557868, 0.40652, 0.744052,  0.62557,
		0.677233, 0.671387, 0.395753,0.461236, 0.229644, 0.205188,0.384446, 0.736109, 0.470506,
		 0.81209, 0.228349, 0.545617,0.898571, 0.288337, 0.144114,0.282391, 0.451296, 0.396541,
		0.304525, 0.219186, 0.522225,0.340464, 0.873487, 0.241475,0.601216, 0.125668, 0.908432,
		0.949059, 0.856115, 0.366398,0.670812, 0.178851, 0.380891, 0.62556,  0.66582, 0.318581,
		  0.20306,  0.160372,  0.261225, 0.678637,  0.213662, 0.0311836, 0.335429,  0.746307,  0.660923,
		0.977625,  0.80036, 0.964161,0.488311, 0.345297, 0.574249,0.026376, 0.842607, 0.531147,
		  0.579516,  0.0839241,   0.673116,  0.607957,    0.33216, 0.00409965,   0.55199,   0.132121,   0.897562,
		0.320536, 0.187267, 0.894589,0.915221, 0.394729, 0.648809,0.712971, 0.926536, 0.591511,
		 0.565713,  0.683051, 0.0481479, 0.611127, 0.0831488,  0.541202,  0.33693,  0.458179,  0.288948,
		0.00312662,   0.819984,    0.95343, 0.0649215,   0.356546,   0.770275,  0.346772,   0.822183,   0.267408,
		0.535523, 0.715216, 0.921771,0.986854, 0.965521, 0.116363,0.668796, 0.160807, 0.552573                                   
	};

	template<typename Scalar>
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> createProblems(size_t nr_problems)
	{
		Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> F(nr_problems);
	
#ifdef USE_PREGEN_DATA
		for (int i = 0; i < (int) nr_problems; i++)
		{
			Eigen::Matrix<Scalar, 3, 3> A;
			A << predefinedProblems[9*i + 0],
			     predefinedProblems[9*i + 1],
			     predefinedProblems[9*i + 2],
			     predefinedProblems[9*i + 3],
			     predefinedProblems[9*i + 4],
			     predefinedProblems[9*i + 5],
			     predefinedProblems[9*i + 6],
			     predefinedProblems[9*i + 7],
			     predefinedProblems[9*i + 8];
			F.template at<Scalar>(i) = A;
		}
#else
		// Random number generator
		std::mt19937_64 rng{ 5489 };
		std::uniform_real_distribution<float> d;

		// Initialize data
		for (int i = 0; i < (int) nr_problems; i++)
		{
			if (i < 8)
			{
				F.template at<Scalar>(i) = Eigen::Matrix<Scalar, 3, 3>::Identity();
			}
			else if (i < 16)
			{
				F.template at<Scalar>(i) = 0.35f * Eigen::Matrix<Scalar, 3, 3>::Identity();
			}
			else
			{
				Eigen::Matrix<Scalar, 3, 3> rnd;
				rnd << d(rng), d(rng), d(rng),
					   d(rng), d(rng), d(rng),
					   d(rng), d(rng), d(rng);
				F.template at<Scalar>(i) = rnd;
			}
		}
#endif

		return F;
	}

	template<typename Scalar>
	void computeReferenceSolution
	(
		size_t nr_problems,
		const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& F,
		Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& U,
		Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& V,
		Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& S
	)
	{
		// Compute reference using Eigen
		for (int i = 0; i < static_cast<int>(nr_problems); i++)
		{
			Vcl::Matrix3f A = F.template at<Scalar>(i);
			Eigen::JacobiSVD<Vcl::Matrix3f> eigen_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
			U.template at<Scalar>(i) = eigen_svd.matrixU();
			V.template at<Scalar>(i) = eigen_svd.matrixV();
			S.template at<Scalar>(i) = eigen_svd.singularValues();
		}
	}

	template<typename Scalar>
	void checkSolution
	(
		size_t nr_problems,
		Scalar tol,
		const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refUa,
		const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& refVa,
		const Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& refSa,
		const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resUa,
		const Vcl::Core::InterleavedArray<Scalar, 3, 3, -1>& resVa,
		const Vcl::Core::InterleavedArray<Scalar, 3, 1, -1>& resSa
	)
	{
		using Vcl::Mathematics::equal;

		Eigen::IOFormat fmt(6, 0, ", ", ";", "[", "]");

		for (int i = 0; i < static_cast<int>(nr_problems); i++)
		{
			Vcl::Matrix3f refU = refUa.template at<Scalar>(i);
			Vcl::Matrix3f refV = refVa.template at<Scalar>(i);
			Vcl::Vector3f refS = refSa.template at<Scalar>(i);

			Vcl::Matrix3f resU = resUa.template at<Scalar>(i);
			Vcl::Matrix3f resV = resVa.template at<Scalar>(i);
			Vcl::Vector3f resS = resSa.template at<Scalar>(i);

			if (refS(0) > 0 && refS(1) > 0 && refS(2) < 0)
				refV.col(2) *= -1;
			if (resS(0) > 0 && resS(1) > 0 && resS(2) < 0)
				resV.col(2) *= -1;

			Vcl::Matrix3f refR = refU * refV.transpose();
			Vcl::Matrix3f resR = resU * resV.transpose();

			Scalar sqLenRefUc0 = refU.col(0).squaredNorm();
			Scalar sqLenRefUc1 = refU.col(1).squaredNorm();
			Scalar sqLenRefUc2 = refU.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenRefUc0, Scalar(1), tol)) << "Reference U(" << i << "): Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefUc1, Scalar(1), tol)) << "Reference U(" << i << "): Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefUc2, Scalar(1), tol)) << "Reference U(" << i << "): Column 2 is not normalized.";

			Scalar sqLenResUc0 = resU.col(0).squaredNorm();
			Scalar sqLenResUc1 = resU.col(1).squaredNorm();
			Scalar sqLenResUc2 = resU.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenResUc0, Scalar(1), tol)) << "Result U(" << i << "): Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenResUc1, Scalar(1), tol)) << "Result U(" << i << "): Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenResUc2, Scalar(1), tol)) << "Result U(" << i << "): Column 2 is not normalized.";

			Scalar sqLenRefVc0 = refV.col(0).squaredNorm();
			Scalar sqLenRefVc1 = refV.col(1).squaredNorm();
			Scalar sqLenRefVc2 = refV.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenRefVc0, Scalar(1), tol)) << "Reference V(" << i << "): Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefVc1, Scalar(1), tol)) << "Reference V(" << i << "): Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefVc2, Scalar(1), tol)) << "Reference V(" << i << "): Column 2 is not normalized.";

			Scalar sqLenResVc0 = resV.col(0).squaredNorm();
			Scalar sqLenResVc1 = resV.col(1).squaredNorm();
			Scalar sqLenResVc2 = resV.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenResVc0, Scalar(1), tol)) << "Result V(" << i << "): Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenResVc1, Scalar(1), tol)) << "Result V(" << i << "): Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenResVc2, Scalar(1), tol)) << "Result V(" << i << "): Column 2 is not normalized.";

			Scalar sqLenRefRc0 = refR.col(0).squaredNorm();
			Scalar sqLenRefRc1 = refR.col(1).squaredNorm();
			Scalar sqLenRefRc2 = refR.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenRefRc0, Scalar(1), tol)) << "Reference R(" << i << "): Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefRc1, Scalar(1), tol)) << "Reference R(" << i << "): Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenRefRc2, Scalar(1), tol)) << "Reference R(" << i << "): Column 2 is not normalized.";

			Scalar sqLenResRc0 = resR.col(0).squaredNorm();
			Scalar sqLenResRc1 = resR.col(1).squaredNorm();
			Scalar sqLenResRc2 = resR.col(2).squaredNorm();
			EXPECT_TRUE(equal(sqLenResRc0, Scalar(1), tol)) << "Result R(" << i << "): Column 0 is not normalized.";
			EXPECT_TRUE(equal(sqLenResRc1, Scalar(1), tol)) << "Result R(" << i << "): Column 1 is not normalized.";
			EXPECT_TRUE(equal(sqLenResRc2, Scalar(1), tol)) << "Result R(" << i << "): Column 2 is not normalized.";

			bool eqS = refS.array().abs().isApprox(resS.array().abs(), tol);
			bool eqR = refR.array().abs().isApprox(resR.array().abs(), tol);

			EXPECT_TRUE(eqS) << "S(" << i << ") -\nRef: " << refS.format(fmt) << ",\nRes: " << resS.format(fmt);
			EXPECT_TRUE(eqR) << "R(" << i << ") -\nRef: " << refR.format(fmt) << ",\nRes: " << resR.format(fmt);
		}
	}
}

#if defined(VCL_VECTORIZE_SSE) || defined(VCL_VECTORIZE_AVX)
template<typename WideScalar>
void runMcAdamsTest(float tol)
{
	using scalar_t  = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	size_t nr_problems = 128;
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> refS(nr_problems);

	auto F = createProblems<scalar_t>(nr_problems);
	computeReferenceSolution(nr_problems, F, refU, refV, refS);

	// Strides
	size_t stride = nr_problems;
	size_t width = sizeof(real_t) / sizeof(scalar_t);

	for (int i = 0; i < static_cast<int>(stride / width); i++)
	{
		matrix3_t S = F.at<real_t>(i);
		matrix3_t U = matrix3_t::Identity();
		matrix3_t V = matrix3_t::Identity();

		Vcl::Mathematics::McAdamsJacobiSVD(S, U, V, 5);
		
		resU.at<real_t>(i) = U;
		resV.at<real_t>(i) = V;
		resS.at<real_t>(i) = S.diagonal();
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, refU, refV, refS, resU, resV, resS);
}
#endif // defined(VCL_VECTORIZE_SSE) || defined(VCL_VECTORIZE_AVX)

template<typename WideScalar>
void runTwoSidedTest(float tol)
{
	using scalar_t = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	size_t nr_problems = 128;
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> refS(nr_problems);

	auto F = createProblems<scalar_t>(nr_problems);
	computeReferenceSolution(nr_problems, F, refU, refV, refS);

	// Strides
	size_t stride = nr_problems;
	size_t width = sizeof(real_t) / sizeof(scalar_t);

	for (int i = 0; i < static_cast<int>(stride / width); i++)
	{
		matrix3_t S = F.at<real_t>(i);
		matrix3_t U = matrix3_t::Identity();
		matrix3_t V = matrix3_t::Identity();

		Vcl::Mathematics::TwoSidedJacobiSVD(S, U, V);

		resU.at<real_t>(i) = U;
		resV.at<real_t>(i) = V;
		resS.at<real_t>(i) = S.diagonal();
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, refU, refV, refS, resU, resV, resS);
}

template<typename WideScalar>
void runQRTest(float tol)
{
	using scalar_t = float;
	using real_t = WideScalar;
	using matrix3_t = Eigen::Matrix<real_t, 3, 3>;
	using vector3_t = Eigen::Matrix<real_t, 3, 1>;

	size_t nr_problems = 128;
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> resV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> resS(nr_problems);

	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refU(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 3, -1> refV(nr_problems);
	Vcl::Core::InterleavedArray<scalar_t, 3, 1, -1> refS(nr_problems);

	auto F = createProblems<scalar_t>(nr_problems);
	computeReferenceSolution(nr_problems, F, refU, refV, refS);

	// Strides
	size_t stride = nr_problems;
	size_t width = sizeof(real_t) / sizeof(scalar_t);

	for (int i = 0; i < static_cast<int>(stride / width); i++)
	{
		matrix3_t S = F.at<real_t>(i);
		matrix3_t U = matrix3_t::Identity();
		matrix3_t V = matrix3_t::Identity();

		Vcl::Mathematics::QRJacobiSVD(S, U, V);

		resU.at<real_t>(i) = U;
		resV.at<real_t>(i) = V;
		resS.at<real_t>(i) = S.diagonal();
	}

	// Check against reference solution
	checkSolution(nr_problems, tol, refU, refV, refS, resU, resV, resS);
}

#ifdef VCL_VECTORIZE_SSE
TEST(SVD33, McAdamsSVDFloat)
{
	runMcAdamsTest<float>(1e-5f);
}

TEST(SVD33, McAdamsSVDFloat4)
{
	runMcAdamsTest<Vcl::float4>(1e-5f);
}
#endif // defined(VCL_VECTORIZE_SSE)

#ifdef VCL_VECTORIZE_AVX
TEST(SVD33, McAdamsSVDFloat8)
{
	runMcAdamsTest<Vcl::float8>(1e-5f);
}
#endif // defined VCL_VECTORIZE_AVX

TEST(SVD33, TwoSidedSVDFloat)
{
	runTwoSidedTest<float>(1e-4f);
}
TEST(SVD33, TwoSidedSVDFloat4)
{
	runTwoSidedTest<Vcl::float4>(1e-5f);
}
TEST(SVD33, TwoSidedSVDFloat8)
{
	runTwoSidedTest<Vcl::float8>(1e-5f);
}

TEST(SVD33, QRSVDFloat)
{
	runQRTest<float>(1e-5f);
}
TEST(SVD33, QRSVDFloat4)
{
	runQRTest<Vcl::float4>(1e-5f);
}
TEST(SVD33, QRSVDFloat8)
{
	runQRTest<Vcl::float8>(1e-5f);
}
