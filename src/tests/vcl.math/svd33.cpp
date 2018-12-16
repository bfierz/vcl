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

VCL_BEGIN_EXTERNAL_HEADERS
// Google test
#include <gtest/gtest.h>
VCL_END_EXTERNAL_HEADERS

// Common functions
#define USE_PREGEN_DATA
namespace
{
	// clang-format off
	float predefinedProblems[] =
	{
		1, 0, 0, 0, 1, 0, 0, 0, 1,
		1, 0, 0, 0, 1, 0, 0, 0, 1,
		1, 0, 0, 0, 1, 0, 0, 0, 1,
		1, 0, 0, 0, 1, 0, 0, 0, 1,
		1, 0, 0, 0, 1, 0, 0, 0, 1,
		1, 0, 0, 0, 1, 0, 0, 0, 1,
		1, 0, 0, 0, 1, 0, 0, 0, 1,
		1, 0, 0, 0, 1, 0, 0, 0, 1,
		0.35f, 0, 0, 0, 0.35f, 0, 0, 0, 0.35f,
		0.35f, 0, 0, 0, 0.35f, 0, 0, 0, 0.35f,
		0.35f, 0, 0, 0, 0.35f, 0, 0, 0, 0.35f,
		0.35f, 0, 0, 0, 0.35f, 0, 0, 0, 0.35f,
		0.35f, 0, 0, 0, 0.35f, 0, 0, 0, 0.35f,
		0.35f, 0, 0, 0, 0.35f, 0, 0, 0, 0.35f,
		0.35f, 0, 0, 0, 0.35f, 0, 0, 0, 0.35f,
		0.35f, 0, 0, 0, 0.35f, 0, 0, 0, 0.35f,
		0.520643f, 0.0227124f, 0.251318f, 0.404902f, 0.0192711f, 0.946668f, 0.710671f, 0.25048f, 0.786821f,
		0.41937f, 0.499774f, 0.857077f, 0.521916f, 0.543856f, 0.140039f, 0.561032f, 0.274196f, 0.34467f,
		0.715893f, 0.19832f, 0.245533f, 0.164833f, 0.910501f, 0.320065f, 0.239277f, 0.249168f, 0.744281f,
		0.4381f, 0.9503f, 0.583883f, 0.777045f, 0.257262f, 0.459925f, 0.0807073f, 0.769432f, 0.967825f,
		0.0309147f, 0.90712f, 0.0912189f, 0.696239f, 0.504976f, 0.0455399f, 0.256359f, 0.532405f, 0.322289f,
		0.924194f, 0.285187f, 0.297847f, 0.480969f, 0.361325f, 0.298767f, 0.620412f, 0.98152f, 0.152036f,
		0.423817f, 0.759622f, 0.0888609f, 0.486834f, 0.069837f, 0.722758f, 0.713892f, 0.755031f, 0.628902f,
		0.21099f, 0.987307f, 0.43414f, 0.225448f, 0.222135f, 0.623808f, 0.272961f, 0.0864162f, 0.596983f,
		0.362117f, 0.0220786f, 0.490911f, 0.859905f, 0.935147f, 0.366719f, 0.836365f, 0.284114f, 0.853215f,
		0.487331f, 0.687829f, 0.980166f, 0.11218f, 0.909242f, 0.0290329f, 0.422952f, 0.519371f, 0.736445f,
		0.888796f, 0.969531f, 0.49149f, 0.870392f, 0.828304f, 0.455645f, 0.528452f, 0.588669f, 0.735908f,
		0.586928f, 0.101607f, 0.0324015f, 0.759026f, 0.548245f, 0.118927f, 0.343911f, 0.436551f, 0.138265f,
		0.644111f, 0.867071f, 0.990865f, 0.953564f, 0.605491f, 0.875281f, 0.849987f, 0.791824f, 0.695038f,
		0.66887f, 0.80717f, 0.147311f, 0.827346f, 0.904333f, 0.0791418f, 0.466998f, 0.912689f, 0.604654f,
		0.28054f, 0.0633181f, 0.323224f, 0.355726f, 0.629748f, 0.956864f, 0.832426f, 0.672333f, 0.138648f,
		0.972835f, 0.951291f, 0.311976f, 0.627886f, 0.13069f, 0.100668f, 0.0736188f, 0.0875155f, 0.256984f,
		0.894202f, 0.748943f, 0.746571f, 0.210137f, 0.718358f, 0.567179f, 0.676285f, 0.72612f, 0.201725f,
		0.868726f, 0.71123f, 0.281168f, 0.365701f, 0.611837f, 0.285751f, 0.0265524f, 0.654266f, 0.116882f,
		0.79339f, 0.506296f, 0.96275f, 0.988584f, 0.567172f, 0.795094f, 0.595384f, 0.887004f, 0.122479f,
		0.458105f, 0.934162f, 0.236615f, 0.811923f, 0.341087f, 0.831381f, 0.116003f, 0.402188f, 0.719541f,
		0.297123f, 0.186586f, 0.00570613f, 0.843262f, 0.5898f, 0.644393f, 0.369665f, 0.74979f, 0.85286f,
		0.888838f, 0.128372f, 0.279316f, 0.128813f, 0.767029f, 0.901857f, 0.994564f, 0.725961f, 0.338689f,
		0.833823f, 0.600164f, 0.517831f, 0.812785f, 0.917667f, 0.297175f, 0.527364f, 0.400797f, 0.805795f,
		0.367725f, 0.127923f, 0.8075f, 0.266564f, 0.467029f, 0.826142f, 0.59225f, 0.329163f, 0.477926f,
		0.371176f, 0.618904f, 0.987139f, 0.0859536f, 0.691681f, 0.434842f, 0.161771f, 0.124018f, 0.0195539f,
		0.36401f, 0.235788f, 0.787104f, 0.952125f, 0.683944f, 0.0994404f, 0.0368733f, 0.158937f, 0.236674f,
		0.358345f, 0.0861154f, 0.692114f, 0.697382f, 0.552795f, 0.200986f, 0.998865f, 0.934142f, 0.362999f,
		0.165289f, 0.0989368f, 0.387434f, 0.152076f, 0.519763f, 0.472191f, 0.943607f, 0.09432f, 0.286332f,
		0.225179f, 0.777137f, 0.989908f, 0.931878f, 0.31606f, 0.544445f, 0.649411f, 0.473452f, 0.583937f,
		0.33019f, 0.690941f, 0.113724f, 0.321522f, 0.30109f, 0.66992f, 0.384338f, 0.0439639f, 0.790953f,
		0.776293f, 0.700266f, 0.205705f, 0.420477f, 0.399689f, 0.751891f, 0.00717771f, 0.782775f, 0.798766f,
		0.311554f, 0.756531f, 0.125406f, 0.559647f, 0.91358f, 0.771515f, 0.748577f, 0.170088f, 0.141306f,
		0.832282f, 0.200414f, 0.815662f, 0.602823f, 0.745415f, 0.0951238f, 0.00443744f, 0.370268f, 0.725176f,
		0.277405f, 0.0941685f, 0.523856f, 0.590795f, 0.573825f, 0.960691f, 0.372524f, 0.129493f, 0.0555534f,
		0.50963f, 0.187052f, 0.367357f, 0.0742729f, 0.613573f, 0.414354f, 0.104138f, 0.569106f, 0.915303f,
		0.0634439f, 0.114006f, 0.697087f, 0.0509637f, 0.0561426f, 0.521955f, 0.0152459f, 0.775391f, 0.428071f,
		0.580879f, 0.489904f, 0.234415f, 0.527771f, 0.293188f, 0.636769f, 0.198587f, 0.531008f, 0.356116f,
		0.242226f, 0.740409f, 0.937673f, 0.579444f, 0.590133f, 0.395352f, 0.720755f, 0.314094f, 0.188345f,
		0.972132f, 0.590951f, 0.962899f, 0.118415f, 0.140093f, 0.633606f, 0.207918f, 0.113886f, 0.519524f,
		0.474761f, 0.908874f, 0.320076f, 0.128101f, 0.979988f, 0.156734f, 0.0923789f, 0.880038f, 0.802339f,
		0.654764f, 0.728632f, 0.568274f, 0.370185f, 0.307231f, 0.558256f, 0.521725f, 0.810924f, 0.200951f,
		0.651893f, 0.358894f, 0.693481f, 0.779626f, 0.893327f, 0.998629f, 0.911216f, 0.7543f, 0.311207f,
		0.95554f, 0.409707f, 0.862113f, 0.535271f, 0.349181f, 0.0390328f, 0.559026f, 0.909052f, 0.000327772f,
		0.404979f, 0.971619f, 0.781072f, 0.254931f, 0.168979f, 0.746042f, 0.0163063f, 0.841278f, 0.492911f,
		0.487119f, 0.532793f, 0.900794f, 0.256963f, 0.608284f, 0.257456f, 0.832431f, 0.955518f, 0.198655f,
		0.401213f, 0.359711f, 0.102809f, 0.0781677f, 0.249421f, 0.581417f, 0.878028f, 0.596802f, 0.790322f,
		0.394662f, 0.924365f, 0.297382f, 0.424487f, 0.428222f, 0.0504178f, 0.740887f, 0.596828f, 0.871441f,
		0.800913f, 0.587855f, 0.611199f, 0.463175f, 0.880588f, 0.547968f, 0.66794f, 0.685165f, 0.57688f,
		0.976944f, 0.588149f, 0.503932f, 0.549357f, 0.661278f, 0.464039f, 0.66517f, 0.316831f, 0.0303706f,
		0.788752f, 0.599136f, 0.0909839f, 0.851882f, 0.481792f, 0.052365f, 0.903292f, 0.570519f, 0.550525f,
		0.504267f, 0.608395f, 0.0637689f, 0.505807f, 0.193483f, 0.870505f, 0.0221698f, 0.496585f, 0.111787f,
		0.87611f, 0.651191f, 0.0857647f, 0.924414f, 0.974275f, 0.276202f, 0.466376f, 0.222165f, 0.578479f,
		0.877945f, 0.831223f, 0.0450919f, 0.214676f, 0.263458f, 0.618934f, 0.457222f, 0.246939f, 0.13779f,
		0.0282853f, 0.804414f, 0.685024f, 0.0915762f, 0.318291f, 0.224083f, 0.624383f, 0.304855f, 0.628726f,
		0.0221841f, 0.201672f, 0.780919f, 0.239676f, 0.818365f, 0.0718277f, 0.806745f, 0.4538f, 0.156149f,
		0.558087f, 0.313727f, 0.71475f, 0.251147f, 0.132566f, 0.662279f, 0.0103248f, 0.500025f, 0.520758f,
		0.380525f, 0.74988f, 0.365758f, 0.509431f, 0.829917f, 0.467508f, 0.887737f, 0.232667f, 0.227199f,
		0.64082f, 0.656539f, 0.458976f, 0.118464f, 0.0237956f, 0.486476f, 0.235979f, 0.224702f, 0.542421f,
		0.200237f, 0.180076f, 0.496747f, 0.00430347f, 0.0600218f, 0.993688f, 0.226868f, 0.637551f, 0.589712f,
		0.531601f, 0.829321f, 0.515012f, 0.689118f, 0.219158f, 0.0940731f, 0.198204f, 0.618056f, 0.329565f,
		0.625022f, 0.617491f, 0.180081f, 0.348768f, 0.82826f, 0.805169f, 0.239331f, 0.946657f, 0.576843f,
		0.425389f, 0.293695f, 0.201379f, 0.83667f, 0.439389f, 0.0382358f, 0.971805f, 0.890589f, 0.559929f,
		0.485052f, 0.226578f, 0.812773f, 0.505264f, 0.79858f, 0.544856f, 0.869584f, 0.121113f, 0.435041f,
		0.650536f, 0.85549f, 0.0245672f, 0.398691f, 0.79577f, 0.195309f, 0.26119f, 0.14189f, 0.797712f,
		0.579084f, 0.413172f, 0.858129f, 0.939198f, 0.557455f, 0.0830819f, 0.772611f, 0.754257f, 0.812641f,
		0.269266f, 0.782017f, 0.668374f, 0.777096f, 0.792946f, 0.830731f, 0.793292f, 0.166222f, 0.100147f,
		0.876922f, 0.276656f, 0.0987145f, 0.281124f, 0.198392f, 0.175454f, 0.986548f, 0.503781f, 0.614568f,
		0.442933f, 0.498453f, 0.42923f, 0.148727f, 0.564425f, 0.82545f, 0.418755f, 0.375626f, 0.85107f,
		0.83865f, 0.228017f, 0.158304f, 0.462783f, 0.315445f, 0.461605f, 0.955961f, 0.398656f, 0.153426f,
		0.929713f, 0.964296f, 0.949445f, 0.963796f, 0.301274f, 0.668396f, 0.842813f, 0.313689f, 0.456873f,
		0.426122f, 0.849068f, 0.49556f, 0.614552f, 0.922741f, 0.0115287f, 0.721641f, 0.292288f, 0.00523035f,
		0.757397f, 0.238789f, 0.0998681f, 0.296046f, 0.81877f, 0.482061f, 0.807112f, 0.725405f, 0.753475f,
		0.31726f, 0.96102f, 0.778193f, 0.515558f, 0.574123f, 0.881345f, 0.110429f, 0.818204f, 0.939234f,
		0.987285f, 0.96694f, 0.373858f, 0.444595f, 0.503499f, 0.757264f, 0.903807f, 0.823076f, 0.829222f,
		0.31973f, 0.799977f, 0.0769521f, 0.462343f, 0.696496f, 0.659913f, 0.367233f, 0.59787f, 0.736685f,
		0.126635f, 0.19819f, 0.661052f, 0.16328f, 0.498686f, 0.818349f, 0.086843f, 0.676035f, 0.427997f,
		0.856602f, 0.456753f, 0.829606f, 0.827364f, 0.398371f, 0.0429127f, 0.857263f, 0.0531367f, 0.805169f,
		0.622407f, 0.811645f, 0.310326f, 0.241334f, 0.795906f, 0.359206f, 0.72657f, 0.610263f, 0.762972f,
		0.154424f, 0.871387f, 0.0699211f, 0.272157f, 0.750185f, 0.491876f, 0.065452f, 0.0996625f, 0.935215f,
		0.446057f, 0.626001f, 0.168493f, 0.732351f, 0.371656f, 0.278478f, 0.698676f, 0.166497f, 0.323832f,
		0.16023f, 0.017112f, 0.980462f, 0.925277f, 0.857562f, 0.0151172f, 0.663758f, 0.0700159f, 0.0301436f,
		0.830635f, 0.075168f, 0.0587445f, 0.734044f, 0.0486698f, 0.913913f, 0.899506f, 0.069936f, 0.747241f,
		0.4213f, 0.129988f, 0.878506f, 0.396422f, 0.782106f, 0.320763f, 0.0489925f, 0.0204567f, 0.932468f,
		0.248066f, 0.338049f, 0.980579f, 0.0166563f, 0.776577f, 0.436277f, 0.096566f, 0.586235f, 0.250935f,
		0.252768f, 0.367929f, 0.832211f, 0.568896f, 0.931525f, 0.387573f, 0.0715725f, 0.621577f, 0.645227f,
		0.577878f, 0.967671f, 0.344483f, 0.0849481f, 0.789609f, 0.510477f, 0.992265f, 0.874716f, 0.496346f,
		0.869768f, 0.124984f, 0.56634f, 0.651146f, 0.607965f, 0.866855f, 0.0953233f, 0.633072f, 0.949746f,
		0.194975f, 0.264466f, 0.528521f, 0.882346f, 0.689586f, 0.940496f, 0.59432f, 0.893492f, 0.662669f,
		0.452568f, 0.13327f, 0.52997f, 0.30248f, 0.677981f, 0.664288f, 0.195069f, 0.369203f, 0.0677825f,
		0.857862f, 0.113165f, 0.881989f, 0.108224f, 0.11142f, 0.242903f, 0.923583f, 0.66295f, 0.386046f,
		0.875f, 0.917315f, 0.183098f, 0.411109f, 0.0470179f, 0.736449f, 0.4843f, 0.0386098f, 0.139198f,
		0.281885f, 0.393752f, 0.270645f, 0.758243f, 0.682393f, 0.721181f, 0.0871171f, 0.0892393f, 0.661774f,
		0.710648f, 0.200916f, 0.631902f, 0.901797f, 0.0933125f, 0.557962f, 0.311426f, 0.780639f, 0.491221f,
		0.972465f, 0.449748f, 0.868409f, 0.590519f, 0.524292f, 0.514068f, 0.463618f, 0.897181f, 0.855578f,
		0.966538f, 0.328141f, 0.227367f, 0.103953f, 0.936987f, 0.0151045f, 0.447349f, 0.100356f, 0.339617f,
		0.420638f, 0.856489f, 0.321358f, 0.614714f, 0.0592269f, 0.836418f, 0.859319f, 0.688564f, 0.92906f,
		0.0900997f, 0.845989f, 0.93764f, 0.467594f, 0.608119f, 0.0881119f, 0.468066f, 0.0795336f, 0.359514f,
		0.513898f, 0.3573f, 0.619364f, 0.29623f, 0.855836f, 0.128509f, 0.104001f, 0.453015f, 0.928457f,
		0.768058f, 0.192599f, 0.292531f, 0.0344313f, 0.66978f, 0.936939f, 0.998312f, 0.420276f, 0.822518f,
		0.116618f, 0.159051f, 0.185804f, 0.380808f, 0.33857f, 0.504723f, 0.0881964f, 0.854314f, 0.684523f,
		0.929677f, 0.194848f, 0.123159f, 0.857129f, 0.35805f, 0.557868f, 0.40652f, 0.744052f, 0.62557f,
		0.677233f, 0.671387f, 0.395753f, 0.461236f, 0.229644f, 0.205188f, 0.384446f, 0.736109f, 0.470506f,
		0.81209f, 0.228349f, 0.545617f, 0.898571f, 0.288337f, 0.144114f, 0.282391f, 0.451296f, 0.396541f,
		0.304525f, 0.219186f, 0.522225f, 0.340464f, 0.873487f, 0.241475f, 0.601216f, 0.125668f, 0.908432f,
		0.949059f, 0.856115f, 0.366398f, 0.670812f, 0.178851f, 0.380891f, 0.62556f, 0.66582f, 0.318581f,
		0.20306f, 0.160372f, 0.261225f, 0.678637f, 0.213662f, 0.0311836f, 0.335429f, 0.746307f, 0.660923f,
		0.977625f, 0.80036f, 0.964161f, 0.488311f, 0.345297f, 0.574249f, 0.026376f, 0.842607f, 0.531147f,
		0.579516f, 0.0839241f, 0.673116f, 0.607957f, 0.33216f, 0.00409965f, 0.55199f, 0.132121f, 0.897562f,
		0.320536f, 0.187267f, 0.894589f, 0.915221f, 0.394729f, 0.648809f, 0.712971f, 0.926536f, 0.591511f,
		0.565713f, 0.683051f, 0.0481479f, 0.611127f, 0.0831488f, 0.541202f, 0.33693f, 0.458179f, 0.288948f,
		0.00312662f, 0.819984f, 0.95343f, 0.0649215f, 0.356546f, 0.770275f, 0.346772f, 0.822183f, 0.267408f,
		0.535523f, 0.715216f, 0.921771f, 0.986854f, 0.965521f, 0.116363f, 0.668796f, 0.160807f, 0.552573f
	};
	// clang-format on

	template<typename Scalar>
	Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> createProblems(size_t nr_problems)
	{
		Vcl::Core::InterleavedArray<Scalar, 3, 3, -1> F(nr_problems);
	
#ifdef USE_PREGEN_DATA
		for (size_t i = 0; i < nr_problems; i++)
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
		for (size_t i = 0; i < nr_problems; i++)
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

		for (size_t i = 0; i < nr_problems; i++)
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

	for (size_t i = 0; i < stride / width; i++)
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

	for (size_t i = 0; i < stride / width; i++)
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

	for (size_t i = 0; i < stride / width; i++)
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
