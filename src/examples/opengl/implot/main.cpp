/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2020 Basil Fierz
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

#include "../common/imguiapp.h"

// ImPlot
#include <implot.h>

// VCL

class DemoImPlotApplication final : public ImGuiApplication
{
public:
	DemoImPlotApplication(const char* title)
	: ImGuiApplication(title)
	{
		ImPlot::CreateContext();
	}
	virtual ~DemoImPlotApplication()
	{
		ImPlot::DestroyContext();
	}

private:
	void updateFrame() override
	{
		ImGuiApplication::updateFrame();

		if (show_demo_window)
			ImPlot::ShowDemoWindow(&show_demo_window);
	}
	void renderFrame() override
	{
		glClearBufferfv(GL_COLOR, 0, (float*) &clear_color);
		glClearBufferfi(GL_DEPTH_STENCIL, 0, 1.0f, 0);

		ImGuiApplication::renderFrame();
	}

	bool show_demo_window = true;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
};

int main(int argc, char** argv)
{
	DemoImPlotApplication app{"ImGui Demo"};
	return app.run();
}
