/*
 * This file is part of the Visual Computing Library (VCL) release under the
 * MIT license.
 *
 * Copyright (c) 2014-2015 Basil Fierz
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
#include <vcl/compute/opencl/device.h>

// C++ standard library
#include <iostream>
#include <vector>

// VCL
#include <vcl/core/contract.h>

namespace Vcl { namespace Compute { namespace OpenCL
{
	Device::Device(cl_device_id dev)
	: _device(dev)
	, _capMajor(0)
	, _capMinor(0)
	{
		cl_platform_id platform_id;
		VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(platform_id), &platform_id, 0));

		// Read the device name
		std::vector<char> buffer(1024, 0);
		VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_NAME, buffer.size(), buffer.data(), 0));
		_name = buffer.data();

		// Read the compute capability
		VCL_CL_SAFE_CALL(clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, buffer.size(), buffer.data(), 0));
		std::string version = buffer.data();
		auto start = version.find(" ");
		auto delim = version.find(".");
		if (start != version.npos && delim != version.npos)
		{
			_capMajor = stoi(version.substr(start, delim));
			_capMinor = stoi(version.substr(delim + 1, version.length()));
		}

		// Read the number of compute units
		VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(_nrComputeUnits), &_nrComputeUnits, 0));

		//cl_device_type dev_type;
		//cl_device_fp_config fp_config;
		//cl_device_mem_cache_type cache_type;
		//cl_device_local_mem_type mem_type;
		//cl_uint param_uint;
		//cl_ulong param_ulong;
		//cl_bool param_bool;
		//size_t param_size_t_n[100];
		//size_t param_size_t;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, 0));
		//std::cout << "CL_DEVICE_TYPE: " << dev_type << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_VENDOR_ID, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_VENDOR_ID: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) *param_uint, param_size_t_n, 0));
		//std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: " << param_size_t_n[0] << ", " << param_size_t_n[1] << ", " << param_size_t_n[2] << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &param_size_t, 0));
		//std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << param_size_t << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << param_uint<< std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(param_bool), &param_bool, 0));
		//std::cout << "CL_DEVICE_IMAGE_SUPPORT: " << param_bool << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_MAX_READ_IMAGE_ARGS: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_MAX_WRITE_IMAGE_ARGS: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(param_size_t), &param_size_t, 0));
		//std::cout << "CL_DEVICE_IMAGE2D_MAX_WIDTH: " << param_size_t << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(param_size_t), &param_size_t, 0));
		//std::cout << "CL_DEVICE_IMAGE2D_MAX_HEIGHT: " << param_size_t << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(param_size_t), &param_size_t, 0));
		//std::cout << "CL_DEVICE_IMAGE3D_MAX_WIDTH: " << param_size_t << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(param_size_t), &param_size_t, 0));
		//std::cout << "CL_DEVICE_IMAGE3D_MAX_HEIGHT: " << param_size_t << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(param_size_t), &param_size_t, 0));
		//std::cout << "CL_DEVICE_IMAGE3D_MAX_DEPTH: " << param_size_t << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_SAMPLERS, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_MAX_SAMPLERS: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(param_size_t), &param_size_t, 0));
		//std::cout << "CL_DEVICE_MAX_PARAMETER_SIZE: " << param_size_t << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_MEM_BASE_ADDR_ALIGN: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fp_config), &fp_config, 0));
		//std::cout << "CL_DEVICE_SINGLE_FP_CONFIG: " << fp_config << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cache_type), &cache_type, 0));
		//std::cout << "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: " << cache_type << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(param_ulong), &param_ulong, 0));
		//std::cout << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: " << param_ulong << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(param_ulong), &param_ulong, 0));
		//std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE: " << param_ulong << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(param_ulong), &param_ulong, 0));
		//std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << param_ulong << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(param_uint), &param_uint, 0));
		//std::cout << "CL_DEVICE_MAX_CONSTANT_ARGS: " << param_uint << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(mem_type), &mem_type, 0));
		//std::cout << "CL_DEVICE_LOCAL_MEM_TYPE: " << mem_type << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(param_ulong), &param_ulong, 0));
		//std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: " << param_ulong << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(param_bool), &param_bool, 0));
		//std::cout << "CL_DEVICE_ERROR_CORRECTION_SUPPORT: " << param_bool << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(param_size_t), &param_size_t, 0));
		//std::cout << "CL_DEVICE_PROFILING_TIMER_RESOLUTION: " << param_size_t << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_ENDIAN_LITTLE, sizeof(param_bool), &param_bool, 0));
		//std::cout << "CL_DEVICE_ENDIAN_LITTLE: " << param_bool << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(param_bool), &param_bool, 0));
		//std::cout << "CL_DEVICE_AVAILABLE: " << param_bool << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_COMPILER_AVAILABLE, sizeof(param_bool), &param_bool, 0));
		//std::cout << "CL_DEVICE_COMPILER_AVAILABLE: " << param_bool << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_EXECUTION_CAPABILITIES, 1024, buffer, 0));
		//std::cout << "CL_DEVICE_EXECUTION_CAPABILITIES: " << stoi(std::string(buffer)) << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_QUEUE_PROPERTIES, 1024, buffer, 0));
		//std::cout << "CL_DEVICE_QUEUE_PROPERTIES: " << stoi(std::string(buffer)) << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_PROFILE, 1024, buffer, 0));
		//std::cout << "CL_DEVICE_PROFILE: " << stoi(std::string(buffer)) << std::endl;
		//
		//VCL_CL_SAFE_CALL(clGetDeviceInfo(dev, CL_DEVICE_VERSION, 1024, buffer, 0));
		//std::cout << "CL_DEVICE_VERSION: " << stoi(std::string(buffer)) << std::endl;
	}
}}}
