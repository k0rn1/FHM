#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "./pfc_cuda_device_info.h"
#include "../CudaLib/pfc_cuda_exception.h"
#include <iostream>

using namespace std::literals;

__global__ void cs_kernel(char * const dp_dst, char * const dp_src ,int const size) {
	auto const i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		dp_dst[i] = dp_src[i];
	}
}

int main()
{
	try {
		int count { 0 };
		cudaGetDeviceCount(&count);

		if (count > 0)
		{
			cudaSetDevice(0);
			auto const  dev_info = pfc::cuda::get_device_info();
			auto const  dev_prop = pfc::cuda::get_device_props();

			std::cout << "device: " << dev_prop.name << std::endl;
			std::cout << "compute capability: " << dev_info.cc_major << '.' << dev_info.cc_minor << std::endl;

			auto const text = "hello world"s;
			auto const size = std::size(text) + 1;
			auto const threads_in_block = 32;
			auto const blocks_in_grid = (size + threads_in_block - 1) / threads_in_block;
			auto const * const hp_src = text.c_str();
			auto * hp_dst = new char [size] {0};

			char * dp_src = nullptr;
			cudaMalloc(&dp_src, size);

			char * dp_dst = nullptr;
			cudaMalloc(&dp_dst, size);

			cudaMemcpy(dp_src, hp_src, size, cudaMemcpyHostToDevice);

			cs_kernel <<<blocks_in_grid , threads_in_block>>>(dp_dst, dp_src, size);

			cudaDeviceSynchronize();

			cudaGetLastError();

			cudaMemcpy(hp_dst, dp_dst, size, cudaMemcpyDeviceToHost);

			std::cout << "result: '" << hp_dst << "'" << std::endl;

			cudaFree(dp_src);
			cudaFree(dp_dst);

			delete[] hp_dst;
			hp_dst = nullptr;
		}
	}
	catch (std::exception const & x) {	
	std::cerr << x.what() << '\n';
	}
	return 0;
}

