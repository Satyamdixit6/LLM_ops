// llm_pipeline/adaptations/custom_cuda_kernel/custom_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename T>
__global__ void clamped_relu_forward_kernel(const T* input, T* output, int size, T clamp_val) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        T val = input[index];
        if (val < 0) {
            output[index] = 0;
        } else if (val > clamp_val) {
            output[index] = clamp_val;
        } else {
            output[index] = val;
        }
    }
}

template <typename T>
__global__ void clamped_relu_backward_kernel(const T* grad_output, const T* input, T* grad_input, int size, T clamp_val) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        T val = input[index];
        if (val > 0 && val < clamp_val) { // Only pass gradient if within active, non-clamped region
            grad_input[index] = grad_output[index];
        } else {
            grad_input[index] = 0;
        }
    }
}

torch::Tensor clamped_relu_forward(torch::Tensor input, float clamp_val) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    auto output = torch::empty_like(input);
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "clamped_relu_forward", ([&] {
        clamped_relu_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size,
            static_cast<scalar_t>(clamp_val)
        );
    }));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in clamped_relu_forward: ", cudaGetErrorString(err));
    return output;
}

torch::Tensor clamped_relu_backward(torch::Tensor grad_output, torch::Tensor input, float clamp_val) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output tensor must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output tensor must be contiguous");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor for backward");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous for backward");

    auto grad_input = torch::empty_like(input);
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "clamped_relu_backward", ([&] {
        clamped_relu_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            size,
            static_cast<scalar_t>(clamp_val)
        );
    }));
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in clamped_relu_backward: ", cudaGetErrorString(err));
    return grad_input;
}

// Binding to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &clamped_relu_forward, "Clamped ReLU forward (CUDA)");
    m.def("backward", &clamped_relu_backward, "Clamped ReLU backward (CUDA)");
}