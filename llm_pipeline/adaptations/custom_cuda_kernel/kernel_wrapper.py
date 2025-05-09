# llm_pipeline/adaptations/custom_cuda_kernel/kernel_wrapper.py
import torch
import torch.nn as nn
from torch.autograd import Function

# Import the compiled CUDA kernel
try:
    import custom_cuda_kernels_cpp # This name comes from setup.py
except ImportError:
    print("CUDA custom_cuda_kernels_cpp not found. Please compile it first.")
    print("Navigate to 'llm_pipeline/adaptations/custom_cuda_kernel/' and run 'python setup.py build_ext --inplace'")
    custom_cuda_kernels_cpp = None


class ClampedReLUFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, clamp_val):
        if not input_tensor.is_cuda or custom_cuda_kernels_cpp is None:
            # Fallback to PyTorch for CPU or if CUDA kernel not compiled
            ctx.clamp_val_cpu = clamp_val # Store for backward if needed on CPU
            output = torch.clamp(input_tensor, min=0, max=clamp_val)
            ctx.save_for_backward(input_tensor) # Save original input for CPU backward
            return output

        output = custom_cuda_kernels_cpp.forward(input_tensor, clamp_val)
        # Save original input (not the output of clamped_relu) for backward pass,
        # as the gradient depends on the original input value.
        ctx.save_for_backward(input_tensor)
        ctx.clamp_val = clamp_val
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda or custom_cuda_kernels_cpp is None:
            # Fallback for CPU
            input_tensor, = ctx.saved_tensors
            clamp_val = ctx.clamp_val_cpu
            grad_input = grad_output.clone()
            grad_input[(input_tensor <= 0) | (input_tensor >= clamp_val)] = 0
            return grad_input, None # None for clamp_val gradient

        input_tensor, = ctx.saved_tensors
        clamp_val = ctx.clamp_val
        grad_input = custom_cuda_kernels_cpp.backward(grad_output, input_tensor, clamp_val)
        return grad_input, None # Gradient for clamp_val is None

class CustomClampedReLU(nn.Module):
    def __init__(self, clamp_val=6.0):
        super().__init__()
        self.clamp_val = clamp_val

    def forward(self, x):
        return ClampedReLUFunction.apply(x, self.clamp_val)

    def __repr__(self):
        return f"{self.__class__.__name__}(clamp_val={self.clamp_val})"


if __name__ == "__main__":
    if custom_cuda_kernels_cpp is None:
        print("Skipping CUDA kernel test as it's not compiled.")
    else:
        print("Testing CustomClampedReLU with CUDA kernel...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cpu':
            print("CUDA not available, testing CPU fallback.")

        custom_relu_layer = CustomClampedReLU(clamp_val=5.0).to(device)
        # Test tensor
        x = torch.randn(2, 3, requires_grad=True, device=device) * 5
        print("Input tensor:\n", x)

        # Forward pass
        y = custom_relu_layer(x)
        print("Output tensor (custom clamped ReLU):\n", y)

        # For comparison: PyTorch's ReLU and clamp
        relu_torch = torch.nn.ReLU()
        y_torch_relu = relu_torch(x)
        y_torch_clamped = torch.clamp(y_torch_relu, max=5.0)
        print("Output tensor (PyTorch ReLU then clamp):\n", y_torch_clamped)

        # Backward pass
        if y.device.type == 'cuda': # Only test backward if CUDA kernel ran
            target = torch.randn_like(y)
            loss = (y - target).sum()
            loss.backward()
            print("Gradient of input (custom kernel):\n", x.grad)

            # Compare with PyTorch's backward
            x.grad = None # Reset grad
            y_ref = torch.clamp(torch.relu(x.detach().clone().requires_grad_(True)), max=5.0) # Recreate for grad
            loss_ref = (y_ref - target).sum()
            loss_ref.backward()
            print("Gradient of input (PyTorch):\n", x.grad) # This might not be an exact match to custom one due to how it's defined for clamp boundaries

    print("\nTesting CustomClampedReLU with CPU fallback (if not already)...")
    custom_relu_layer_cpu = CustomClampedReLU(clamp_val=5.0) #.to('cpu') implicit
    x_cpu = torch.randn(2, 3, requires_grad=True) * 5
    print("Input CPU tensor:\n", x_cpu)
    y_cpu = custom_relu_layer_cpu(x_cpu)
    print("Output CPU tensor:\n", y_cpu)
    target_cpu = torch.randn_like(y_cpu)
    loss_cpu = (y_cpu - target_cpu).sum()
    loss_cpu.backward()
    print("Gradient of input (CPU fallback):\n", x_cpu.grad)