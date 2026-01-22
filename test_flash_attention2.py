import dataclasses
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

try:
    import cupy as cp
    from cupy.cuda import compiler
    HAS_CUPY = True
except ImportError:
    print("Warning: CuPy not found. Install with: pip install cupy-cuda11x (or cupy-cuda12x)")
    HAS_CUPY = False
    sys.exit(1)


@dataclass
class TestConfig:
    name: str
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    test_backward: bool = False
    test_both: bool = False
    kernel_type: str = "fa2"
    seed: int = 42


@dataclass
class TestResult:
    config: TestConfig
    passed: bool
    max_abs_error: float
    mean_abs_error: float
    mse: float
    max_rel_error: float
    kernel_time_ms: float
    torch_time_ms: float
    speedup: float
    tflops: float
    bandwidth_gbps: float
    test_type: str = "forward"
    gradient_names: Optional[List[str]] = None
    error_message: str = ""


class FlashAttention2Tester:
    def __init__(self, stop_on_failure=True, tolerance=1e-3, test_mode='forward', 
                 save_results=False, output_dir='./experiment_results', use_gpu_reference=True):
        self.stop_on_failure = stop_on_failure
        self.tolerance = tolerance
        self.test_mode = test_mode
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        self.use_gpu_reference = use_gpu_reference
        self.results: List[TestResult] = []
        
        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Results will be saved to: {self.output_dir}")
        
        if not HAS_CUPY:
            raise RuntimeError("CuPy is required for inline CUDA compilation")
        
        self.forward_fa2_kernel_code = self._load_kernel_code('kernel_fa2_optimized.cu')
        self.backward_fa2_kernel_code = self._load_kernel_code('f-attn2-backward.cu')
        self.forward_fa1_kernel_code = self._load_kernel_code('f-attn.cu')
        self.forward_naive_kernel_code = self._load_kernel_code('vanilla-attn.cu')
        self.forward_fa2_naive = self._load_kernel_code('plain-attn.cu')
        
        self.forward_fa2_kernel_module = None
        self.backward_fa2_kernel_module = None
        self.forward_fa1_kernel_module = None
        self.forward_naive_kernel_module = None
        self.forward_fa2_naive_kernels = None
        
    def _load_kernel_code(self, filename: str) -> str:
        kernel_path = os.path.join(os.path.dirname(__file__), 'kernels', filename)
        
        if not os.path.exists(kernel_path):
            raise FileNotFoundError(f"Kernel file not found: {kernel_path}")
        
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()
        
        return kernel_code
    
    def _naive_fa2_compile_forward_kernel(self):
        if self.forward_fa2_naive_kernels is None:
            print("Compiling CUDA kernel...")
            try:
                self.forward_fa2_naive_kernels = cp.RawModule(
                    code=self.forward_fa2_naive,
                    options=('-std=c++14', '-DCUPY_INLINE_COMPILE'),
                    name_expressions=('flash_attention2_forward_kernel_wrapper',)
                )
                self.forward_fa_naive = self.forward_fa2_naive_kernels.get_function('flash_attention2_forward_kernel_wrapper')
                print("Kernel compiled successfully")
            except Exception as e:
                print(f"Kernel compilation failed: {e}")
                raise

    def _fa2_compile_forward_kernel(self):
        if self.forward_fa2_kernel_module is None:
            print("Compiling CUDA forward kernel...")
            try:
                self.forward_fa2_kernel_module = cp.RawModule(
                    code=self.forward_fa2_kernel_code,
                    options=('-std=c++14', '-DCUPY_INLINE_COMPILE'),
                    name_expressions=('flash_attention2_forward_kernel_wrapper',)
                )
                self.fa2_kernel = self.forward_fa2_kernel_module.get_function('flash_attention2_forward_kernel_wrapper')
                print("FA2 Forward kernel compiled successfully")
            except Exception as e:
                print(f"Forward kernel compilation failed: {e}")
                raise
    
    def _fa2_compile_backward_kernel(self):
        if self.backward_fa2_kernel_module is None:
            print("Compiling CUDA backward kernel...")
            try:
                self.backward_fa2_kernel_module = cp.RawModule(
                    code=self.backward_fa2_kernel_code,
                    options=('-std=c++14', '-DCUPY_INLINE_COMPILE'),
                    name_expressions=(
                        'flash_attention2_backward_kernel_wrapper',
                        'D_computation_reduction_kernel_wrapper'
                    )
                )
                self.backward_kernel = self.backward_fa2_kernel_module.get_function('flash_attention2_backward_kernel_wrapper')
                self.d_kernel = self.backward_fa2_kernel_module.get_function('D_computation_reduction_kernel_wrapper')
                print("Backward kernel compiled successfully")
            except Exception as e:
                print(f"Backward kernel compilation failed: {e}")
                raise
    
    def _fa1_compile_forward_kernel(self):
        if self.forward_fa1_kernel_module is None:
            print("Compiling CUDA FA1 kernel...")
            try:
                self.forward_fa1_kernel_module = cp.RawModule(
                    code=self.forward_fa1_kernel_code,
                    options=('-std=c++14', '-DCUPY_INLINE_COMPILE'),
                    name_expressions=('flash_attention_forward_kernel_wrapper',)
                )
                self.fa1_kernel = self.forward_fa1_kernel_module.get_function('flash_attention_forward_kernel_wrapper')
                print("FA1 kernel compiled successfully")
            except Exception as e:
                print(f"FA1 kernel compilation failed: {e}")
                raise
    
    def _naive_compile_forward_kernel(self):
        if self.forward_naive_kernel_module is None:
            print("Compiling CUDA Naive kernel...")
            try:
                self.forward_naive_kernel_module = cp.RawModule(
                    code=self.forward_naive_kernel_code,
                    options=('-std=c++14', '-DCUPY_INLINE_COMPILE'),
                    name_expressions=('vanilla_attention_kernel_wrapper',)
                )
                self.naive_kernel = self.forward_naive_kernel_module.get_function('vanilla_attention_kernel_wrapper')
                print("Naive kernel compiled successfully")
            except Exception as e:
                print(f"Naive kernel compilation failed: {e}")
                raise
    
    def generate_test_data(self, config: TestConfig):
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        Q = torch.rand(config.batch_size, config.num_heads, config.seq_len, 
                      config.head_dim, dtype=torch.float32)
        K = torch.rand(config.batch_size, config.num_heads, config.seq_len, 
                      config.head_dim, dtype=torch.float32)
        V = torch.rand(config.batch_size, config.num_heads, config.seq_len, 
                      config.head_dim, dtype=torch.float32)
        

        # for backward pass
        if config.test_backward or config.test_both:
            Q.requires_grad = True
            K.requires_grad = True
            V.requires_grad = True
        
        return Q, K, V
    
    def compute_reference(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute reference output using PyTorch"""
        head_dim = Q.shape[-1]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores = attn_scores / (head_dim ** 0.5)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_weights, V)
        
        return output

    def compute_reference_scaled_dot_product(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute reference using PyTorch scaled_dot_product_attention on GPU (CUDA cores only)"""
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            output = F.scaled_dot_product_attention(Q, K, V)
        return output
    
    def compute_reference_backward(self, O: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute reference output and gradients using PyTorch autograd"""
        grad_output = torch.ones_like(O) # simulating loss as L = sum(O), then dL/dO = 1
        
        O.backward(grad_output)
        
        gradients = {
            'dQ': Q.grad.clone(),
            'dK': K.grad.clone(),
            'dV': V.grad.clone()
        }
        
        return gradients
    
    def compute_reference_backward_timed(self, O: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[dict, float]:
        """Compute reference gradients with timing, including warm-up"""
        grad_output = torch.ones_like(O)
        O_warmup = self.compute_reference(Q, K, V)
        _ = self.compute_reference_backward(O_warmup, Q, K, V)
        Q.grad = None
        K.grad = None
        V.grad = None
        torch.cuda.synchronize() if Q.is_cuda else None
        
        start = time.time()
        O_timed = self.compute_reference(Q, K, V)
        gradients = self.compute_reference_backward(O_timed, Q, K, V)
        torch.cuda.synchronize() if Q.is_cuda else None
        elapsed_ms = (time.time() - start) * 1000
        
        return gradients, elapsed_ms

    def run_fa2_forward_kernel(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Run forward kernel"""
        config = Q.shape
        batch_size, num_heads, seq_len, head_dim = config
        
        Q_cp = cp.asarray(Q.detach().numpy())
        K_cp = cp.asarray(K.detach().numpy())
        V_cp = cp.asarray(V.detach().numpy())
        
        # allocate output
        output_cp = cp.zeros_like(Q_cp)
        logsumexp_cp = cp.zeros((batch_size, num_heads, seq_len), dtype=cp.float32)
        
        # kernel configuration
        BLOCK_SIZE_R = 32
        BLOCK_SIZE_C = 32
        HEAD_DIM = 64
        TM = 4
        TN = 4
        BK = 4
        
        T_r = (seq_len + BLOCK_SIZE_R - 1) // BLOCK_SIZE_R
        total_blocks = batch_size * num_heads * T_r
        
        num_threads = 256

        shared_mem = (BLOCK_SIZE_R * HEAD_DIM * 2 +  # q_buff + o_buff
                     BLOCK_SIZE_C * HEAD_DIM +       # kv_buff
                     BLOCK_SIZE_R * BLOCK_SIZE_C +   # s_buff
                     BLOCK_SIZE_R * 3) * 4           # logsumexp + maxes + maxes_prev
        
        # warm-up
        self.fa2_kernel(
            (total_blocks,), (num_threads,),
            (Q_cp, K_cp, V_cp, output_cp, logsumexp_cp,
             batch_size, num_heads, seq_len, head_dim),
            shared_mem=shared_mem
        )
        cp.cuda.Stream.null.synchronize()
        
        # timed runs
        num_runs = 10
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        for _ in range(num_runs):
            self.fa2_kernel(
                (total_blocks,), (num_threads,),
                (Q_cp, K_cp, V_cp, output_cp, logsumexp_cp,
                 batch_size, num_heads, seq_len, head_dim),
                shared_mem=shared_mem
            )
        end.record()
        end.synchronize()
        
        elapsed_ms = cp.cuda.get_elapsed_time(start, end) / num_runs
        
        output_np = cp.asnumpy(output_cp)
        logsumexp_np = cp.asnumpy(logsumexp_cp)

        return output_np, logsumexp_np, elapsed_ms

    def run_cuda_fa1_kernel(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Run FA1 kernel"""
        config = Q.shape
        batch_size, num_heads, seq_len, head_dim = config
        
        # convert to CuPy arrays
        Q_cp = cp.asarray(Q.detach().numpy())
        K_cp = cp.asarray(K.detach().numpy())
        V_cp = cp.asarray(V.detach().numpy())
        
        # allocate output
        output_cp = cp.zeros_like(Q_cp)
        logsumexp_cp = cp.zeros((batch_size, num_heads, seq_len), dtype=cp.float32)
        maxes_cp = cp.zeros((batch_size, num_heads, seq_len), dtype=cp.float32)
        
        # kernel configuration for FA1
        BLOCK_SIZE_R = 32
        BLOCK_SIZE_C = 32
        HEAD_DIM = 64
        
        total_blocks = batch_size * num_heads
        num_threads = 256
        
        shared_mem = (BLOCK_SIZE_R * HEAD_DIM * 2 +  # q_buff + o_buff
                     BLOCK_SIZE_C * HEAD_DIM * 2 +  # k_buff + v_buff
                     BLOCK_SIZE_R * BLOCK_SIZE_C +   # s_buff
                     BLOCK_SIZE_R * 2) * 4           # logsumexp + maxes
        
        # warm-up run
        self.fa1_kernel(
            (total_blocks,), (num_threads,),
            (Q_cp, K_cp, V_cp, output_cp, logsumexp_cp, maxes_cp,
             batch_size, num_heads, seq_len),
            shared_mem=shared_mem
        )
        cp.cuda.Stream.null.synchronize()
        
        # timed runs
        num_runs = 10
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        for _ in range(num_runs):
            self.fa1_kernel(
                (total_blocks,), (num_threads,),
                (Q_cp, K_cp, V_cp, output_cp, logsumexp_cp, maxes_cp,
                 batch_size, num_heads, seq_len),
                shared_mem=shared_mem
            )
        end.record()
        end.synchronize()
        
        elapsed_ms = cp.cuda.get_elapsed_time(start, end) / num_runs
        
        output_np = cp.asnumpy(output_cp)
        
        return output_np, elapsed_ms

    def run_naive_fa2_kernel(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[np.ndarray, float]:
        config = Q.shape
        batch_size, num_heads, seq_len, head_dim = config
        
        Q_cp = cp.asarray(Q.numpy())
        K_cp = cp.asarray(K.numpy())
        V_cp = cp.asarray(V.numpy())
        
        output_cp = cp.zeros_like(Q_cp)
        logsumexp_cp = cp.zeros((batch_size, num_heads, seq_len), dtype=cp.float32)
        
        BLOCK_SIZE_R = 32
        BLOCK_SIZE_C = 32
        HEAD_DIM = 64
        
        T_r = (seq_len + BLOCK_SIZE_R - 1) // BLOCK_SIZE_R
        total_blocks = batch_size * num_heads * T_r
        num_threads = 256
        
        shared_mem = (BLOCK_SIZE_R * HEAD_DIM * 2 +  # q_buff + o_buff
                     BLOCK_SIZE_C * HEAD_DIM * 2 +  # k_buff + v_buff
                     BLOCK_SIZE_R * BLOCK_SIZE_C +   # s_buff
                     BLOCK_SIZE_R * 3) * 4           # logsumexp + maxes + maxes_prev
        
        self.forward_fa_naive(
            (total_blocks,), (num_threads,),
            (Q_cp, K_cp, V_cp, output_cp, logsumexp_cp,
             batch_size, num_heads, seq_len),
            shared_mem=shared_mem
        )
        cp.cuda.Stream.null.synchronize()
        
        num_runs = 10
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        for _ in range(num_runs):
            self.forward_fa_naive(
                (total_blocks,), (num_threads,),
                (Q_cp, K_cp, V_cp, output_cp, logsumexp_cp,
                 batch_size, num_heads, seq_len),
                shared_mem=shared_mem
            )
        end.record()
        end.synchronize()
        
        elapsed_ms = cp.cuda.get_elapsed_time(start, end) / num_runs
        
        output_np = cp.asnumpy(output_cp)
        
        return output_np, elapsed_ms


    def run_cuda_naive_kernel(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Run naive kernel"""
        config = Q.shape
        batch_size, num_heads, seq_len, head_dim = config
        
        # convert to CuPy arrays
        Q_cp = cp.asarray(Q.detach().numpy())
        K_cp = cp.asarray(K.detach().numpy())
        V_cp = cp.asarray(V.detach().numpy())
        
        attn_cp = cp.zeros((batch_size, num_heads, seq_len, seq_len), dtype=cp.float32)

        # allocate output
        output_cp = cp.zeros_like(Q_cp)
        
        # kernel configuration for naive
        HEAD_DIM = 64
        total_blocks = batch_size * num_heads
        num_threads = 128
        
        # warm-up run
        self.naive_kernel(
            (total_blocks,), (num_threads,),
            (Q_cp, K_cp, V_cp, output_cp, attn_cp, batch_size, num_heads, seq_len)
            # shared_mem=shared_mem
        )
        cp.cuda.Stream.null.synchronize()
        
        # timed runs
        num_runs = 10
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        for _ in range(num_runs):
            self.naive_kernel(
                (total_blocks,), (num_threads,),
                (Q_cp, K_cp, V_cp, output_cp, attn_cp, batch_size, num_heads, seq_len)
                # shared_mem=shared_mem
            )
        end.record()
        end.synchronize()
        
        elapsed_ms = cp.cuda.get_elapsed_time(start, end) / num_runs
        
        output_np = cp.asnumpy(output_cp)
        
        return output_np, elapsed_ms
    
    def run_cuda_fa2_backward_kernel(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                  output: torch.Tensor, grad_output: torch.Tensor,
                                  logsumexp: np.ndarray) -> Tuple[dict, float]:
        """Run backward kernel"""
        config = Q.shape
        batch_size, num_heads, seq_len, head_dim = config
        
        # convert to CuPy arrays
        Q_cp = cp.asarray(Q.detach().numpy())
        K_cp = cp.asarray(K.detach().numpy())
        V_cp = cp.asarray(V.detach().numpy())
        output_cp = cp.asarray(output.detach().numpy())
        grad_output_cp = cp.asarray(grad_output.numpy())
        logsumexp_cp = cp.asarray(logsumexp)
        
        # allocate gradient outputs
        dQ_cp = cp.zeros_like(Q_cp)
        dK_cp = cp.zeros_like(K_cp)
        dV_cp = cp.zeros_like(V_cp)
        d_cp = cp.zeros((batch_size, num_heads, seq_len), dtype=cp.float32)
        
        # first, compute D (element-wise sum of output * grad_output)
        BLOCK_SIZE_R = 32
        BLOCK_SIZE_C = 32
        HEAD_DIM = 64
        
        # compute D using reduction kernel
        threads_per_block_d = 64 # next_power_of_2(head_dim)
        total_blocks_d = batch_size * num_heads * seq_len
        shared_mem_d = threads_per_block_d * 4  # sizeof(float)
        
        self.d_kernel(
            (total_blocks_d,), (threads_per_block_d,),
            (grad_output_cp, output_cp, batch_size, num_heads, seq_len, HEAD_DIM, d_cp),
            shared_mem=shared_mem_d
        )
        cp.cuda.Stream.null.synchronize()

        # kernel configuration for backward
        T_r = (seq_len + BLOCK_SIZE_R - 1) // BLOCK_SIZE_R
        T_c = (seq_len + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
        total_blocks = batch_size * num_heads * T_c
        num_threads = 256
        

        shared_mem = (
            BLOCK_SIZE_R * HEAD_DIM +      # q_buff
            BLOCK_SIZE_C * HEAD_DIM * 4 +  # k_buff, v_buff, d_k_buff, d_v_buff
            BLOCK_SIZE_R +                 # logsumexp/d
            BLOCK_SIZE_R * BLOCK_SIZE_C    # p_buff
        ) * 4  # sizeof(float)
        
        # warm-up run
        self.backward_kernel(
            (total_blocks,), (num_threads,),
            (Q_cp, K_cp, V_cp, output_cp, grad_output_cp, logsumexp_cp, d_cp,
             dQ_cp, dK_cp, dV_cp, batch_size, num_heads, seq_len, HEAD_DIM),
            shared_mem=shared_mem
        )
        cp.cuda.Stream.null.synchronize()
        
        # timed runs
        num_runs = 10
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        for _ in range(num_runs):
            # reset gradients
            dQ_cp.fill(0)
            dK_cp.fill(0)
            dV_cp.fill(0)
            
            self.backward_kernel(
                (total_blocks,), (num_threads,),
                (Q_cp, K_cp, V_cp, output_cp, grad_output_cp, logsumexp_cp, d_cp,
                 dQ_cp, dK_cp, dV_cp, batch_size, num_heads, seq_len, HEAD_DIM),
                shared_mem=shared_mem
            )
        end.record()
        end.synchronize()
        
        elapsed_ms = cp.cuda.get_elapsed_time(start, end) / num_runs
        
        gradients = {
            'dQ': cp.asnumpy(dQ_cp),
            'dK': cp.asnumpy(dK_cp),
            'dV': cp.asnumpy(dV_cp)
        }
        
        return gradients, elapsed_ms
    
    def compute_metrics(self, actual: np.ndarray, expected: np.ndarray, 
                       kernel_time: float, torch_time: float,
                       config: TestConfig) -> dict:
        # accuracy metrics
        abs_errors = np.abs(actual - expected)
        max_abs_error = np.max(abs_errors)
        mean_abs_error = np.mean(abs_errors)
        mse = np.mean((actual - expected) ** 2)
        
        # relative error
        rel_errors = np.where(np.abs(expected) > 1e-8,
                             abs_errors / np.abs(expected),
                             0.0)
        max_rel_error = np.max(rel_errors)
        
        # performance metrics
        # flops: 4 * batch * heads * seq^2 * head_dim (for Q@K^T and P@V)
        # plus softmax operations (negligible)
        flops = (2 * config.batch_size * config.num_heads * 
                config.seq_len * config.seq_len * config.head_dim * 2)
        tflops = (flops / (kernel_time * 1e-3)) / 1e12
        
        # bandwidth: reading Q,K,V and writing O
        bytes_transferred = (config.batch_size * config.num_heads * 
                           config.seq_len * config.head_dim * 4 * 4)  # 4 tensors, 4 bytes/float
        bandwidth_gbps = (bytes_transferred / (kernel_time * 1e-3)) / 1e9
        
        speedup = torch_time / kernel_time if kernel_time > 0 else 0.0
        
        return {
            'max_abs_error': max_abs_error,
            'mean_abs_error': mean_abs_error,
            'mse': mse,
            'max_rel_error': max_rel_error,
            'tflops': tflops,
            'bandwidth_gbps': bandwidth_gbps,
            'speedup': speedup
        }

    def _run_test_both(self, config: TestConfig) -> Tuple[TestResult, TestResult, TestResult]:
        """Run forward kernel, then use its output to run backward kernel"""
        print(f"\nRunning test: {config.name} (forward + backward pass)")
        print(f"Config: B={config.batch_size}, H={config.num_heads}, "
              f"S={config.seq_len}, D={config.head_dim}")

        try:
            Q, K, V = self.generate_test_data(config)

            print("  Computing PyTorch CPU reference (forward)...")
            torch_fwd_start = time.time()
            expected_output = self.compute_reference(Q, K, V)
            torch_fwd_time_ms = (time.time() - torch_fwd_start) * 1000

            Q_ref = Q.detach().clone().requires_grad_(True)
            K_ref = K.detach().clone().requires_grad_(True)
            V_ref = V.detach().clone().requires_grad_(True)

            print("  Computing PyTorch CPU reference (backward)...")
            expected_output_ref = self.compute_reference(Q_ref, K_ref, V_ref)
            torch_bwd_start = time.time()
            expected_grads = self.compute_reference_backward(expected_output_ref, Q_ref, K_ref, V_ref)
            torch_bwd_time_ms = (time.time() - torch_bwd_start) * 1000
            torch_total_time_ms = torch_fwd_time_ms + torch_bwd_time_ms

            fwd_flops = 2 * config.batch_size * config.num_heads * config.seq_len * config.seq_len * config.head_dim * 2
            bwd_flops = fwd_flops * 2.5  # backward is roughly 2.5x forward
            total_flops = fwd_flops + bwd_flops
            bytes_transferred = (config.batch_size * config.num_heads *
                               config.seq_len * config.head_dim * 4 * 4)  # 4 tensors, 4 bytes/float

            cpu_tflops = (total_flops / (torch_total_time_ms * 1e-3)) / 1e12
            cpu_bandwidth = (bytes_transferred / (torch_total_time_ms * 1e-3)) / 1e9
            cpu_config = dataclasses.replace(config)
            cpu_config.kernel_type = "PyTorch CPU"
            cpu_pytorch_result = TestResult(
                config=cpu_config, passed=True, max_abs_error=0.0, mean_abs_error=0.0,
                mse=0.0, max_rel_error=0.0, kernel_time_ms=torch_total_time_ms,
                torch_time_ms=torch_total_time_ms, speedup=1.0, tflops=cpu_tflops,
                bandwidth_gbps=cpu_bandwidth, test_type="both"
            )

            gpu_pytorch_result = None
            if self.use_gpu_reference:
                print("  Computing PyTorch GPU reference...")
                Q_gpu = Q.cuda().requires_grad_(True)
                K_gpu = K.cuda().requires_grad_(True)
                V_gpu = V.cuda().requires_grad_(True)

                # Warmup
                out_warmup = self.compute_reference_scaled_dot_product(Q_gpu, K_gpu, V_gpu)
                grad_warmup = torch.ones_like(out_warmup)
                out_warmup.backward(grad_warmup)
                torch.cuda.synchronize()

                # Timed run
                Q_gpu = Q.detach().cuda().requires_grad_(True)
                K_gpu = K.detach().cuda().requires_grad_(True)
                V_gpu = V.detach().cuda().requires_grad_(True)

                torch_gpu_start = time.time()
                out_gpu = self.compute_reference_scaled_dot_product(Q_gpu, K_gpu, V_gpu)
                grad_output_gpu = torch.ones_like(out_gpu)
                out_gpu.backward(grad_output_gpu)
                torch.cuda.synchronize()
                torch_gpu_time_ms = (time.time() - torch_gpu_start) * 1000

                # Compute GPU vs CPU error for output
                gpu_out_np = out_gpu.detach().cpu().numpy()
                gpu_fwd_abs_err = np.abs(gpu_out_np - expected_output.detach().numpy())
                gpu_fwd_max_err = np.max(gpu_fwd_abs_err)
                gpu_fwd_mean_err = np.mean(gpu_fwd_abs_err)

                # Compute GPU vs CPU error for gradients
                gpu_grads = {
                    'dQ': Q_gpu.grad.cpu().numpy(),
                    'dK': K_gpu.grad.cpu().numpy(),
                    'dV': V_gpu.grad.cpu().numpy()
                }
                all_gpu_grads = np.concatenate([gpu_grads['dQ'].flatten(), gpu_grads['dK'].flatten(), gpu_grads['dV'].flatten()])
                all_cpu_grads = np.concatenate([expected_grads['dQ'].numpy().flatten(), expected_grads['dK'].numpy().flatten(), expected_grads['dV'].numpy().flatten()])
                gpu_bwd_abs_err = np.abs(all_gpu_grads - all_cpu_grads)
                gpu_bwd_max_err = np.max(gpu_bwd_abs_err)
                gpu_bwd_mean_err = np.mean(gpu_bwd_abs_err)

                gpu_max_err = max(gpu_fwd_max_err, gpu_bwd_max_err)
                gpu_mean_err = (gpu_fwd_mean_err + gpu_bwd_mean_err) / 2

                gpu_tflops = (total_flops / (torch_gpu_time_ms * 1e-3)) / 1e12
                gpu_bandwidth = (bytes_transferred / (torch_gpu_time_ms * 1e-3)) / 1e9

                gpu_config = dataclasses.replace(config)
                gpu_config.kernel_type = "PyTorch GPU"
                gpu_pytorch_result = TestResult(
                    config=gpu_config, passed=True, max_abs_error=gpu_max_err, mean_abs_error=gpu_mean_err,
                    mse=0.0, max_rel_error=0.0, kernel_time_ms=torch_gpu_time_ms,
                    torch_time_ms=torch_total_time_ms, speedup=torch_total_time_ms/torch_gpu_time_ms,
                    tflops=gpu_tflops, bandwidth_gbps=gpu_bandwidth, test_type="both"
                )

            print("  Running CUDA FA2 forward kernel...")
            actual_output, logsumexp_np, fwd_kernel_time_ms = self.run_fa2_forward_kernel(Q, K, V)

            # Compute forward metrics
            fwd_metrics = self.compute_metrics(
                actual_output, expected_output.detach().numpy(),
                fwd_kernel_time_ms, torch_fwd_time_ms,
                config
            )

            fwd_passed = (fwd_metrics['max_abs_error'] < self.tolerance and
                         not np.isnan(actual_output).any() and
                         not np.isinf(actual_output).any())

            print("  Running CUDA FA2 backward kernel (using forward output)...")
            grad_output = torch.ones_like(expected_output)
            actual_output_torch = torch.from_numpy(actual_output)

            actual_grads, bwd_kernel_time_ms = self.run_cuda_fa2_backward_kernel(
                Q, K, V, actual_output_torch, grad_output, logsumexp_np
            )

            # Compute backward metrics
            all_actual_grads = np.concatenate([
                actual_grads['dQ'].flatten(),
                actual_grads['dK'].flatten(),
                actual_grads['dV'].flatten()
            ])
            all_expected_grads = np.concatenate([
                expected_grads['dQ'].numpy().flatten(),
                expected_grads['dK'].numpy().flatten(),
                expected_grads['dV'].numpy().flatten()
            ])

            bwd_metrics = self.compute_metrics(
                all_actual_grads, all_expected_grads,
                bwd_kernel_time_ms, torch_bwd_time_ms,
                config
            )

            bwd_passed = (bwd_metrics['max_abs_error'] < self.tolerance and
                         not any(np.isnan(actual_grads[n]).any() for n in ['dQ', 'dK', 'dV']) and
                         not any(np.isinf(actual_grads[n]).any() for n in ['dQ', 'dK', 'dV']))

            total_kernel_time_ms = fwd_kernel_time_ms + bwd_kernel_time_ms
            overall_passed = fwd_passed and bwd_passed
            max_error = max(fwd_metrics['max_abs_error'], bwd_metrics['max_abs_error'])
            mean_error = (fwd_metrics['mean_abs_error'] + bwd_metrics['mean_abs_error']) / 2

            combined_tflops = (total_flops / (total_kernel_time_ms * 1e-3)) / 1e12
            combined_speedup = torch_total_time_ms / total_kernel_time_ms if total_kernel_time_ms > 0 else 0.0

            result = TestResult(
                config=config,
                passed=overall_passed,
                max_abs_error=max_error,
                mean_abs_error=mean_error,
                mse=(fwd_metrics['mse'] + bwd_metrics['mse']) / 2,
                max_rel_error=max(fwd_metrics['max_rel_error'], bwd_metrics['max_rel_error']),
                kernel_time_ms=total_kernel_time_ms,
                torch_time_ms=torch_total_time_ms,
                speedup=combined_speedup,
                tflops=combined_tflops,
                bandwidth_gbps=(fwd_metrics['bandwidth_gbps'] + bwd_metrics['bandwidth_gbps']),
                test_type="both",
                gradient_names=['dQ', 'dK', 'dV']
            )

            if overall_passed:
                print(f"  PASSED (fwd_err={fwd_metrics['max_abs_error']:.2e}, bwd_err={bwd_metrics['max_abs_error']:.2e})")
                print(f"  Timing: fwd={fwd_kernel_time_ms:.3f}ms, bwd={bwd_kernel_time_ms:.3f}ms, total={total_kernel_time_ms:.3f}ms")
            else:
                error_msg = ""
                if not fwd_passed:
                    error_msg += f"Forward failed (err={fwd_metrics['max_abs_error']:.2e}). "
                if not bwd_passed:
                    error_msg += f"Backward failed (err={bwd_metrics['max_abs_error']:.2e}). "
                result.error_message = error_msg
                print(f"  FAILED: {error_msg}")

            return result, cpu_pytorch_result, gpu_pytorch_result

        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def run_test(self, config: TestConfig) -> TestResult:
        """Run a single test case"""
        # handle test_both mode by running both passes and summing metrics
        if config.test_both:
            return self._run_test_both(config)
        
        test_type = "backward" if config.test_backward else "forward"
        print(f"\nRunning test: {config.name} ({test_type} pass)")
        print(f"Config: B={config.batch_size}, H={config.num_heads}, "
              f"S={config.seq_len}, D={config.head_dim}")
        
        try:
            Q, K, V = self.generate_test_data(config)
            Q_gpu = Q.cuda().requires_grad_(True)
            K_gpu = K.cuda().requires_grad_(True)
            V_gpu = V.cuda().requires_grad_(True)
            
            if config.test_backward:
                # test backward pass
                expected_output = self.compute_reference(Q, K, V)
                expected_grads_1, torch_time_ms = self.compute_reference_backward_timed(expected_output, Q, K, V)
                print("  Computing PyTorch reference (forward + backward)...")
                expected_output = self.compute_reference(Q, K, V)
                torch_start = time.time()
                expected_output_timed = self.compute_reference(Q, K, V)
                expected_grads = self.compute_reference_backward(expected_output_timed, Q, K, V)
                torch_time_ms = (time.time() - torch_start) * 1000
                expected_grads_numpy = {k: v.numpy() for k, v in expected_grads.items()}
                
                # for CPU reference, compute metrics based on gradient magnitudes
                all_grads = np.concatenate([expected_grads_numpy['dQ'].flatten(), 
                                           expected_grads_numpy['dK'].flatten(), 
                                           expected_grads_numpy['dV'].flatten()])
                cpu_pytorch_metrics = self.compute_metrics(
                    all_grads, all_grads,
                    torch_time_ms, torch_time_ms,
                    config
                )
                cpu_config = dataclasses.replace(config)
                cpu_config.kernel_type = "PyTorch CPU"
                cpu_pytorch_results = TestResult(
                    config=cpu_config, passed=True, max_abs_error=0.0, mean_abs_error=0.0,
                    mse=0.0, max_rel_error=0.0, kernel_time_ms=torch_time_ms, torch_time_ms=torch_time_ms,
                    speedup=1.0, tflops=cpu_pytorch_metrics['tflops'], bandwidth_gbps=cpu_pytorch_metrics['bandwidth_gbps'], test_type="backward"
                )

                if self.use_gpu_reference:
                    print(f"Computing PyTorch GPU reference...")

                    expected_out_gpu = self.compute_reference_scaled_dot_product(Q_gpu, K_gpu, V_gpu)
                    grad_output = torch.ones_like(expected_out_gpu)
            
                    expected_out_gpu.backward(grad_output)
            
                    
                    torch.cuda.synchronize()

                    # Zero gradients and recreate tensors for timed pass
                    Q_gpu = Q.detach().cuda().requires_grad_(True)
                    K_gpu = K.detach().cuda().requires_grad_(True)
                    V_gpu = V.detach().cuda().requires_grad_(True)
                    Q_gpu.grad = None
                    K_gpu.grad = None
                    V_gpu.grad = None

                    # expected_grads_gpu = {
                    #     'dQ': Q.grad.clone(),
                    #     'dK': K.grad.clone(),
                    #     'dV': V.grad.clone()
                    # }

                    torch_start = time.time()
                    expected_out_gpu = self.compute_reference_scaled_dot_product(Q_gpu, K_gpu, V_gpu)
                    grad_output = torch.ones_like(expected_out_gpu)
            
                    expected_out_gpu.backward(grad_output)
            
                    
                    torch.cuda.synchronize()
                    torch_gpu_time_ms = (time.time() - torch_start) * 1000
                    expected_grads_gpu = {
                        'dQ': Q_gpu.grad.clone(),
                        'dK': K_gpu.grad.clone(),
                        'dV': V_gpu.grad.clone()
                    }
                    
                    all_grads_gpu = np.concatenate([expected_grads_gpu['dQ'].cpu().numpy().flatten(), 
                                                   expected_grads_gpu['dK'].cpu().numpy().flatten(), 
                                                   expected_grads_gpu['dV'].cpu().numpy().flatten()])
                    all_expected_grads = np.concatenate([expected_grads_1['dQ'].numpy().flatten(), 
                                                        expected_grads_1['dK'].numpy().flatten(), 
                                                        expected_grads_1['dV'].numpy().flatten()])
                    gpu_pytorch_metrics = self.compute_metrics(
                        all_grads_gpu, all_expected_grads,
                        torch_gpu_time_ms, torch_time_ms,
                        config
                    )
                    gpu_config = dataclasses.replace(config)
                    gpu_config.kernel_type = "PyTorch GPU"
                    gpu_pytorch_results = TestResult(
                        config=gpu_config, passed=True, max_abs_error=gpu_pytorch_metrics['max_abs_error'], mean_abs_error=gpu_pytorch_metrics['mean_abs_error'],
                        mse=0.0, max_rel_error=0.0, kernel_time_ms=torch_gpu_time_ms, torch_time_ms=torch_time_ms,
                        speedup=gpu_pytorch_metrics['speedup'], tflops=gpu_pytorch_metrics['tflops'], bandwidth_gbps=gpu_pytorch_metrics['bandwidth_gbps'], test_type="backward"
                    )
                else:
                    gpu_pytorch_results = None
                
                expected_grads = expected_grads_1

                
                # use PyTorch forward pass for output and compute logsumexp
                print("  Computing PyTorch forward for CUDA backward input...")
                Q_nograd = Q.detach().clone()
                K_nograd = K.detach().clone()
                V_nograd = V.detach().clone()
                
                # compute logsumexp manually for backward kernel
                config_shape = Q.shape
                batch_size, num_heads, seq_len, head_dim = config_shape
                
                # compute logsumexp from attention scores
                attn_scores = torch.matmul(Q_nograd, K_nograd.transpose(-2, -1)) / np.sqrt(head_dim)
                max_scores = torch.max(attn_scores, dim=-1, keepdim=True)[0]
                exp_scores = torch.exp(attn_scores - max_scores)
                sum_exp = torch.sum(exp_scores, dim=-1)
                logsumexp_torch = max_scores.squeeze(-1) + torch.log(sum_exp)
                logsumexp_np = logsumexp_torch.numpy()
                
                print("  Running CUDA backward kernel...")
                grad_output = torch.ones_like(expected_output)
                actual_grads, kernel_time_ms = self.run_cuda_fa2_backward_kernel(
                    Q, K, V, expected_output, grad_output, logsumexp_np
                )

                all_actual_grads = np.concatenate([actual_grads['dQ'].flatten(), 
                                                  actual_grads['dK'].flatten(), 
                                                  actual_grads['dV'].flatten()])
                all_expected_grads = np.concatenate([expected_grads['dQ'].numpy().flatten(), 
                                                    expected_grads['dK'].numpy().flatten(), 
                                                    expected_grads['dV'].numpy().flatten()])
                overall_metrics = self.compute_metrics(
                    all_actual_grads, all_expected_grads,
                    kernel_time_ms, torch_time_ms,
                    config
                )

                result = TestResult(
                    config=config, passed=True, max_abs_error=overall_metrics['max_abs_error'], mean_abs_error=overall_metrics['mean_abs_error'],
                    mse=overall_metrics['mse'], max_rel_error=overall_metrics['max_rel_error'], kernel_time_ms=kernel_time_ms, torch_time_ms=torch_time_ms,
                    speedup=overall_metrics['speedup'], tflops=overall_metrics['tflops'], bandwidth_gbps=overall_metrics['bandwidth_gbps'], test_type="backward", gradient_names=['dQ', 'dK', 'dV']
                )
                
                passed = (result.max_abs_error < self.tolerance and
                         not any(np.isnan(actual_grads[n]).any() for n in ['dQ', 'dK', 'dV']) and
                         not any(np.isinf(actual_grads[n]).any() for n in ['dQ', 'dK', 'dV']))
                
            else:
                # test forward pass
                print(f"  Kernel type: {config.kernel_type.upper()}")
                print("  Computing PyTorch CPU reference...")
                torch_start = time.time()
                expected = self.compute_reference(Q, K, V)
                torch_time_ms = (time.time() - torch_start) * 1000
                cpu_pytorch_metrics = self.compute_metrics(
                    expected.numpy(), expected.numpy(),
                    torch_time_ms, torch_time_ms,
                    config
                )
                cpu_config = dataclasses.replace(config)
                cpu_config.kernel_type = "PyTorch CPU"
                cpu_pytorch_results = TestResult(
                    config=cpu_config, passed=True, max_abs_error=cpu_pytorch_metrics['max_abs_error'], mean_abs_error=cpu_pytorch_metrics['mean_abs_error'],
                    mse=cpu_pytorch_metrics['mse'], max_rel_error=cpu_pytorch_metrics['max_rel_error'], kernel_time_ms=torch_time_ms,torch_time_ms=torch_time_ms,
                    speedup=cpu_pytorch_metrics['speedup'], tflops=cpu_pytorch_metrics['tflops'], bandwidth_gbps=cpu_pytorch_metrics['bandwidth_gbps'], test_type="forward"
                )
                if self.use_gpu_reference:
                    print(f"Computing PyTorch GPU reference...")
                    Q_gpu = Q.cuda()
                    K_gpu = K.cuda()
                    V_gpu = V.cuda()
                    _ = self.compute_reference_scaled_dot_product(Q_gpu, K_gpu, V_gpu)
                    torch.cuda.synchronize()
                    torch_start = time.time()
                    expected_gpu = self.compute_reference_scaled_dot_product(Q_gpu, K_gpu, V_gpu)
                    torch.cuda.synchronize()
                    torch_gpu_time_ms = (time.time() - torch_start) * 1000
                    gpu_pytorch_metrics = self.compute_metrics(
                        expected_gpu.cpu().numpy(), expected.numpy(), 
                        torch_gpu_time_ms, torch_time_ms,
                        config
                    )
                    gpu_config = dataclasses.replace(config)
                    gpu_config.kernel_type = "PyTorch GPU"
                    gpu_pytorch_results = TestResult(
                        config=gpu_config, passed=True, max_abs_error=gpu_pytorch_metrics['max_abs_error'], mean_abs_error=gpu_pytorch_metrics['mean_abs_error'], 
                        mse=gpu_pytorch_metrics['mse'], max_rel_error=gpu_pytorch_metrics['max_rel_error'], kernel_time_ms=torch_gpu_time_ms, torch_time_ms=torch_time_ms, 
                        speedup=gpu_pytorch_metrics['speedup'], tflops=gpu_pytorch_metrics['tflops'], bandwidth_gbps=gpu_pytorch_metrics['bandwidth_gbps'], test_type="forward"
                    )
                else:
                    gpu_pytorch_results = None

                if config.kernel_type == "fa2":
                    print("  Running CUDA FA2 kernel...")
                    actual, _, kernel_time_ms = self.run_fa2_forward_kernel(Q, K, V)
                elif config.kernel_type == "fa1":
                    print("  Running CUDA FA1 kernel...")
                    actual, kernel_time_ms = self.run_cuda_fa1_kernel(Q, K, V)
                elif config.kernel_type == "naive":
                    print("  Running CUDA Naive FA2 kernel...")
                    actual, kernel_time_ms = self.run_naive_fa2_kernel(Q, K, V)
                elif config.kernel_type == "naive-attn":
                    print("  Running CUDA Naive Attention kernel...")
                    actual, kernel_time_ms = self.run_cuda_naive_kernel(Q, K, V)
                else:
                    raise ValueError(f"Unknown kernel type: {config.kernel_type}")
                
                metrics = self.compute_metrics(
                    actual, expected.numpy(),
                    kernel_time_ms, torch_time_ms,
                    config
                )
                
                passed = (metrics['max_abs_error'] < self.tolerance and
                         not np.isnan(actual).any() and
                         not np.isinf(actual).any())
                
                result = TestResult(
                    config=config, passed=passed, max_abs_error=metrics['max_abs_error'], mean_abs_error=metrics['mean_abs_error'],
                    mse=metrics['mse'], max_rel_error=metrics['max_rel_error'], kernel_time_ms=kernel_time_ms, torch_time_ms=torch_time_ms,
                    speedup=metrics['speedup'], tflops=metrics['tflops'], bandwidth_gbps=metrics['bandwidth_gbps'], test_type="forward"
                )
            
            if passed:
                print(f"PASSED (max_error={result.max_abs_error:.2e})")
            else:
                error_msg = f"Max error {result.max_abs_error:.2e} exceeds tolerance {self.tolerance}"
                if config.test_backward:
                    if any(np.isnan(actual_grads[n]).any() for n in ['dQ', 'dK', 'dV']):
                        error_msg += " (contains NaN)"
                    if any(np.isinf(actual_grads[n]).any() for n in ['dQ', 'dK', 'dV']):
                        error_msg += " (contains Inf)"
                else:
                    if np.isnan(actual).any():
                        error_msg += " (contains NaN)"
                    if np.isinf(actual).any():
                        error_msg += " (contains Inf)"
                result.error_message = error_msg
                print(f"FAILED: {error_msg}")
            
            return result, cpu_pytorch_results, gpu_pytorch_results
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise 
    
    def run_all_tests(self, configs: List[TestConfig]):
        print("="*80)
        print("Flash Attention - Comprehensive Test Suite")
        print("NOTE: All implementations are CUDA cores only (no Tensor Cores)")
        print(f"Test Mode: {self.test_mode.upper()}")
        if self.test_mode == 'backward':
            print("Note: Using PyTorch forward pass, testing CUDA backward only")
        print("="*80)
        
        # compile appropriate kernels
        kernel_types = set(c.kernel_type for c in configs)
        # Need forward kernels for: forward mode, any forward test, or any 'both' test
        if self.test_mode in ['forward', 'both'] or any(not c.test_backward or c.test_both for c in configs):
            if 'fa2' in kernel_types:
                self._fa2_compile_forward_kernel()
            if 'fa1' in kernel_types:
                self._fa1_compile_forward_kernel()
            if 'naive' in kernel_types:
                self._naive_compile_forward_kernel()
            if 'naive-attn' in kernel_types:
                self._naive_fa2_compile_forward_kernel()
        # Need backward kernels for: backward mode, any backward test, or any 'both' test
        if self.test_mode in ['backward', 'both'] or any(c.test_backward or c.test_both for c in configs):
            self._fa2_compile_backward_kernel()
        
        test_done = set()
        for i, config in enumerate(configs, 1):
            print(f"\n[Test {i}/{len(configs)}]")
            result, result_cpu, result_gpu = self.run_test(config)
            if config.name in test_done:
                self.results.append(result)
            else:
                if result_gpu is not None:
                    self.results.extend([result, result_cpu, result_gpu])
                else:
                    self.results.extend([result, result_cpu])
            test_done.add(result.config.name)
            
            if not result.passed and self.stop_on_failure:
                print(f"\n{'='*80}")
                print("Stopping on first failure (stop_on_failure=True)")
                print(f"{'='*80}")
                break
        self.results.sort(key=lambda x: x.config.batch_size*x.config.num_heads*x.config.seq_len*x.config.head_dim)
        
        self.print_summary()        
        if self.save_results:
            self.save_results_to_files()
    
    def save_results_to_files(self):
        """Save experiment results to CSV and generate plots"""
        print(f"\n{'='*80}")
        print("SAVING RESULTS")
        print(f"{'='*80}")
        
        data = []
        for result in self.results:
            cfg = result.config
            data.append({
                'Test': cfg.name, 'Kernel': cfg.kernel_type.upper(), 'Type': result.test_type.upper()[:3], 'Batch': cfg.batch_size,
                'Heads': cfg.num_heads, 'SeqLen': cfg.seq_len, 'HeadDim': cfg.head_dim, 'Status': 'PASS' if result.passed else 'FAIL', 
                'MaxError': result.max_abs_error, 'MeanError': result.mean_abs_error, 'MSE': result.mse, 'MaxRelError': result.max_rel_error, 
                'KernelTime_ms': result.kernel_time_ms, 'TorchTime_ms': result.torch_time_ms, 'Speedup': result.speedup, 'TFLOPS': result.tflops,
                'Bandwidth_GBps': result.bandwidth_gbps, 'ErrorMessage': result.error_message
            })
        
        df = pd.DataFrame(data)
        
        csv_path = self.output_dir / 'experiment_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved results to: {csv_path}")
        
        self._generate_plots(df)
    
    def _generate_plots(self, df: pd.DataFrame):
        """Generate performance comparison plots"""
        print("\nGenerating plots...")
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        df_passed = df[df['Status'] == 'PASS'].copy()
        
        if len(df_passed) == 0:
            print("No passed tests to plot")
            return
        
        kernels = sorted(df_passed['Kernel'].unique())
        test_names = df_passed['Test'].unique()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(kernels)))
        kernel_colors = dict(zip(kernels, colors))
        
        if len(kernels) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            kernel_groups = df_passed.groupby('Kernel')
            
            ax = axes[0, 0]
            for kernel in kernels:
                kernel_data = df_passed[df_passed['Kernel'] == kernel]
                ax.plot(range(len(kernel_data)), kernel_data['KernelTime_ms'], 
                       marker='o', label=kernel, linewidth=2)
            ax.set_xlabel('Test Index', fontsize=11)
            ax.set_ylabel('Time (ms)', fontsize=11)
            ax.set_title('Execution Time by Kernel Type', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[0, 1]
            for kernel in kernels:
                kernel_data = df_passed[df_passed['Kernel'] == kernel]
                ax.plot(range(len(kernel_data)), kernel_data['Speedup'], 
                       marker='s', label=kernel, linewidth=2)
            ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Test Index', fontsize=11)
            ax.set_ylabel('Speedup vs PyTorch', fontsize=11)
            ax.set_title('Speedup by Kernel Type', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[1, 0]
            for kernel in kernels:
                kernel_data = df_passed[df_passed['Kernel'] == kernel]
                ax.plot(range(len(kernel_data)), kernel_data['TFLOPS'], 
                       marker='^', label=kernel, linewidth=2)
            ax.set_xlabel('Test Index', fontsize=11)
            ax.set_ylabel('TFLOPS', fontsize=11)
            ax.set_title('TFLOPS by Kernel Type', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[1, 1]
            for kernel in kernels:
                kernel_data = df_passed[df_passed['Kernel'] == kernel]
                ax.plot(range(len(kernel_data)), kernel_data['Bandwidth_GBps'], 
                       marker='d', label=kernel, linewidth=2)
            ax.set_xlabel('Test Index', fontsize=11)
            ax.set_ylabel('Bandwidth (GB/s)', fontsize=11)
            ax.set_title('Bandwidth by Kernel Type', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.suptitle('Performance Comparison Across Kernel Implementations\n(CUDA Cores Only - No Tensor Cores)', 
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            plot_path = self.output_dir / 'kernel_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {plot_path}")
            plt.close()
        
        if len(df_passed[df_passed['Test'].str.contains('SeqLen', case=False, na=False)]) > 0:
            self._generate_seqlen_plots(df_passed)
    
    def _generate_seqlen_plots(self, df: pd.DataFrame):
        """Generate sequence length scaling analysis plots"""
        print("\nGenerating sequence length scaling plots...")
        
        df_seqlen = df[df['Test'].str.contains('SeqLen', case=False, na=False)].copy()
        
        if len(df_seqlen) == 0:
            return
        
        df_seqlen = df_seqlen.sort_values('SeqLen')
        
        kernels = sorted(df_seqlen['Kernel'].unique())
        seq_lengths = sorted(df_seqlen['SeqLen'].unique())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax = axes[0, 0]
        for kernel in kernels:
            kernel_data = df_seqlen[df_seqlen['Kernel'] == kernel]
            ax.plot(kernel_data['SeqLen'], kernel_data['TFLOPS'], 
                   marker='o', linewidth=2, markersize=6, label=kernel)
        ax.set_xlabel('Sequence Length', fontsize=11)
        ax.set_ylabel('TFLOPS', fontsize=11)
        ax.set_title('TFLOPS vs Sequence Length', fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_xticks(seq_lengths)
        ax.set_xticklabels([str(s) for s in seq_lengths])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        
        ax = axes[0, 1]
        for kernel in kernels:
            kernel_data = df_seqlen[df_seqlen['Kernel'] == kernel]
            ax.plot(kernel_data['SeqLen'], kernel_data['Bandwidth_GBps'], 
                   marker='s', linewidth=2, markersize=6, label=kernel)
        ax.set_xlabel('Sequence Length', fontsize=11)
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=11)
        ax.set_title('Bandwidth vs Sequence Length', fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_xticks(seq_lengths)
        ax.set_xticklabels([str(s) for s in seq_lengths])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        
        ax = axes[1, 0]
        for kernel in kernels:
            kernel_data = df_seqlen[df_seqlen['Kernel'] == kernel]
            ax.plot(kernel_data['SeqLen'], kernel_data['Speedup'], 
                   marker='^', linewidth=2, markersize=6, label=kernel)
        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Sequence Length', fontsize=11)
        ax.set_ylabel('Speedup vs PyTorch', fontsize=11)
        ax.set_title('Speedup vs Sequence Length', fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_xticks(seq_lengths)
        ax.set_xticklabels([str(s) for s in seq_lengths])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        
        ax = axes[1, 1]
        for kernel in kernels:
            kernel_data = df_seqlen[df_seqlen['Kernel'] == kernel]
            ax.plot(kernel_data['SeqLen'], kernel_data['KernelTime_ms'], 
                   marker='d', linewidth=2, markersize=6, label=kernel)
        ax.set_xlabel('Sequence Length', fontsize=11)
        ax.set_ylabel('Execution Time (ms)', fontsize=11)
        ax.set_title('Execution Time vs Sequence Length', fontsize=12, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(seq_lengths)
        ax.set_xticklabels([str(s) for s in seq_lengths])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        
        fig.suptitle('Sequence Length Scaling Analysis\n(CUDA Cores Only - No Tensor Cores)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plot_path = self.output_dir / 'seqlen_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()

    def print_summary(self):
        """Print summary table of all results"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        headers = ["Test", "Kernel", "Type", "Config", "Status", "Max Err", "Mean Err", 
                  "Time (ms)", "TFLOPS", "BW (GB/s)"]
        
        rows = []
        for result in self.results:
            cfg = result.config
            config_str = f"B{cfg.batch_size}_H{cfg.num_heads}_S{cfg.seq_len}_D{cfg.head_dim}"
            status = "✓ PASS" if result.passed else "✗ FAIL"
            test_type = result.test_type.upper()
            kernel_type = cfg.kernel_type.upper()

            rows.append([
                cfg.name,
                kernel_type,
                test_type,
                config_str,
                status,
                f"{result.max_abs_error:.2e}",
                f"{result.mean_abs_error:.2e}",
                f"{result.kernel_time_ms:.6f}",
                f"{result.tflops:.2f}",
                f"{result.bandwidth_gbps:.2f}"
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed} ({100*passed/total:.1f}%)")
        print(f"Failed: {failed} ({100*failed/total:.1f}%)")
        
        kernel_types = set(r.config.kernel_type for r in self.results)
        if len(kernel_types) > 1:
            print("\n" + "="*80)
            print("SUMMARY BY KERNEL TYPE")
            print("="*80)
            for ktype in sorted(kernel_types):
                ktype_results = [r for r in self.results if r.config.kernel_type == ktype and r.passed]
                if ktype_results:
                    avg_speedup = np.mean([r.speedup for r in ktype_results])
                    avg_tflops = np.mean([r.tflops for r in ktype_results])
                    avg_bandwidth = np.mean([r.bandwidth_gbps for r in ktype_results])
                    print(f"\n{ktype.upper()}:")
                    print(f"  Average Speedup: {avg_speedup:.2f}x")
                    print(f"  Average TFLOPS: {avg_tflops:.2f}")
                    print(f"  Average Bandwidth: {avg_bandwidth:.2f} GB/s")
        
        if passed > 0:
            avg_speedup = np.mean([r.speedup for r in self.results if r.passed])
            avg_tflops = np.mean([r.tflops for r in self.results if r.passed])
            print(f"\nOverall Average Speedup: {avg_speedup:.2f}x")
            print(f"Overall Average Performance: {avg_tflops:.2f} TFLOPS")
        
        if failed > 0:
            print("\n" + "="*80)
            print("FAILED TESTS DETAIL")
            print("="*80)
            for result in self.results:
                if not result.passed:
                    print(f"\n{result.config.name} ({result.config.kernel_type.upper()} - {result.test_type}):")
                    print(f"  Error: {result.error_message}")
                    print(f"  Max abs error: {result.max_abs_error:.2e}")
                    print(f"  Max rel error: {result.max_rel_error:.2e}")
        
        print("\n" + "="*80)


def create_test_configs(test_mode='forward', kernel_type='fa2') -> List[TestConfig]:
    """Create a list of test configurations"""
    test_backward = (test_mode == 'backward')
    test_both = (test_mode == 'both')
    
    configs = [
        # Small tests
        TestConfig("Small-1", batch_size=1, num_heads=1, seq_len=128, head_dim=64, 
                  test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        TestConfig("Small-2", batch_size=2, num_heads=4, seq_len=256, head_dim=64, 
                  test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        TestConfig("Small-3", batch_size=2, num_heads=8, seq_len=256, head_dim=64, 
                  test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        
        # Medium tests
        TestConfig("Medium-1", batch_size=2, num_heads=8, seq_len=512, head_dim=64, 
                  test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        TestConfig("Medium-2", batch_size=4, num_heads=8, seq_len=512, head_dim=64, 
                  test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        
        # Large tests
        TestConfig("Large-1", batch_size=2, num_heads=8, seq_len=1024, head_dim=64, 
                  test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        TestConfig("Large-2", batch_size=4, num_heads=12, seq_len=1024, head_dim=64, 
                  test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        
        # Edge cases
        TestConfig("Edge-NonPowerOf2", batch_size=8, num_heads=16, seq_len=100, head_dim=64, 
                  test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        TestConfig("Edge-SmallSeq", batch_size=8, num_heads=16, seq_len=32, head_dim=64, 
                  test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        
        # Stress tests (uncomment if needed)
        TestConfig("Stress-1", batch_size=8, num_heads=16, seq_len=2048, head_dim=64, 
                  test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        # TestConfig("Stress-2", batch_size=16, num_heads=32, seq_len=2048, head_dim=64, 
        #           test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        # TestConfig("Stress-3", batch_size=32, num_heads=32, seq_len=2048, head_dim=64, 
        #           test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        # TestConfig("Stress-4", batch_size=64, num_heads=32, seq_len=2048, head_dim=64, 
        #           test_backward=test_backward, test_both=test_both, kernel_type=kernel_type),
        # TestConfig("Stress-5", batch_size=64, num_heads=64, seq_len=2048, head_dim=64, 
        #           test_backward=test_backward, kernel_type=kernel_type),
    ]
    
    return configs


def create_experiment_configs(mode) -> List[TestConfig]:
    """Create experiment configurations for comparison between implementations
    Tests all kernels on comprehensive test suite"""
    configs = []
    
    if mode == 'forward':
        kernel_types = ["naive", "naive-attn", "fa2"]
    elif mode == 'backward':
        kernel_types = ["fa2"]
    else:  # both
        kernel_types = ["fa2"]
    
    for kernel_type in kernel_types:
        configs.extend(create_test_configs(test_mode=mode, kernel_type=kernel_type))
    
    return configs


def create_sequence_length_experiment_configs(mode) -> List[TestConfig]:
    """Create configurations to test sequence length scaling
    Fixed: B=4, H=8, D=64, varying S from 128 to 2048"""
    configs = []
    
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    if mode == 'forward':
        kernel_types = ["naive", "naive-attn", "fa2"]
    elif mode == 'backward':
        kernel_types = ["fa2"]
    else:  # both
        kernel_types = ["fa2"]
    
    for seq_len in seq_lengths:
        for kernel_type in kernel_types:
            config = TestConfig(
                name=f"SeqLen-S{seq_len}-{kernel_type.upper()}",
                batch_size=4,
                num_heads=8,
                seq_len=seq_len,
                head_dim=64,
                test_backward=(mode == 'backward'),
                test_both=(mode == 'both'),
                kernel_type=kernel_type
            )
            configs.append(config)
    
    return configs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test Flash Attention CUDA kernels (forward and backward passes)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--mode', type=str, default='forward', 
                       choices=['forward', 'backward', 'both'],
                       help='Test mode: forward, backward, or both passes (default: forward)',
                       required=True)
    parser.add_argument('--kernel', type=str, default='fa2',
                       choices=['fa2', 'fa1', 'naive-attn', 'naive'],
                       help='Kernel type to test (default: fa2)')
    parser.add_argument('--experiment', action='store_true',
                       help='Run comprehensive experiment comparing all implementations on all test configs')
    parser.add_argument('--seqlen-experiment', action='store_true',
                       help='Run sequence length scaling experiment (S=128 to 2048)')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                       help='Error tolerance for passing tests (default: 1e-3)')
    parser.add_argument('--no-stop-on-failure', action='store_true',
                       help='Continue testing after first failure')
    parser.add_argument('--save-results', action='store_true',
                       help='Save experiment results to CSV and generate plots')
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                       help='Output directory for results (default: ./experiment_results)')
    parser.add_argument('--no-gpu-reference', action='store_true',
                       help='Disable GPU reference computations (only use CPU reference)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Flash Attention Testing Framework")
    print("CUDA Cores Only Implementation (No Tensor Cores)")
    print(f"Test Mode: {args.mode.upper()}")
    if args.experiment:
        print("Running COMPREHENSIVE experiment: all kernels on all test configs")
    elif args.seqlen_experiment:
        print("Running SEQUENCE LENGTH scaling experiment: S=128 to 2048")
    else:
        print(f"Kernel: {args.kernel.upper()}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Stop on failure: {not args.no_stop_on_failure}")
    if args.save_results:
        print(f"Results will be saved to: {args.output_dir}")
    print(f"{'='*80}\n")
    
    if args.experiment:
        configs = create_experiment_configs(args.mode)
    elif args.seqlen_experiment:
        configs = create_sequence_length_experiment_configs(args.mode)
    else:
        configs = create_test_configs(test_mode=args.mode, kernel_type=args.kernel)

    
    tester = FlashAttention2Tester(
        stop_on_failure=not args.no_stop_on_failure,
        tolerance=args.tolerance,
        test_mode=args.mode,
        save_results=args.save_results,
        output_dir=args.output_dir,
        use_gpu_reference=not args.no_gpu_reference
    )
    
    try:
        tester.run_all_tests(configs)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        tester.print_summary()
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        if tester.results:
            tester.print_summary()


if __name__ == '__main__':
    main()
