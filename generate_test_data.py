import numpy as np
import argparse
import os


def generate_test_data(batch_size, num_heads, seq_len, head_dim, output_dir="data", seed=42):
    """
    Generate random Q, K, V matrices for attention computation.
    """
    np.random.seed(seed)
    
    folder_name = f"B{batch_size}_H{num_heads}_S{seq_len}_D{head_dim}"
    full_path = os.path.join(output_dir, folder_name)
    os.makedirs(full_path, exist_ok=True)
    
    print(f"Generating test data:")
    print(f"Batch size:  {batch_size}")
    print(f"Num heads:   {num_heads}")
    print(f"Seq length:  {seq_len}")
    print(f"Head dim:    {head_dim}")
    print(f"Output dir:  {full_path}")
    
    shape = (batch_size, num_heads, seq_len, head_dim)
    total_elements = batch_size * num_heads * seq_len * head_dim
    
    print(f"\nGenerating Q matrix... ({total_elements} elements)")
    Q = np.random.randn(*shape).astype(np.float32)
    
    print(f"Generating K matrix... ({total_elements} elements)")
    K = np.random.randn(*shape).astype(np.float32)
    
    print(f"Generating V matrix... ({total_elements} elements)")
    V = np.random.randn(*shape).astype(np.float32)
    
    q_path = os.path.join(full_path, "Q.bin")
    k_path = os.path.join(full_path, "K.bin")
    v_path = os.path.join(full_path, "V.bin")
    
    print(f"\nSaving matrices to binary files...")
    Q.tofile(q_path)
    print(f"Saved: {q_path} ({os.path.getsize(q_path)} bytes)")
    
    K.tofile(k_path)
    print(f"Saved: {k_path} ({os.path.getsize(k_path)} bytes)")
    
    V.tofile(v_path)
    print(f"Saved: {v_path} ({os.path.getsize(v_path)} bytes)")
    
    print(f"\nDone! Data saved to: {full_path}")
    return full_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate Q, K, V matrices for Flash Attention testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('batch_size', type=int, help='Batch size')
    parser.add_argument('num_heads', type=int, help='Number of attention heads')
    parser.add_argument('seq_len', type=int, help='Sequence length')
    parser.add_argument('head_dim', type=int, help='Dimension of each head')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Base directory for output')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.batch_size <= 0 or args.num_heads <= 0 or args.seq_len <= 0 or args.head_dim <= 0:
        parser.error("All dimensions must be positive integers")
    
    generate_test_data(
        args.batch_size,
        args.num_heads,
        args.seq_len,
        args.head_dim,
        args.output_dir,
        args.seed
    )


if __name__ == "__main__":
    main()
