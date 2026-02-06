"""
Convert CSV datasets to NPY format for faster loading

Usage:
    python convert_csv_to_npy.py datasets/unified/train/raw/train_D.csv theta_gnn
    python convert_csv_to_npy.py datasets/unified/train/raw/train_D.csv p_value
"""

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import time

def convert_theta_gnn_to_npy(csv_file, output_dir):
    """
    Convert theta_gnn CSV to NPY format

    Output structure:
        output_dir/
            sample_00000.npz
            sample_00001.npz
            ...

    Each npz contains:
        - edge_index: (2, num_edges) int array
        - edge_attr: (num_edges,) float array
        - y: float (theta value)
        - metadata: [n, rho, h, epsilon]
    """
    print(f"Converting {csv_file} to NPY format...")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    print("Reading CSV file...")
    start_time = time.time()

    # Use chunks for large files
    chunk_size = 100
    num_samples = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
        for idx, row in chunk.iterrows():
            # Parse the row
            if isinstance(row, pd.Series):
                # If pandas read as single value, split by comma
                values = row.iloc[0].split(',') if len(row) == 1 else row.values
            else:
                values = row

            # Extract fields
            # Format: num_rows, num_cols, theta, rho, h, nnz, [values], [row_ptrs], [col_indices]
            n = int(values[0])
            theta = float(values[2])
            rho = float(values[3])
            h = float(values[4])
            nnz = int(values[5])

            # Parse CSR sparse matrix
            val_start = 6
            val_end = val_start + nnz
            values_csr = np.array([float(x) for x in values[val_start:val_end]], dtype=np.float32)

            ptr_len = n + 1
            ptr_start = val_end
            ptr_end = ptr_start + ptr_len
            row_ptrs = np.array([int(x) for x in values[ptr_start:ptr_end]], dtype=np.int32)

            col_start = ptr_end
            col_end = col_start + nnz
            col_indices = np.array([int(x) for x in values[col_start:col_end]], dtype=np.int32)

            # Convert CSR to edge_index and edge_attr
            edge_index_list = []
            for i in range(n):
                for j in range(row_ptrs[i], row_ptrs[i+1]):
                    edge_index_list.append([i, col_indices[j]])

            edge_index = np.array(edge_index_list, dtype=np.int32).T  # Shape: (2, num_edges)
            edge_attr = values_csr  # Already in correct order

            # Save as npz
            output_file = os.path.join(output_dir, f"sample_{num_samples:05d}.npz")
            np.savez_compressed(
                output_file,
                edge_index=edge_index,
                edge_attr=edge_attr,
                theta=np.array([theta], dtype=np.float32),
                y=np.array([rho], dtype=np.float32),  # Target for rho prediction
                metadata=np.array([n, rho, h], dtype=np.float32)
            )

            num_samples += 1

            if num_samples % 100 == 0:
                elapsed = time.time() - start_time
                rate = num_samples / elapsed
                print(f"  Converted {num_samples} samples ({rate:.1f} samples/s)")

    elapsed = time.time() - start_time
    print(f"Conversion complete!")
    print(f"  Total samples: {num_samples}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Rate: {num_samples/elapsed:.1f} samples/s")

    # Check file sizes
    total_size = sum(os.path.getsize(os.path.join(output_dir, f))
                     for f in os.listdir(output_dir) if f.endswith('.npz'))
    print(f"  Total size: {total_size / 1024**3:.2f} GB")
    print(f"  Average size per sample: {total_size / num_samples / 1024:.1f} KB")


def convert_p_value_to_npy(csv_file, output_dir):
    """
    Convert p_value CSV to NPY format

    Each npz contains:
        - A_values, A_row_ptr, A_col_idx: CSR matrix A
        - coarse_nodes: coarse node indices
        - P_values, P_row_ptr, P_col_idx: CSR matrix P
        - S_values, S_row_ptr, S_col_idx: CSR matrix S
        - metadata: [n, theta, rho, h]
    """
    print(f"Converting {csv_file} (p_value format) to NPY format...")
    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Reading CSV file...")
    start_time = time.time()

    # Read line by line for p_value format (very large rows)
    num_samples = 0

    with open(csv_file, 'r') as f:
        for line in f:
            values = np.array([float(x) for x in line.strip().split(',')])

            # Parse format: n, n, theta, rho, h, nnz_A, [A], num_coarse, [coarse], nnz_P, [P], nnz_S, [S]
            idx = 0
            n = int(values[idx])
            idx += 2  # skip second n
            theta = values[idx]
            idx += 1
            rho = values[idx]
            idx += 1
            h = values[idx]
            idx += 1

            # Matrix A
            nnz_A = int(values[idx])
            idx += 1
            A_values = values[idx:idx + nnz_A].astype(np.float32)
            idx += nnz_A
            A_row_ptr = values[idx:idx + n + 1].astype(np.int32)
            idx += n + 1
            A_col_idx = values[idx:idx + nnz_A].astype(np.int32)
            idx += nnz_A

            # Coarse nodes
            num_coarse = int(values[idx])
            idx += 1
            coarse_nodes = values[idx:idx + num_coarse].astype(np.int32)
            idx += num_coarse

            # Matrix P
            nnz_P = int(values[idx])
            idx += 1
            P_values = values[idx:idx + nnz_P].astype(np.float32)
            idx += nnz_P
            P_row_ptr = values[idx:idx + n + 1].astype(np.int32)
            idx += n + 1
            P_col_idx = values[idx:idx + nnz_P].astype(np.int32)
            idx += nnz_P

            # Matrix S
            nnz_S = int(values[idx])
            idx += 1
            S_values = values[idx:idx + nnz_S].astype(np.float32)
            idx += nnz_S
            S_row_ptr = values[idx:idx + n + 1].astype(np.int32)
            idx += n + 1
            S_col_idx = values[idx:idx + nnz_S].astype(np.int32)

            # Save as npz
            output_file = os.path.join(output_dir, f"sample_{num_samples:05d}.npz")
            np.savez_compressed(
                output_file,
                A_values=A_values,
                A_row_ptr=A_row_ptr,
                A_col_idx=A_col_idx,
                coarse_nodes=coarse_nodes,
                P_values=P_values,
                P_row_ptr=P_row_ptr,
                P_col_idx=P_col_idx,
                S_values=S_values,
                S_row_ptr=S_row_ptr,
                S_col_idx=S_col_idx,
                metadata=np.array([n, theta, rho, h], dtype=np.float32)
            )

            num_samples += 1

            if num_samples % 100 == 0:
                elapsed = time.time() - start_time
                rate = num_samples / elapsed
                print(f"  Converted {num_samples} samples ({rate:.1f} samples/s)")

    elapsed = time.time() - start_time
    print(f"Conversion complete!")
    print(f"  Total samples: {num_samples}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Rate: {num_samples/elapsed:.1f} samples/s")

    # Check file sizes
    total_size = sum(os.path.getsize(os.path.join(output_dir, f))
                     for f in os.listdir(output_dir) if f.endswith('.npz'))
    print(f"  Total size: {total_size / 1024**3:.2f} GB")
    print(f"  Average size per sample: {total_size / num_samples / 1024:.1f} KB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_csv_to_npy.py <csv_file> <format>")
        print("  format: theta_gnn or p_value")
        sys.exit(1)

    csv_file = sys.argv[1]
    format_type = sys.argv[2]

    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    # Create output directory name
    base_dir = os.path.dirname(csv_file)
    csv_name = os.path.basename(csv_file).replace('.csv', '')
    output_dir = os.path.join(base_dir, f"{format_type}_npy", csv_name)

    if format_type == "theta_gnn":
        convert_theta_gnn_to_npy(csv_file, output_dir)
    elif format_type == "p_value":
        convert_p_value_to_npy(csv_file, output_dir)
    else:
        print(f"Error: Unknown format: {format_type}")
        print("  Supported formats: theta_gnn, p_value")
        sys.exit(1)

    print(f"\n✅ Conversion complete! Output in: {output_dir}")
