"""
Utility script to filter NaN/Inf losses from hard/easy samples
Run this before running the main analysis if you encounter NaN issues
"""
import json
import numpy as np
import sys
import argparse

def filter_valid_samples(input_file: str, output_file: str) -> tuple:
    """
    Filter out samples with NaN or Inf losses
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
        
    Returns:
        Tuple of (valid_count, invalid_count)
    """
    valid_count = 0
    invalid_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            data = json.loads(line)
            loss = data.get('loss', float('nan'))
            
            if np.isnan(loss) or np.isinf(loss):
                invalid_count += 1
                print(f"Filtering out sample {data.get('sample_idx')}, "
                      f"token {data.get('token_idx')}: loss={loss}")
                continue
            
            fout.write(line)
            valid_count += 1
    
    return valid_count, invalid_count


def main():
    parser = argparse.ArgumentParser(
        description="Filter NaN/Inf losses from hard/easy sample files"
    )
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("output_file", help="Output JSONL file path")
    
    args = parser.parse_args()
    
    print(f"Filtering {args.input_file}...")
    valid_count, invalid_count = filter_valid_samples(args.input_file, args.output_file)
    
    print(f"\nResults:")
    print(f"  Valid samples: {valid_count}")
    print(f"  Invalid samples (NaN/Inf): {invalid_count}")
    print(f"  Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
