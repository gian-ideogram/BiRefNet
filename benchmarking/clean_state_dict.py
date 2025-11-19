#!/usr/bin/env python3
'''
Clean state dict keys from a .pth file.

This script transforms model checkpoint keys by:
1. Removing the first prefix (splits by "." and takes from index 1 onwards)
2. Replacing "module._orig_mod." with ""

Usage:
    python clean_state_dict.py <input.pth> <output.pth>

Gian Favero
Ideogram
2025-11-18
'''

import argparse
import torch
from collections import OrderedDict


def clean_state_dict_keys(state_dict):
    '''
    Clean up the keys in a state dictionary.
    
    Args:
        state_dict: OrderedDict or dict, the state dictionary to clean
        
    Returns:
        clean_state_dict: OrderedDict, the cleaned state dictionary
    '''
    clean_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove the first prefix by splitting and taking from index 1
        k = k.split(".")[1:]
        k = ".".join(k)
        # Remove module._orig_mod. prefix
        k = k.replace("module._orig_mod.", "")
        clean_state_dict[k] = v
    
    return clean_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Clean state dict keys from a PyTorch .pth file"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to input .pth file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save cleaned .pth file"
    )
    
    args = parser.parse_args()
    
    # Load the state dict
    print(f"Loading state dict from: {args.input_path}")
    state_dict = torch.load(args.input_path, map_location='cpu')
    
    # Check if the loaded object is a dict or has a 'state_dict' key
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        print("Found 'state_dict' key in checkpoint, extracting...")
        state_dict = state_dict['state_dict']
    
    print(f"Original number of keys: {len(state_dict)}")
        
    # Clean the state dict
    clean_state_dict = clean_state_dict_keys(state_dict)
    
    print(f"Cleaned number of keys: {len(clean_state_dict)}")
    
    # Save the cleaned state dict
    print(f"\nSaving cleaned state dict to: {args.output_path}")
    torch.save(clean_state_dict, args.output_path)
    
    print("Done!")


if __name__ == "__main__":
    main()