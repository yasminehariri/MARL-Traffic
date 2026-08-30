import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def test_neighbour_aggregation():
    # Two fake neighbour messages, each 8 numbers
    neighbor1 = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    neighbor2 = torch.tensor([4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0])

    # Stack them and average — same operation as comm_model.py
    stacked = torch.stack([neighbor1, neighbor2])
    result = stacked.mean(dim=0)

    # The correct average, computed by hand
    expected = torch.tensor([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

    assert torch.allclose(result, expected), f"Expected {expected} but got {result}"
    print("test_neighbour_aggregation PASSED")

if __name__ == "__main__":
    test_neighbour_aggregation()