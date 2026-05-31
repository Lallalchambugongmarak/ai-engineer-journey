import torch
import torch.nn as nn

class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        return x @ self.weight.T + self.bias

def test():
    model = SimpleLinear(3, 2)
    x = torch.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 2), f"Wrong shape: {out.shape}"
    print("Test passed. Output shape:", out.shape)
    return True

if __name__ == "__main__":
    test()
