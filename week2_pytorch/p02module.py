import torch
import torch.nn as nn

class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        return x @ self.weight.T + self.bias

class TwoLayerNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = SimpleLinear(in_dim, hidden_dim)
        self.relu = nn.ReLU()  # non-linearity
        self.layer2 = SimpleLinear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.layer1(x)   # [batch, in_dim] -> [batch, hidden_dim]
        x = self.relu(x)     # ReLU: max(0, x) - adds non-linearity
        x = self.layer2(x)   # [batch, hidden_dim] -> [batch, out_dim]
        return x

def test():
    # Test SimpleLinear
    model1 = SimpleLinear(3, 2)
    x1 = torch.randn(4, 3)
    out1 = model1(x1)
    assert out1.shape == (4, 2), f"SimpleLinear failed: {out1.shape}"
    
    # Test TwoLayerNet  
    model2 = TwoLayerNet(in_dim=3, hidden_dim=8, out_dim=2)
    x2 = torch.randn(4, 3)  # batch=4, features=3
    out2 = model2(x2)
    assert out2.shape == (4, 2), f"TwoLayerNet failed: {out2.shape}"
    
    print("Both tests passed!")
    print("SimpleLinear output shape:", out1.shape)
    print("TwoLayerNet output shape:", out2.shape)
    print("\nModel architecture:")
    print(model2)
    return True

if __name__ == "__main__":
    test()
