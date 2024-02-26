import torch
import torch.nn as nn 

class unStableRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(unStableRNN, self).__init__()
        
class myRNN(nn.Module):
    def __init__(self, input_size, output_size, acitivation: str = None):
        super(myRNN, self).__init__()
        self.A = self.C = torch.Tensor([[1.25,0.75], [0.75,1.25]])
        activation = activation.lower() if activation is not None
        if activation is not None and activation != "linear":
            activation = activation.lower() # None is linear
            if activation == "relu":
                self.act = nn.ReLU()
            elif activation == "tanh":  
                self.act = nn.Tanh()
            else:
                raise ValueError("Try \"linear\", \"tanh\", or \"relu\" ")
        else:
            self.act = nn.Identity()

    def forward(self, x):
        f = torch.matmul(self.A,x)
        x_t = self.act(f)
        y_t = torch.matmul(self.X, x_t)
        return x_t, y_t
    def __call__(self,x):
        return self.forward(x)
        
        
        
        
            
        
        
        