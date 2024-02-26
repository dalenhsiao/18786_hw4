import torch
import torch.nn as nn 

class unStableRNN(nn.Module):
    def __init__(self, n_layers=1, activation=None):
        super(unStableRNN, self).__init__()
        self.layers = nn.ModuleList([myRNN(activation) for _ in range(n_layers)])
        
    def forward(self,x):
        xs = []
        ys = []
        for module in self.layers:
            x, y = module(x)
            xs.append(x)
            ys.append(y)
            
        return xs, ys
            
            
        
        
class myRNN(nn.Module):
    def __init__(self, activation: str = None):
        super(myRNN, self).__init__()
        self.A = self.C = torch.Tensor([[1.25,0.75], [0.75,1.25]])
        activation = activation.lower() if activation is not None else None
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
        x = x.transpose(1,0) # (2,10)
        f = torch.matmul(self.A,x)
        x_t = self.act(f)
        y_t = torch.matmul(self.C, x_t)
        return x_t.transpose(1,0), y_t.transpose(1,0)
    def __call__(self,x):
        return self.forward(x)
        
        

if __name__ == "__main__":
    from torch.distributions import Normal
    import numpy as np
    import matplotlib.pyplot as plt
    
    def l2_norm(x):
        return np.sqrt(x[:,0]**2+x[:,1]**2)
    
    ## Normal distribution 
    normal = Normal(torch.Tensor([0.0]), torch.Tensor([1.0])) # standard normal distribution
    ## Sampling 
    # since we dont need to track back propagate, simply random sampling w/ sample
    # for tracking backpropagation, use rsample instead
    vectors = normal.sample((10,2)).squeeze(-1)

    
    # RNN 
    t = 16
    activations = ["relu", "tanh", "linear"]
    logs = {}
    for act in activations:
        logs[act] = {}
        model = unStableRNN(n_layers=t, activation=act)
        _, out_y = model(vectors)
        logs[act] = out_y

    
    # plot the result 
    plt.figure(figsize=(10,8))
    for i, mode in enumerate(logs):
        ys = logs[mode] 
        all_y = []
        for y in ys: 
            y = y.detach().numpy()
            y_norm = l2_norm(y)
            all_y.append(y_norm)
        
        plt.subplot(len(logs), 1,i+1)
        plt.plot(all_y)
        plt.xlabel("t")
        plt.ylabel("$||y_t||_2$")
        plt.title(mode)
        plt.grid(True)
    
    plt.suptitle("10 Trajectories")
    plt.tight_layout()
    plt.savefig("./10_trajectories.png")
    plt.show()
    
    ###### plot trajectories for [1,1] and [1,-1]
    x0 = torch.Tensor([[1,1], [1,-1]])
    logs2 = {}
    for act in activations:
        logs2[act] = {}
        model = unStableRNN(n_layers=t, activation=act)
        _, out_y = model(x0)
        logs2[act] = out_y
    
    # plot the result
    plt.figure(figsize=(10,8))
    for i, mode in enumerate(logs2):
        ys = logs2[mode] 
        all_y = []
        for y in ys: 
            y = y.detach().numpy()
            y_norm = l2_norm(y)
        
            all_y.append(y_norm)
        all_y = np.array(all_y)
        plt.subplot(len(logs), 1,i+1)
        
        plt.plot(all_y[:, 0], label="[1,1]")
        plt.plot(all_y[:, 1], label="[1,-1]")
        plt.xlabel("t")
        plt.ylabel("$||y_t||_2$")
        plt.title(mode)
        plt.legend(loc = "upper left")
        plt.grid(True)
    
    plt.suptitle("[1,1] and [1,-1] Trajectories")
    plt.tight_layout()
    plt.savefig("./2_trajectories.png")
    plt.show()
    
        