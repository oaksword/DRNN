import torch
import torch.nn as nn

def similarity(x, y, method='norm'):
    
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    
    px = x @ torch.inverse(x.T @ x) @ x.T
    py = y @ torch.inverse(y.T @ y) @ y.T
    
    if method == 'norm':
        
        sim = torch.norm(px - py)
        
    elif method == 'cos':
        
        sim = torch.sum(px * py) / (torch.norm(px) * torch.norm(py))
        
    return sim


class dr_net(nn.Module):
    
    def __init__(self, n_in, K):
        
        super(dr_net, self).__init__()
        self.net = nn.Linear(n_in, K, bias=False)
    
    def forward(self, x):
        
        return self.net(x)

    
class after_dr_net(nn.Module):
    
    def __init__(self, n_in, n_hidden, n_out):
        
        super(after_dr_net, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(n_in, n_hidden),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden, n_hidden // 2),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden // 2, n_out))
        
    def forward(self, x):
        
        return self.net(x)


class DRNN_Single():
    
    def __init__(self, n_in, n_hidden, n_out, K, device):
        
        self.K = K
        self.n_in = n_in
        self.device = device
        self.dr = dr_net(n_in, self.K).to(self.device)
        self.after_dr = after_dr_net(self.K, n_hidden, n_out).to(self.device)
        
    def train(self, x, y, epochs, lam, gamma, lr, step_size=200, batch_size=64):
        
        self.n = x.shape[0]
        self.loss_func = nn.MSELoss()
        
        self.optim = torch.optim.Adam([{'params': self.dr.net.parameters(),
                                        'weight_decay': 0.005 * lam},
                                       {'params': self.after_dr.parameters(),
                                        'weight_decay': 0.005 * lam}], lr=lr)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=step_size, gamma=0.9)
        self.loss_count = []
        
        for _ in range(epochs):
            
            sample = torch.randperm(self.n)[:batch_size]
            x_tmp = x[sample].to(self.device)
            y_tmp = y[sample].to(self.device)
            
            self.optim.zero_grad()
            
            w = self.dr.net.weight
            
            loss1 = self.loss_func(self.after_dr(self.dr(x_tmp)), y_tmp)
            loss2 = gamma * torch.norm(w, dim=1, p=2).sum()
            loss = loss1 + loss2
            
            self.loss_count.append(loss.detach().cpu().tolist())
            loss.backward()
            self.optim.step()
            
            if self.scheduler.get_last_lr()[0] > 1e-5:
                
                self.scheduler.step()  
                
                
class DRNN():
    
    def __init__(self, k, gamma, 
                 lr, n_iter, n_hidden, step_size, 
                 eps, device):
        
        self.k = k
        self.gamma = gamma
        self.lr = lr
        self.n_iter = n_iter
        self.n_hidden = n_hidden
        self.step_size = step_size
        self.eps = eps
        self.device = device
    
    def _run(self, x, y):
        
        p = x.shape[1]
        q = y.shape[1]
        model = DRNN_Single(n_in=p, n_hidden=self.n_hidden, 
                            n_out=q, K=self.k, device=self.device)
        model.train(x, y, epochs=self.n_iter, lam=0, gamma=self.gamma, 
                    lr=self.lr, step_size=self.step_size)
        
        return model
    
    def run(self, x, y, max_try=10):
        
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        model = self._run(x, y)

        flag = 1
        best_loss = model.loss_count[-1]
        self.beta = model.dr.net.weight.T.cpu().data

        while best_loss > self.eps and max_try > 1:

            model = self._run(x, y)
            flag += 1

            if model.loss_count[-1] < best_loss:

                best_loss = model.loss_count[-1]
                self.beta = model.dr.net.weight.T.cpu().data

            if flag == max_try:

                break
                