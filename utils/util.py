import math

class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.model = model.eval()
        self.decay = decay
        self.shadow_params = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
             #   print(name)
                self.shadow_params[name] = param.data.clone() 
        self.apply_ema()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] -= (1.0 - self.decay) * (self.shadow_params[name] - param.data)
        self.apply_ema()

    def apply_ema(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow_params[name])

def adjust_learning_rate_spike(optimizer, epoch, warmup_epochs, lr, min_lr, no_of_epochs):
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (no_of_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def adjust_learning_rate_linear(optimizer, epoch, num_epochs, initial_lr):
    if epoch < num_epochs // 2:
        lr = initial_lr  
    else:
        lr = initial_lr * (1 - (epoch - num_epochs // 2) / (num_epochs // 2))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

