import math
import torch
from torch.optim.lr_scheduler import LambdaLR, ConstantLR


######################################################################################
# This function incorporates code from pytorch/torchtune
# Licenced under the BSD-3 Clause Licence: https://opensource.org/license/BSD-3-clause
######################################################################################
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_placeholder_schedule(
    optimizer: torch.optim.Optimizer,
    last_epoch: int = -1,
) -> ConstantLR:
    
    return ConstantLR(optimizer, factor=1.0, total_iters=0x7FFFFFF, last_epoch=last_epoch)


def test():
    num_epochs = 10
    warmup_steps = 100
    epoch_steps = 1000
    total_steps = num_epochs * epoch_steps

    optimizer = torch.optim.Adam([torch.zeros(1)], lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, num_cycles=num_epochs//2-0.5)

    lrs = []

    from tqdm import tqdm
    for epoch in tqdm(range(num_epochs)):
        for step in range(epoch_steps):
            optimizer.step()
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)

    import matplotlib.pyplot as plt
    # save the figure to the disk
    plt.plot(lrs)
    plt.savefig('lrs.png') 

if __name__ == "__main__":
    test()