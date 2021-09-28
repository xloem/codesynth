import torch

# we can try just making up how training works, and see if anything happens ;p

class ConstantWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    '''Applies a constant warmup learning rate.  Should be last in chain to override other learning rates.'''
    def __init__(self, optimizer, epochs = 1, lr = 1e-5):
        super().__init__(optimizer)
        self.epochs = epochs
        self.lr = lr
    def get_lr(self):
        if self.last_epoch <= self.epochs:
            return [self.lr for base_lr in self.baselrs)]
        else:
            return self.baselrs
            

# can get embeddings from tokens with model.wte(input_ids)
class SoftPromptTrainable:
    def __init__(
        self,
        model,
        num_embeds = 128,
        optimizer = torch.optim.SGD,
        optimizer_params = dict(lr=0.002),
        lr_schedulers = [
            (torch.optim.lr_scheduler.CosineAnnealingLR, dict(T_max=200)),
            (ConstantWarmupLR, dict()),
        ],
    ):
        self.model = model

        self._optimizer_class = optimizer
        self._optimizer_params = optimizer_params

        self._lr_schedulers = lr_schedulers
    
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.embeds = Torch.tensor([])
        self.param = torch.nn.Parameter(self.embeds)

        self.num_embeds = num_embeds
        self.randomize_embeds()

        # the first example used SGD with an LR of 0.002, and 50 or 200 epochs.
        # used an LR scheduler 'cosine', with a single constant warmup epoch with an LR of 1e-5
            # two schedulers: warmup and cosine
            # cosine is simple annealing with max epoch = (200,50)
            # constant warmup has the epoch and learning rare

    def step(self, *params, **kwparams):
        self.optim.zero_grad()
        output = self.model(*params, inputs_embeds=self.embeds, **kwparams)
        # really we want to process every token, so loss would go in the callback
        loss = loss_fn(output, target)
        loss.backward()
        self.optim.step()
        # call after batch
        for sched in self.scheds:
            sched.step()

    @property
    def vocab_size(self):
        return self.model.wte.num_embeddings
    @property
    def embed_dim(self):
        return self.model.wte.embedding_dim
    @property:
    def num_embeds(self):
        return self.embeds.shape[0]
    @nembeds.setter
    def num_embeds(self, value):
        self.param.requires_grad_(False)
        prev_embeds = self.embeds
        self.embeds = self.model.wte(torch.empty(num_embeds))
        copy_len = min(len(prev_embeds), num_embeds)
        self.embeds[-copy_len:] = prev_embeds
        self.param = torch.nn.Parameter(self.embeds)

        self.optim = self._optimizer_class(**self._optimizer_params)
        self.scheds = [
            cls(self.optim, **params),
            for cls, params in self._lr_schedulers
        ]
    def randomize_embeds(self):
        with torch.no_grad():
            torch.nn.init.uniform_(0, self.model.vocab_size)
    def set_embeds(self, input_ids):
        self.randomize_embeds():
        self.embeds[-len(input_ids):] = self.model.wte(input_ids)
    
