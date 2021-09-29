import torch

# we can try just making up how training works, and see if anything happens ;p

class ConstantWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    '''Applies a constant warmup learning rate.  Should be last in chain to override other learning rates.'''
    def __init__(self, optimizer, epochs = 1, lr = 1e-5):
        self.epochs = epochs
        self.lr = lr
        super().__init__(optimizer)
    def get_lr(self):
        if self.last_epoch <= self.epochs:
            return [self.lr for base_lr in self.base_lrs]
        else:
            return self.base_lrs
            

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
        if hasattr(self.model, 'config'):
            self.config = self.model.config
        self._training = None

        self._optimizer_class = optimizer
        self._optimizer_params = optimizer_params

        self._lr_schedulers = lr_schedulers
    
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.embeds = torch.empty((0,0), device=self.model.device)
        self.param = torch.nn.Parameter(self.embeds)

        self.num_embeds = num_embeds
        self.randomize_embeds()

        #wrapped_prepare_inputs_for_generation = model.prepare_inputs_for_generation
        #def prepare_inputs_for_generation_wrapper(*params, **kwparams):
        #    result = wrapped_prepare_inputs_for_generation(*params, **kwparams)
        #    del result['input_ids']
        #    result['inputs_embeds'] = self.embeds
        #    return result
        #expand_inputs_for_generation = model.

        # the first example used SGD with an LR of 0.002, and 50 or 200 epochs.
        # used an LR scheduler 'cosine', with a single constant warmup epoch with an LR of 1e-5
            # two schedulers: warmup and cosine
            # cosine is simple annealing with max epoch = (200,50)
            # constant warmup has the epoch and learning rare

    #def step(self, *params, **kwparams):
    #    self.optim.zero_grad()
    #    output = self.model(*params, inputs_embeds=self.embeds, **kwparams)
    #    # really we want to process every token, so loss would go in the callback
    #    loss = loss_fn(output, target)
    #    loss.backward()
    #    self.optim.step()
    #    # call after batch
    #    for sched in self.scheds:
    #        sched.step()

    def save_embeds(self, filename):
        import os
        torch.save(self.embeds, filename + '.new')
        os.rename(filename + '.new', filename)

    def load_embeds(self, filename):
        self.set_embeds(torch.load(filename))

    def __enter__(self, *params):
        if self._training is not True:
            self.model.train()
            self._training = True
        self.optim.zero_grad()
        self.model.zero_grad()
        return self

    def __exit__(self, *params):
        self.optim.step()
        for sched in self.scheds:
            sched.step()

    def forward_and_backward(self, requested_outputs, *params, **kwparams):
        # stretch embeddings to include the requested outputs, minus one token for the shift
        embeds = torch.cat([self.param.expand(requested_outputs.shape[0], *self.param.shape), self.wte(requested_outputs[:,:-1])], dim=1)
        
        # labels are compared with the full text.  a value of -100 is supposed to be ignored.

        #  didn't find where -100 was respected in the source yet, so below draft is replaced with manual loss
        #labels = torch.cat([torch.full(self.embeds.shape,-100, device=self.model.device), requested_outputs], dim=1)
        #outputs = self.model(*params, inputs_embeds=self.embeds, labels=requested_outputs, **kwparams)
        #loss, logits = outputs[:2]

        # manual loss
        logits = self.model(*params, inputs_embeds=embeds, labels=None, return_dict=False, **kwparams)[0]
        logits = logits[:, self.num_embeds-1:, :].contiguous()
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), requested_outputs.view(-1))

        loss.backward()
        return loss.detach()

    def find_chunksize_for_data(self, requested_outputs, *params, **kwparams):
        self.forward_and_backward(requested_outputs[:1], *params, **kwparams)
        try:
            self.forward_and_backward(requested_outputs, *params, **kwparams)
            return requested_outputs.shape[0]
        except:
            pass
        low = 1
        high = requested_outputs.shape[0]
        while low + 1 < high:
            mid = (low + high) // 2
            if mid == low:
                mid += 1
            if mid == high:
                mid -= 1
            try:
                self.forward_and_backward(requested_outputs[:mid], *params, **kwparams)
                low = mid
            except:
                high = mid
        return low

    def epoch(self, requested_outputs, chunksize=16, *params, verbose = False, **kwparams):
        loss_sum = 0
        chunks = 0

        range_iter = range(0, requested_outputs.shape[0], chunksize)
        if verbose:
            import tqdm
            range_iter = tqdm.tqdm(range_iter, leave=False, desc='chunk')

        # atm batch size (data between model updates) is equated to epoch size (full data pass)

        with self: # enter batch
            for subrange in range(0, requested_outputs.shape[0], chunksize): # cross epoch in chunks
                chunk = requested_outputs[subrange:subrange+chunksize]
                loss_sum += self.forward_and_backward(chunk, *params, **kwparams)
                chunks += 1
        return loss_sum / chunks

    def __call__(self, input_ids = None, attention_mask = None, **kwparams):
        if self._training is not False:
            self.model.eval()
            self._training = False
        return self.model(inputs_embeds=self.embeds,**kwparams)

    #def generate(self, input_ids = None, **kwparams):
    #    # generate is a little confusing because ew have embeds, huggingface assumes running token ids, and token ids are produced as output (not embeds)
    #    # if we are producing our own output, we'd havce to decide how much
    #    if self._training is not False:
    #        self.model.eval()
    #        self._training = False
    #    return self.model.generate(input_ids = torch.zeros(1, self.num_embeds, dtype=torch.int, device=self.model.device), **kwparams)
    def generate_tokens(self, **kwparams):
        if self._training is not False:
            self.model.eval()
            self._training = False
        running_embeds = self.embeds.clone().detach()
        idx = 0
        while True:
            new_tokens = torch.argmax(self.model(inputs_embeds = running_embeds, **kwparams)[0][idx:], dim=-1)
            yield from new_tokens
            idx += new_tokens.shape[0]
            running_embeds = torch.cat([running_embeds, self.wte(new_tokens[-1:])])
    def array_tokens(self, len, **kwparams):
        return [token for token, idx in zip(self.generate_tokens(**kwparams), range(len))]
            

    @property
    def vocab_size(self):
        return self.wte.num_embeddings
    @property
    def embed_dim(self):
        return self.wte.embedding_dim
    @property
    def wte(self):
        return self.model.transformer.wte
    @property
    def num_embeds(self):
        return self.embeds.shape[0]
    @num_embeds.setter
    def num_embeds(self, num_embeds):
        self.param.requires_grad_(False)
        prev_embeds = self.embeds
        self.embeds = self.wte(torch.zeros(num_embeds, dtype=torch.int, device=self.model.device))
        copy_len = min(len(prev_embeds), num_embeds)
        if copy_len:
            self.embeds[-copy_len:] = prev_embeds[-copy_len:]
        self.param = torch.nn.Parameter(self.embeds)

        # might be able to mutate the parameter list live an dnot recreate the optimizers,
        # don't know
        self.optim = self._optimizer_class([self.param], **self._optimizer_params)
        self.scheds = [
            cls(self.optim, **params)
            for cls, params in self._lr_schedulers
        ]
    # could add a discretization step where loss is multiplied by distance of embeddings from token ids
    def randomize_embeds(self):
        with torch.no_grad():
            # sampling from embedding space might be better but unsure how to quickly find its bounds
            token_ids = torch.empty(self.num_embeds, device=self.model.device)
            torch.nn.init.uniform_(token_ids, 0, self.vocab_size)
            self.embeds[:] = self.wte(token_ids.to(torch.int))
    def set_input_ids(self, input_ids):
        return self.set_embeds(self.wte(input_ids))
    def set_embeds(self, embeds):
        if len(embeds) != self.num_embeds:
            self.num_embeds = len(embeds)
        with torch.no_grad():
            self.embeds[:] = embeds
