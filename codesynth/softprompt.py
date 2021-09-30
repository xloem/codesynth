import torch


# we can try just making up how training works, and see if anything happens ;p
            

# something like this was used in the example code referenced by the paper
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
        optimizer_params = dict(lr=0.02),#0.002),
        hardness = 0, # weighting for nearness to real token ids, can be changed later
        hardness_lite = True, # only measures gradients for each embed to one vocab word
        #hardness_cpu = True, # calculates distance to all vocab words on cpu, saves vram, but doesn't make use of gpu
        straight_logits_loss = False,
        lr_schedulers = [
            (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, dict(T_0=50)),
            (ConstantWarmupLR, dict()),
        ],
    ):
        self.model = model
        self.hardness = hardness
        self.hardness_lite = hardness_lite
        #self.hardness_cpu = hardness_cpu
        self.straight_logits_loss = straight_logits_loss
        if hasattr(self.model, 'config'):
            self.config = self.model.config
        self._training = None
        self._etw = None

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

    #@property:
    #def etw(self):
    #    if self._etw is None:
    #        # each embedding has a neighborhood size, where a token is more near to it than any other
    #        # the closest 2 embeddings have the smallest neighborhood size
    #        self.etw
                    

    def forward_and_backward(self, pad_token_id, requested_outputs, *params, **kwparams):
        # stretch embeddings to include the requested outputs, shifting one token for being input
        embeds = torch.cat([self.param.expand(requested_outputs.shape[0], *self.param.shape), self.wte(requested_outputs[:,:-1])], dim=1)
        
        # labels are compared with the full text.  a value of -100 is supposed to be ignored.

        #  i did loss manually because i couldn't find where -100 was handled when verifying control flow.
        #  it's handled in cross_entropy.  so it should be fine to use huggingface's training mechanism by the debugging below
        #labels = torch.cat([torch.full(self.embeds.shape,-100, device=self.model.device), requested_outputs], dim=1)
        #outputs = self.model(*params, inputs_embeds=self.embeds, labels=requested_outputs, **kwparams)
        #loss, logits = outputs[:2]

        # manual loss
        logits = self.model(*params, inputs_embeds=embeds, labels=None, return_dict=False, **kwparams)[0]
        #                    batch      token spot   logits
        output_logits = logits[:, self.num_embeds-1:, :].contiguous()
            # todo? a weight could optionally be multiplied in for each requested output
        if not self.straight_logits_loss:
            loss = torch.nn.functional.cross_entropy(output_logits.view(-1, output_logits.size(-1)), requested_outputs.view(-1), ignore_index = pad_token_id)
        else:
            shape = requested_outputs.shape
            requested_logits = output_logits[torch.arange(shape[0]).repeat(shape[1],1).t(), torch.arange(shape[1]).repeat(shape[0],1), requested_outputs]
            #requested_probs = torch.sigmoid(requested_logits).prod(dim=-1)
            loss = -requested_logits.mean()

        # hardness provides for embeddinss to be trained to match tokens
        '''
            the logits output by the model are not the input tokens.
            they would be near them only if the input is similar to the model's training set.
            the fastest way to find them would be to form a graph of nearness (precalculated 128d voronoi diagram)
        '''
        if self.hardness:
            embeds = self.embeds if self.hardness_lite else self.param
            vocab = self.wte.weight
            #if self.hardness_cpu:
            #    embeds = embeds.cpu()
            #    vocab = vocab.cpu()

            embed_vocab_distances = []
            for embed in embeds:
                distance = embed.expand(self.vocab_size, *embed.shape) - vocab
                distance = (distance * distance).sum(dim=-1)
                embed_vocab_distances.append(distance)
            embed_vocab_distances = torch.stack(embed_vocab_distances).t()
                    
            # vector cross difference
            #embed_vocab_distances = (
            #    # embeds rows
            #    embeds.expand(self.vocab_size, *self.param.shape) -
            #    # vocab cols
            #    vocab.expand(self.num_embeds, *vocab.shape).permute(1, 0, 2)
            #)
            # self-dot makes squared distance
            #embed_vocab_distances = (embed_vocab_distances * embed_vocab_distances).sum(dim=-1)

            # consider only the distance of each embed to the nearest vocab word
            nearest_vocab_words = embed_vocab_distances.min(dim=0)
                    # NEAREST TOKENS ARE .indices HERE

            if self.hardness_lite:
                embed_vocab_distances = self.wte.weight[nearest_vocab_words.indices] - self.param
                embed_vocab_distances = (embed_vocab_distances * embed_vocab_distances).sum(dim=-1)
            else:
                embed_vocab_distances = nearest_vocab_words.values
            loss += self.hardness * embed_vocab_distances.mean()

#asdfasdf
#            token_ids = torch.cat((torch.tensor((first_token_id,), device=self.model.device), torch.argmax(logits[0][:self.num_embeds-1], dim=-1)))
#            # manual loss
#            input_ids = torch.cat([token_ids.expand(requested_outputs.shape[0], *token_ids.shape), requested_outputs[:,:-1]], dim=1)
#            logits = self.model(*params, input_ids=input_ids, labels=None, return_dict=False, **kwparams)[0]
#            output_logits = logits[:, self.num_embeds-1:, :].contiguous()
#            loss += self.hardness * torch.nn.functional.cross_entropy(output_logits.view(-1, output_logits.size(-1)), requested_outputs.view(-1), ignore_index = pad_token_id)

        loss.backward()
        return loss.detach()

    def find_chunksize_for_data(self, requested_outputs, *params, **kwparams):
        self.forward_and_backward(-100, requested_outputs[:1], *params, **kwparams)
        try:
            self.forward_and_backward(-100, requested_outputs, *params, **kwparams)
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
                self.forward_and_backward(-100, requested_outputs[:mid], *params, **kwparams)
                low = mid
            except:
                high = mid
        return low

    def epoch(self, pad_token_id, requested_outputs, chunksize=16, *params, **kwparams):
        loss_sum = 0
        chunks = 0

        # atm batch size (data between model updates) is equated to epoch size (full data pass)

        with self: # enter batch
            for subrange in range(0, requested_outputs.shape[0], chunksize): # cross epoch in chunks
                chunk = requested_outputs[subrange:subrange+chunksize]
                loss_sum += self.forward_and_backward(pad_token_id, chunk, *params, **kwparams)
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

        # first yield tokens associated with the embeddings temselves
        for embed in self.embeds:
            dists = (self.wte.weight * embed.expand(self.wte.weight.shape[0], self.embed_dim)).sum(dim=-1)
            yield torch.argmax(dists)

        while True:
            new_token = torch.argmax(self.model(inputs_embeds = running_embeds, **kwparams)[0][-1:],dim=-1)
            yield new_token[0]
            running_embeds = torch.cat([running_embeds, self.wte(new_token)])
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

        copy_len = min(len(prev_embeds), num_embeds)

        self.embeds = torch.empty((num_embeds, self.embed_dim), device=self.model.device)

        if copy_len < num_embeds:
            rand_tokens = torch.empty(num_embeds - copy_len, device=self.model.device)
            torch.nn.init.uniform_(rand_tokens, 0, self.vocab_size)
            self.embeds[:num_embeds-copy_len] = self.wte(rand_tokens.to(torch.int))

        if copy_len:
            self.embeds[-copy_len:] = prev_embeds[-copy_len:]

        self.param = torch.nn.Parameter(self.embeds)

        self.embeds_cells = [None] * num_embeds

        # might be able to mutate the parameter list live an dnot recreate the optimizers,
        # don't know
        self.optim = self._optimizer_class([self.param], **self._optimizer_params)
        self.scheds = [
            cls(self.optim, **params)
            for cls, params in self._lr_schedulers
        ]
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

def nonvoronoi_cell(vec, other_vecs):
    '''approximates the nd vectors from a list that neighbor a given nd vector in nearness'''
    orig_vecs = other_vecs
    other_vecs = other_vecs - vec
    neighbors = [0]
    untested = [*zip(range(1,len(other_vecs)),other_vecs[1:])]
    while len(untested):
        candidate_idx, candidate = untested.pop()
        cand_dist_squared = (candidate * candidate).sum()
        candidate_unobstructed = True
        for neighbor_idx in neighbors:
            if neighbor_idx == candidate_idx:
                continue
            neighbor = other_vecs[neighbor_idx]
            neighb_dot_cand = (neighbor * candidate).sum()
            if neighb_dot_cand <= 0:
                # vectors are in opposite directions
                continue
            neighb_dist_squared = (neighbor * neighbor).sum()

            ratio = (cand_dist_squared - neighb_dist_squared) / (2 * (cand_dist_squared - neighb_dot_cand))
            # ratio is the squared distance to the candidate at which a point is equidistant to candidate and neighbor
            # derivation is preserved in this block of comments to help me understand this in the future; typed by hand so likely has typos
            # below, 't' is ratio and 'vec' is candidate, 'elem' is an element of vec and 'elem2' is an element of neighbor
            # dot(t * vec - vec) = dot(t * vec - neighbor)
            # sum((t * elem - elem2)**2)
            # sum(t**2 * elem**2 - 2t * elem * elem2 + elem2**2)
            # t**2 * sum(elem**2) - 2t * sum(elem * elem2) + sum(elem2**2)
            # quadratic equations in t
            # A = dot(vec, vec)
            # B = -2 dot(vec, neighbor)
            # C = dot(neighbor, neighbor)
            # except with vec=neighbor on one side
            # so combining them
            # A = 0
            # B = -2 dot(vec, vec) + 2 dot(vec, neighbor)
            # C = dot(vec, vec) - dot(neighbor, neighbor)
            # and if there's any correctness to that, it's now linear
            # t = (dot(vec, vec) - dot(neighbor, neighbor)) / (2 * (dot(vec, vec) - dot(vec, neighbor)))
            # t**2 * sum(elem**2) - 2t * sum(elem * elem2) + sum(elem2**2)
            if ratio < 0:
                continue
            if ratio < 1:
                candidate_unobstructed = False
                break
            if ratio > 1:
                untested.extend(zip(neighbors, other_vecs[neighbors]))
                neighbors = []
                break
        if candidate_unobstructed:
            neighbors.append(candidate_idx)
    return orig_vecs[neighbors]
            
# it's not voronoi.  voronoi is hard.
#def _test_voronoi(manual_seed = 0):
#    torch.manual_seed(manual_seed)
#    import scipy.spatial
#    points = torch.randn(16, 2)
#    npoints = points.numpy()
#
#    # just looking for neighboring points, not sure how to use voronoi/delauny structures
#    # below makes a dict of neighboring point idxs using scipy
#    voronoi = scipy.spatial.Voronoi(npoints)
#
#    vertices_to_neighboring_points = {}
#    points_to_neighbors = {}
#
#    for point_idx, region_idx in enumerate(voronoi.point_region):
#        region = voronoi.regions[region_idx]
#        for vertex_idx in region:
#            if vertex_idx != -1:
#                if vertex_idx not in vertices_to_neighboring_points:
#                    vertices_to_neighboring_points[vertex_idx] = set()
#                vertices_to_neighboring_points[vertex_idx].add(point_idx)
#        if -1 not in region:
#            points_to_neighbors[point_idx] = set()
#        
#    for vertex_idx, vertex_points in vertices_to_neighboring_points.items():
#        for point in vertex_points:
#            if point in points_to_neighbors:
#                points_to_neighbors[point].update(vertex_points - set((point,)))
#                
#    #print('Neighboring Voronoi points:')
#    #for point, neighbors in points_to_neighbors.items():
#    #    print(points[point],':',[points[point] for point in neighbors])
#    #
#    #import matplotlib.pyplot as plt
#    #fig = scipy.spatial.voronoi_plot_2d(voronoi)
#    #plt.show()
#    
#    # compare with handmade function
#    for region_point_idx, neighbor_point_idcs in points_to_neighbors.items():
#
#        # - also curious regarding plot: are there ridges [neighbors] that are not reachable via direct line of sight to point?
#        # - could be quite useful to have a screenshot of the plot.  it can also be made in a separate window via copy paste of points.
#
#        print('Neighboring Voronoi points:')
#        print(points[region_point_idx],':',[points[point_idx] for point_idx in neighbor_point_idcs])
#        print('Shifted neighboring voronoi points:')
#        print(points[region_point_idx],':',[points[point_idx] - points[region_point_idx] for point_idx in neighbor_point_idcs])
#
#        handmade_neighbors = set(voronoi_cell(points[region_point_idx], points))
#
#        print('Handmade neighboring points:')
#        print(points[region_point_idx], ':', [*handmade_neighbors])
#        
#        import matplotlib.pyplot as plt
#        fig = scipy.spatial.voronoi_plot_2d(voronoi)
#        plt.show()
#
#        if len(handmade_neighbors) != len(neighbor_point_idcs) + 1:
#            raise Exception('scipy got', len(neighbor_point_idcs) + 1, 'points, handmade got', len(handmade_neighbors), 'points')
#        for point_idx in (region_point_idx, *neighbor_point_idcs):
#            if points[point_idx] not in handmade_neighbors:
#                raise Exception('handmade not containing point', point_idx)
#    # todo: compare not coincident with points
#    return True
#
