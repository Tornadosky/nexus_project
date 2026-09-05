"""Execution-only environment batching diagnostic. No reward/algorithm changes."""
def install(size=32):
    import jax
    from brax.envs.wrappers import training
    cls=training.VmapWrapper
    if getattr(cls,'_campaign_chunk',None)==size:return
    if getattr(cls,'_campaign_chunk',None) is not None:
        raise RuntimeError('Different environment chunk size already installed')
    def mapped(fun,tree,n):
        if n<=size:return jax.vmap(fun)(tree)
        if n%size:raise ValueError('Environment batch must divide the execution chunk')
        # Explicit dimensions are essential: MJX states contain zero-size leaves.
        batched=jax.tree.map(lambda x:x.reshape((n//size,size)+x.shape[1:]),tree)
        result=jax.lax.map(jax.vmap(fun),batched)
        return jax.tree.map(lambda x:x.reshape((n,)+x.shape[2:]),result)
    def reset(self,rng):
        if self.batch_size is not None:
            rng=jax.random.split(rng,self.batch_size)
        return mapped(self.env.reset,rng,rng.shape[0])
    def step(self,state,action):
        return mapped(lambda pair:self.env.step(pair[0],pair[1]),
                      (state,action),action.shape[0])
    cls.reset=reset
    cls.step=step
    cls._campaign_chunk=size
