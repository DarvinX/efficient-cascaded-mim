"""Exact shuffled training passes, with a durable per-update data cursor."""
import copy
import hashlib
import math

import numpy as np


class ShuffledEpochs:
    def __init__(self,ids,batch,accumulate,epochs,rng):
        if any(type(x) is not int or x<1 for x in (batch,accumulate,epochs)):
            raise ValueError("Batch, accumulation and epochs must be positive integers")
        values=np.asarray(ids)
        if values.ndim!=1 or not np.issubdtype(values.dtype,np.integer):
            raise ValueError("Training IDs must be an integer vector")
        self.ids=values.astype(np.int64).copy()
        if not len(self.ids) or len(np.unique(self.ids))!=len(self.ids):
            raise ValueError("Epoch sampler requires nonempty unique training IDs")
        self.batch,self.accumulate,self.epochs,self.rng=batch,accumulate,epochs,rng
        self.effective=batch*accumulate
        self.config={"format":"shuffled-epochs-v1","ids_sha256":hashlib.sha256(self.ids.astype('<i8').tobytes()).hexdigest(),"batch":batch,"accumulate":accumulate,"epochs":epochs}
        self.epoch,self.position,self.order=0,0,None

    @property
    def completed_epochs(self):
        return self.epoch+int(self.position==len(self.ids))

    @property
    def examples_seen(self):
        return self.epoch*len(self.ids)+self.position

    @property
    def updates_completed(self):
        return self.epoch*math.ceil(len(self.ids)/self.effective)+math.ceil(self.position/self.effective)

    def next_batch(self):
        if self.completed_epochs==self.epochs:
            raise StopIteration
        if self.position==len(self.ids):
            self.epoch+=1;self.position=0;self.order=None
        if self.order is None:
            self.order=self.rng.permutation(self.ids)
        ids=self.order[self.position:self.position+self.effective].copy()
        self.position+=len(ids)
        return ids

    def state_dict(self):
        return {"config":dict(self.config),"epoch":self.epoch,"position":self.position,"order":self.order.copy() if self.order is not None else None,"rng_state":copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self,state):
        if state["config"]!=self.config:
            raise ValueError("Epoch sampler data/budget changed")
        epoch,position=state["epoch"],state["position"]
        if type(epoch) is not int or type(position) is not int or not 0<=epoch<self.epochs or not 0<=position<=len(self.ids):
            raise ValueError("Invalid epoch cursor")
        if position!=len(self.ids) and position%self.effective:
            raise ValueError("Epoch checkpoint is not at a completed update")
        order=state["order"]
        if order is None:
            if epoch or position:
                raise ValueError("Missing saved epoch permutation")
        else:
            order=np.asarray(order)
            if not np.issubdtype(order.dtype,np.integer):
                raise ValueError("Saved permutation IDs must be integers")
            order=order.astype(np.int64)
            if order.shape!=self.ids.shape or not np.array_equal(np.sort(order),np.sort(self.ids)):
                raise ValueError("Saved permutation is not the training split")
            order=order.copy()
        self.rng.bit_generator.state=copy.deepcopy(state["rng_state"])
        self.epoch,self.position,self.order=epoch,position,order
