import torch
from torch import nn

from tqdm.auto import trange, tqdm

import wandb

import buffers
from samplers import losses, sutils
import utils


class D2ESB:
    def __init__(self, fwd_model, bwd_model, data, config):
        self.fwd_model = fwd_model
        self.bwd_model = bwd_model
        
        self.fwd_optim = torch.optim.AdamW(self.fwd_model.parameters(), 
                                           lr=self.config.fwd_lr)
        self.bwd_optim = torch.optim.AdamW(self.bwd_model.parameters(), 
                                           lr=self.config.bwd_lr)
        
        self.data = data
        self.config = config

        self.ref_process = sutils.ReferenceProcess2(self.config.alpha)
        self.p1_buffer = buffers.ReplayBuffer(self.config.buffer_size)

    def train(self, experiment):
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = self.config.n_steps
        
        with wandb.init(name=experiment.name, project=experiment.project,
                        config=self.config) as run:
            for it in trange(self.config.num_sb_steps):
                # BACKWARD PROCESS
                bwd_ema_loss = utils.EMALoss(0.1)
                for _ in trange(self.config.num_per_step_ites, 
                                leave=False, desc=f'It {it} | Backward'):
                    self.bwd_optim.zero_grad(set_to_none=True)

                    x_0, _ = self.data.sample(512)
                    fwd_model = self.ref_process if it == 0 else self.fwd_model
                    traj_loss = losses.compute_bwd_tlm_loss(fwd_model, self.bwd_model, 
                                                            x_0, dt, t_max, n_steps)
                        
                    self.bwd_optim.step()
                    bwd_ema_loss.update(traj_loss.item() / n_steps)

                # FORWARD PROCESS
                fwd_ema_loss = utils.EMALoss(0.1)

                if it == 0:
                    x_0, _ = self.data.sample(512)
                else: 
                    x_1_buffer = self.p1_buffer.sample(512)
                    with torch.no_grad():
                        x_0 = utils.sample_trajectory(self.bwd_model, x_1_buffer,
                                                      "backward", dt, n_steps, t_max, 
                                                      only_last=True)

                for _ in trange(self.config.num_per_step_ites, 
                                leave=False, desc=f'It {it} | Forward'):
                    self.fwd_optim.zero_grad(set_to_none=True)

                    loss = losses.compute_fwd_ctb_loss(self.fwd_model, self.bwd_model, 
                                                       self.log_p_1.log_prob, x_0, dt, 
                                                       t_max, n_steps, 
                                                       p1_buffer=self.p1_buffer)

                    loss.backward()
                    self.fwd_optim.step()

                    fwd_ema_loss.update(loss.mean().item() / n_steps)
