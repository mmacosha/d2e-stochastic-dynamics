#!/bin/bash

# Ensure we're launching from the project root
cd "$(dirname "$0")/../.."

# train --seed=42 --device=mps --cfg=d2d-langevin --name=sb-d2d_langevin-normal_to_gmm-match_ll-train_log_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,buffer.buffer_size=10000,buffer.num_langevin_steps=50'
train --seed=43 --device=mps --cfg=d2d-langevin --name=sb-d2d_langevin-normal_to_gmm-match_ll-train_log_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,buffer.buffer_size=10000,buffer.num_langevin_steps=50'
train --seed=44 --device=mps --cfg=d2d-langevin --name=sb-d2d_langevin-normal_to_gmm-match_ll-train_log_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,buffer.buffer_size=10000,buffer.num_langevin_steps=50'
train --seed=45 --device=mps --cfg=d2d-langevin --name=sb-d2d_langevin-normal_to_gmm-match_ll-train_log_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,buffer.buffer_size=10000,buffer.num_langevin_steps=50'
train --seed=46 --device=mps --cfg=d2d-langevin --name=sb-d2d_langevin-normal_to_gmm-match_ll-train_log_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,buffer.buffer_size=10000,buffer.num_langevin_steps=50'