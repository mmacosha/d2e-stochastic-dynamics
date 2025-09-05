#!/bin/bash
# ll matching learnt var
train --seed=42 --device=mps --cfg=d2e-ablate --name=sb-d2e-normal_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=43 --device=mps --cfg=d2e-ablate --name=sb-d2e-normal_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=44 --device=mps --cfg=d2e-ablate --name=sb-d2e-normal_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=45 --device=mps --cfg=d2e-ablate --name=sb-d2e-normal_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=46 --device=mps --cfg=d2e-ablate --name=sb-d2e-normal_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll'

# ll matching fixed var
train --seed=42 --device=mps --cfg=d2e-ablate --name=sb-d2e-normal_to_gmm-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=43 --device=mps --cfg=d2e-ablate --name=sb-d2e-normal_to_gmm-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=44 --device=mps --cfg=d2e-ablate --name=sb-d2e-normal_to_gmm-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=45 --device=mps --cfg=d2e-ablate --name=sb-d2e-normal_to_gmm-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=46 --device=mps --cfg=d2e-ablate --name=sb-d2e-normal_to_gmm-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll'

# ll matching learnt var moons <--> gmm
train --seed=42 --device=mps --cfg=d2e-ablate --name=sb-d2e-moons_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=43 --device=mps --cfg=d2e-ablate --name=sb-d2e-moons_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=44 --device=mps --cfg=d2e-ablate --name=sb-d2e-moons_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=45 --device=mps --cfg=d2e-ablate --name=sb-d2e-moons_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=46 --device=mps --cfg=d2e-ablate --name=sb-d2e-moons_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll'
