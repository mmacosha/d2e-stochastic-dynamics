#!/bin/bash
python train.py --cfg=ablations-sb --wandb=online --name=sb-ll_match_train_var-8gaussains    --overrides="sampler.matching_method=ll,sampler.train_log_var=true"     && \
python train.py --cfg=ablations-sb --wandb=online --name=sb-score_match-8gaussians           --overrides="sampler.matching_method=score,sampler.train_log_var=false" && \
python train.py --cfg=ablations-sb --wandb=online --name=sb-ll_match_no_var_train-8gaussians --overrides="sampler.matching_method=ll,sampler.train_log_var=false"    && \
python train.py --cfg=ablations-sb --wandb=online --name=sb-mean_match-8gaussians            --overrides="sampler.matching_method=mean,sampler.train_log_var=false"
