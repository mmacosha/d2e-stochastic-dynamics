#!/bin/bash
VERSION=$1
MODEL=$2
OUTDIR=$3

if [ ! -d $OUTDIR ]; then
  mkdir -p $OUTDIR
fi

wget -P $OUTDIR "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan${VERSION}/versions/1/files/stylegan${VERSION}-${MODEL}.pkl"