#!/bin/bash
NAME=$1
OUTDIR=$2

if [ ! -d $OUTDIR ]; then
  mkdir -p $OUTDIR
fi

wget -P $OUTDIR "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/${NAME}"

