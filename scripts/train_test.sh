#!/usr/bin/env sh 

for SEED in 0 7 27 42 128 
do
    # ResNet-18
    python ../src/main.py model=1D-ResNet-18 \
        data.input_norm=dataset \
        train.seed=${SEED} \
        train.num_workers=4 \
        data.awgn=None \
        data.mgn=None \
        train.do_testing=True

    # VGG 19
    python ../src/main.py model=1D-VGG-19 \
        data.input_norm=dataset \
        train.seed=${SEED} \
        train.num_workers=4 \
        data.awgn=0.00812 \
        data.mgn=0.011 \
        train.do_testing=True

    # SSFormer EinFFTInd
    python ../src/main.py model=SSFormerv2 \
        data.input_norm=dataset \
        model.cm_type=EinFFTInd \
        train.seed=${SEED} \
        train.num_workers=4 \
        train.do_testing=True \
        model.dropout_1d_rate=0.0 \
        model.ssm_dropout=0.3 \
        model.base_lr=0.0001 \
        model.ds_ratio=1 \
        model.ds_kernel_size=7 \
        'model.embed_dims=[128, 128]' \
        model.patch_dim=64

    # # SSFormer PWInd w/ Age
    python ../src/main.py model=SSFormerv2 \
        data.input_norm=dataset \
        data.use_age=True \
        model.cm_type=PWInd \
        train.seed=${SEED} \
        train.num_workers=4 \
        train.do_testing=True \
        model.dropout_1d_rate=0.0 \
        model.ssm_dropout=0.3 \
        model.base_lr=0.0001 \
        model.ds_ratio=1 \
        model.ds_kernel_size=7 \
        'model.embed_dims=[128, 128]' \
        model.patch_dim=64 \
        model.use_age='fc'

    # SSFormer PWInd w/o Token LN
    python ../src/main.py model=SSFormerv2 \
        data.input_norm=dataset \
        model.cm_type=PWInd \
        train.seed=${SEED} \
        train.num_workers=4 \
        train.do_testing=True \
        model.dropout_1d_rate=0.0 \
        model.ssm_dropout=0.3 \
        model.base_lr=0.0001 \
        model.ds_ratio=1 \
        model.ds_kernel_size=7 \
        'model.embed_dims=[128, 128]' \
        model.patch_dim=64 \
        model.norm_along_tokens=False

    # SSFormerv2 with token LN (Standard SSFormer)
    python ../src/main.py model=SSFormerv2 \
        data.input_norm=dataset \
        model.cm_type=PWInd \
        train.seed=${SEED} \
        train.num_workers=4 \
        train.do_testing=True \
        model.dropout_1d_rate=0.0 \
        model.ssm_dropout=0.3 \
        model.base_lr=0.0001 \
        model.ds_ratio=1 \
        model.ds_kernel_size=7 \
        'model.embed_dims=[128, 128]' \
        model.patch_dim=64 \
        model.norm_along_tokens=True

done