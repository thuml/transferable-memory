#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
cd ..
nohup python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths datasets/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths datasets/moving-mnist-example/moving-mnist-test.npz \
    --save_dir checkpoints/convlstm_meta_test_maml \
    --gen_frm_dir results/cconvlstm_meta_test_maml \
    --model_name metaconvlstm \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 64,64,64,64 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 1 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr 0.0001 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 1000 \
    --snapshot_interval 1000 > logs/meta_convlstm_mnist2_test_maml.log 2>&1 &
    #--pretrained_model checkpoints/mnist_predrnn_t \
    #--pretrained_model checkpoints/convlstm_mnist_2/model.ckpt-80000 \