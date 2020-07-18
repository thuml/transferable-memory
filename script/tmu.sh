#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
cd ..
nohup python -u run.py \
    --is_training 2 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths ../predrnn-pytorch/datasets/moving-mnist-3-example/moving-mnist-train.npz \
    --valid_data_paths ../predrnn-pytorch/datasets/moving-mnist-3-example/moving-mnist-valid.npz \
    --save_dir checkpoints/113_TMF_mnist \
    --gen_frm_dir results/113_TMF_mnist \
    --model_name adaptive_model \
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
    --lr 0.0003 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 1000 \
    --pretrained_model ../predrnn-pytorch/checkpoints/convlstm_mnist_2/model.ckpt-80000 \
    --pretrained_model1 ../predrnn-pytorch/checkpoints/convlstm_mnist_1/model.ckpt-80000 \
    --snapshot_interval 1000 > logs/113_TMF_mnist.log 2>&1 &
    #--pretrained_model checkpoints/mnist_predrnn_t \
    #--pretrained_model checkpoints/convlstm_mnist_2/model.ckpt-80000 \
