CUDA_VISIBLE_DEVICES=0 python main.py --n_GPUs 1 --mode mitnet --lr 2e-4 --save mitnet_its --patch_size 256 --batch_size 8 --loss 1*L1 --lr_decay 200 --save_model