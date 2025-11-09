CUDA_VISIBLE_DEVICES=6 python eval_spair.py \
--dataset spair \
--dataset_path /mnt/nvme0n1/chaofan/dataset/SPair-71k \
--save_path Features/spair_flux_final \
--dit_model flux \
--img_size 640 640 \
--t 260 \
--k 28 \
--ensemble_size 8 \
--cd