CUDA_VISIBLE_DEVICES=7 \
python eval_davis.py \
--img_size 960 \
--k 24 \
--t 90 \
--cd \
--output_dir ./results/ft_24_t_90_960_cd_final/

CUDA_VISIBLE_DEVICES=7 python /mnt/nvme0n1/chaofan/dataset/davis17/davis2017-evaluation/evaluation_method.py \
    --task semi-supervised \
    --results_path ./results/ft_24_t_90_960_cd_final \
    --davis_path /mnt/nvme0n1/chaofan/dataset/davis17/davis-2017/DAVIS/