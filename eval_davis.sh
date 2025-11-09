CUDA_VISIBLE_DEVICES=7 \
python eval_davis.py \
--img_size 480 \
--ensemble_size 8 \
--k 24 \
--t 90 \
--cd \
--ensemble_size 1 \
--output_dir ./results/ft_24_t_90_480_cd_final/

CUDA_VISIBLE_DEVICES=7 python /mnt/nvme0n1/chaofan/dataset/davis17/davis2017-evaluation/evaluation_method.py \
    --task semi-supervised \
    --results_path ./results/ft_24_t_90_960_cd_final \
    --davis_path /mnt/nvme0n1/chaofan/dataset/davis17/davis-2017/DAVIS/