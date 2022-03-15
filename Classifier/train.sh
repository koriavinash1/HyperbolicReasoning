# python train.py \
#         --data_root /vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq \
#         --nclasses 3 \
#         --batch_size 16 \
#         --input_size 128 \
#         --model densenet121 \
#         --seed 2022 \
#         --logdir /vol/biomedic2/agk21/PhDLogs/codes/stylEX-extention/Classifier/Logs \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 250 \

# python train.py \
#         --data_root /vol/biomedic2/agk21/PhDLogs/datasets/STL10 \
#         --nclasses 10 \
#         --batch_size 32 \
#         --input_size 128 \
#         --model densenet121 \
#         --seed 2022 \
#         --logdir /vol/biomedic2/agk21/PhDLogs/codes/stylEX-extention/Classifier/Logs/STL10 \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 250 \



# python train.py \
#         --data_root /vol/biomedic2/agk21/PhDLogs/datasets/MorphoMNISTv2/TSWI/data  \
#         --nclasses 10 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model vanilla \
#         --seed 2022 \
#         --logdir /vol/biomedic2/agk21/PhDLogs/codes/stylEX-extention/Classifier/Logs/MorphoMNISTv2/TSWI \
#         --decreasing_lr '20, 30, 40' \
#         --epochs 50 \

# python train.py \
#         --data_root /vol/biomedic2/agk21/PhDLogs/datasets/MorphoMNISTv2/TSWIv2/data  \
#         --nclasses 10 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model vanilla \
#         --seed 2022 \
#         --logdir /vol/biomedic2/agk21/PhDLogs/codes/stylEX-extention/Classifier/Logs/MorphoMNISTv2/TSWIv2 \
#         --decreasing_lr '20, 30, 40' \
#         --epochs 50 \

# python train.py \
#         --data_root /vol/biomedic2/agk21/PhDLogs/datasets/MorphoMNIST2/TS/data  \
#         --nclasses 10 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model vanilla \
#         --seed 2022 \
#         --logdir /vol/biomedic2/agk21/PhDLogs/codes/stylEX-extention/Classifier/Logs/MorphoMNISTv2/TS \
#         --decreasing_lr '10, 20, 30, 40' \
#         --epochs 50 \

# python train.py \
#         --data_root /vol/biomedic2/agk21/PhDLogs/datasets/MorphoMNIST2/IT/data  \
#         --nclasses 10 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model vanilla \
#         --seed 2022 \
#         --logdir /vol/biomedic2/agk21/PhDLogs/codes/stylEX-extention/Classifier/Logs/MorphoMNISTv2/IT \
#         --decreasing_lr '10, 20, 30, 40' \
#         --epochs 50 \


python train.py \
        --data_root /vol/biomedic2/agk21/PhDLogs/datasets/MorphoMNISTv2/TI/data  \
        --nclasses 10 \
        --batch_size 32 \
        --input_size 32 \
        --model vanilla \
        --seed 2022 \
        --logdir /vol/biomedic2/agk21/PhDLogs/codes/stylEX-extention/Classifier/Logs/MorphoMNISTv2/TI \
        --decreasing_lr '10, 20, 30, 40' \
        --epochs 50 \