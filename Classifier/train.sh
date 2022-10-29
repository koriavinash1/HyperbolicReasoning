# python /vol/biomedic2/agk21/PhDLogs/codes/HyperbolicReasoning/Classifier/train.py \
#         --data_root /vol/biomedic2/agk21/PhDLogs/datasets/STL10 \
#         --nclasses 10 \
#         --batch_size 32 \
#         --input_size 128 \
#         --model densenet121 \
#         --seed 2022 \
#         --logdir /vol/biomedic2/agk21/PhDLogs/codes/HyperbolicReasoning/Classifier/Logs/STL10 \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 250 &

# python /vol/biomedic2/agk21/PhDLogs/codes/HyperbolicReasoning/Classifier/train.py \
#         --data_root /vol/biomedic2/agk21/PhDLogs/datasets/AFHQ/afhq \
#         --nclasses 3 \
#         --batch_size 32 \
#         --input_size 128 \
#         --model densenet121 \
#         --seed 2022 \
#         --logdir /vol/biomedic2/agk21/PhDLogs/codes/HyperbolicReasoning/Classifier/Logs/AFHQ \
#         --decreasing_lr '80, 120, 160, 200' \
#         --epochs 250 &

# python /vol/biomedic2/agk21/PhDLogs/codes/HyperbolicReasoning/Classifier/train.py \
#         --data_root /vol/biomedic2/agk21/PhDLogs/datasets/MNIST \
#         --nclasses 10 \
#         --batch_size 32 \
#         --input_size 32 \
#         --model vanilla \
#         --seed 2022 \
#         --logdir /vol/biomedic2/agk21/PhDLogs/codes/HyperbolicReasoning/Classifier/Logs/MNIST \
#         --decreasing_lr '10, 20, 30, 40' \
#         --epochs 50  &

python /vol/biomedic2/agk21/PhDLogs/codes/HyperbolicReasoning/Classifier/train.py \
        --data_root /vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/ \
        --nclasses 2 \
        --batch_size 32 \
        --input_size 224 \
        --model densenet121 \
        --seed 2022 \
        --logdir /vol/biomedic2/agk21/PhDLogs/codes/HyperbolicReasoning/Classifier/Logs/MIMIC \
        --decreasing_lr '80, 120, 160, 200' \
        --epochs 250 \
        --binary True