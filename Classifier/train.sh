python train.py \
        --data_root ../../data  \
        --nclasses 10 \
        --batch_size 32 \
        --input_size 32 \
        --model vanilla \
        --seed 2022 \
        --logdir ../TrainedClassifier/Logs/MorphoMNISTv2/TI \
        --decreasing_lr '10, 20, 30, 40' \
        --epochs 50 \