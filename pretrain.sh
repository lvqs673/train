export TOKENIZERS_PARALLELISM="true"
export OMP_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
torchrun --nproc_per_node=4 \
         src/pretrain.py \
         --sentence_path="./data/sentences.txt" \
         --model_save_path="./model" \
         --n_epoch=5 \
         --lr=5e-5 \
         --batch_size=24 \
         --print_interval=50 \
         > stdout.txt
