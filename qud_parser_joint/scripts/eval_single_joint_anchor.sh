export CUDA_VISIBLE_DEVICES=4

python open_instruct/predict.py \
    --model_name_or_path output/single_joint_7B_lora/ \
    --input_files data/processed/single_joint_anchor_val.jsonl \
    --output_file data/processed/single_joint_anchor_val_outputs.jsonl \
    --batch_size 1 \
    --load_in_8bit \
    --num_return_sequences 5 \
    --stop_sequences answering