export IMG_DIR='/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/cache'
export MODEL_DIR="/mnt/nfs/file_server/public/liujia/Models/StableDiffusionXL/SDXL_1_0/sd_xl_base_1.0.safetensors"
export META_FILE="/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/meta_v1_cache.json"
export OUT_DIR="/mnt/nfs/file_server/public/mingjiahui/data/debug/result/base"
export lineart_dir="/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/depth"


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 /home/mingjiahui/anaconda3/envs/T2I/bin/accelerate launch --main_process_port 29590 sdxl_train_adapter_with_text_cache.py --pretrained_model_name_or_path=$MODEL_DIR \
                                                --in_json $META_FILE \
                                                --learning_rate=1e-5 --train_batch_size=1 \
                                                --diffusers_xformers --gradient_checkpointing \
                                                --optimizer_type="AdamW" \
                                                --save_every_n_steps=5000 \
                                                --mixed_precision=bf16 \
                                                --max_train_steps=480000 \
                                                --full_bf16 \
                                                --train_data_dir=$IMG_DIR \
                                                --output_dir=$OUT_DIR \
                                                --logging_dir=$OUT_DIR/log \
                                                --log_with='all' \
                                                --cache_latents \
                                                --text_encoder_cache \
                                                --caption_dropout_rate=0.1 \
                                                --lineart_dir=$lineart_dir  \
                                                --log_image_every_n_steps=100 \
                                                --adapter_resume_path='/mnt/nfs/file_server/public/mingjiahui/data/debug/result/at-step00050000.ckpt'

                                                # --adapter_resume_path=None 