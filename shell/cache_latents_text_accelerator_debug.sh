export NULL_TORKE_SAVE_DIR="/mnt/nfs/file_server/public/mingjiahui/data/debug/cache"
export MODEL_DIR="/mnt/nfs/file_server/public/liujia/Models/StableDiffusionXL/SDXL_1_0/sd_xl_base_1.0.safetensors"
export META_FILE="/mnt/nfs/file_server/public/mingjiahui/data/debug/meta_debug.json"     # hires_cache_dir.json
export OUT_META_FILE="/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/meta_accelerate_cache_debug.json"   # hires_cache_dir_cache.json
export CACHE_DIR="/mnt/nfs/file_server/public/mingjiahui/data/LAION12M-highreso/cache"
CUDA_VISIBLE_DEVICES=0,2, /home/mingjiahui/anaconda3/envs/T2I/bin/accelerate launch --main_process_port 29591 \
                                    prepare_text_data_accelerate.py \
                                    $META_FILE \
                                    $OUT_META_FILE \
                                    $MODEL_DIR \
                                    --max_resolution 1024,1024 \
                                    --batch_size 2 \
                                    --skip_existing \
                                    --max_bucket_reso 1280 \
                                    --min_bucket_reso 512 \
                                    --cache_dir $CACHE_DIR \
                                    --full_path \
                                    --max_data_loader_n_workers 1 \
#                                    --null_token \
#                                    --null_word_path $NULL_WORD_PATH \
#