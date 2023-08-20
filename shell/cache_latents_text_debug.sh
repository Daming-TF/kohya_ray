export NULL_TORKE_SAVE_DIR="/mnt/nfs/file_server/public/mingjiahui/data/debug/cache"
export MODEL_DIR="/mnt/nfs/file_server/public/liujia/Models/StableDiffusionXL/SDXL_1_0/sd_xl_base_1.0.safetensors"
export META_FILE="/mnt/nfs/file_server/public/mingjiahui/data/debug/meta_debug.json"
export OUT_META_FILE="/mnt/nfs/file_server/public/mingjiahui/data/debug/meta_cache_debug.json"
export CACHE_DIR="/mnt/nfs/file_server/public/mingjiahui/data/debug/cache"
CUDA_VISIBLE_DEVICES=6 python prepare_text_data_df.py \
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
                                    --null_token $NULL_TORKE_SAVE_DIR