export IMG_DIR='/mnt/nfs/file_server/public/mingjiahui/data/debug/image'
export META_FILE="/mnt/nfs/file_server/public/mingjiahui/data/debug/meta_debug.json"
#export IN_META_FILE="/mnt/nfs/file_server/public/mingjiahui/data/debug/hires_cache_dir.json"
python merge_captions_to_metadata.py $IMG_DIR $META_FILE --caption_extention='.txt' --full_path 