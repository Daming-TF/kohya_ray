import argparse
import os
import json
import multiprocessing

from accelerate import Accelerator
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
import library.sdxl_train_util as sdxl_train_util
import library.sdxl_model_util as sdxl_model_util
import fcntl

import library.model_util as model_util
import library.train_util as train_util

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


# def acquire_file_lock(file_path):
#     lock_file = open(file_path, 'r', encoding='utf-8')
#     fcntl.flock(lock_file, fcntl.LOCK_EX)  # Acquire an exclusive lock
#     return lock_file
#
#
# def release_file_lock(file):
#     fcntl.flock(file, fcntl.LOCK_UN)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def get_latents(vae, key_and_images, tokenizers, text_encoders, weight_dtype=torch.float32, cache_dir=None):
    img_tensors = [IMAGE_TRANSFORMS(image) for _, image in key_and_images]
    img_tensors = torch.stack(img_tensors)
    img_tensors = img_tensors.to(args.device, weight_dtype)
    with torch.no_grad():
        latents = vae.module.encode(img_tensors).latent_dist.sample()

    # check NaN
    for (key, _), latents1 in zip(key_and_images, latents):
        if torch.isnan(latents1).any():
            raise ValueError(f"NaN detected in latents of {key}")
    text_latents_hid_1 = []
    text_latents_hid_2 = []
    text_latents_hid_pool2 = []
    for key, image in key_and_images:
        caption = open(os.path.splitext(key)[0] + ".txt", "r", encoding="utf-8").read().strip()
        input_id1 = get_input_ids(caption, tokenizers[0]).to(args.device)
        input_id2 = get_input_ids(caption, tokenizers[1]).to(args.device)
        with torch.no_grad():
            encoder_hidden_states1, encoder_hidden_states2, pool2 = sdxl_train_util.get_hidden_states(
                args,
                input_id1,
                input_id2,
                tokenizers[0],
                tokenizers[1],
                text_encoders[0],
                text_encoders[1],
                torch.float16,
            )
            encoder_hidden_states1 = encoder_hidden_states1.detach().to("cpu").squeeze(0)  # n*75+2,768
            encoder_hidden_states2 = encoder_hidden_states2.detach().to("cpu").squeeze(0)  # n*75+2,1280
            pool2 = pool2.detach().to("cpu").squeeze(0)  # 1280
            text_latents_hid_1.append(encoder_hidden_states1)
            text_latents_hid_2.append(encoder_hidden_states2)
            text_latents_hid_pool2.append(pool2)
    return latents, text_latents_hid_1, text_latents_hid_2, text_latents_hid_pool2


def get_npz_filename_wo_ext(data_dir, image_key, is_full_path, flip, recursive, is_text=False):
    if is_full_path:
        base_name = os.path.splitext(os.path.basename(image_key))[0] if not is_text else \
        os.path.splitext(os.path.basename(image_key))[0] + "_text"

        relative_path = os.path.relpath(os.path.dirname(image_key), data_dir)
    else:
        base_name = image_key
        relative_path = ""

    if flip:
        base_name += "_flip"

    if recursive and relative_path:
        return os.path.join(data_dir, relative_path, base_name)
    else:
        return os.path.join(data_dir, base_name)


def get_input_ids(caption, tokenizer):
    input_ids = tokenizer(
        caption, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt"
    ).input_ids
    return input_ids


def get_null_token(args, tokenizers, text_encoders):
    caption = ""
    input_id1 = get_input_ids(caption, tokenizers[0]).to(args.device)
    input_id2 = get_input_ids(caption, tokenizers[1]).to(args.device)

    with torch.no_grad():
        encoder_hidden_states1, encoder_hidden_states2, pool2 = sdxl_train_util.get_hidden_states(
            args,
            input_id1,
            input_id2,
            tokenizers[0],
            tokenizers[1],
            text_encoders[0],
            text_encoders[1],
            torch.float16,
        )
        encoder_hidden_states1 = encoder_hidden_states1.detach().to("cpu").squeeze(0)  # n*75+2,768
        encoder_hidden_states2 = encoder_hidden_states2.detach().to("cpu").squeeze(0)  # n*75+2,1280
        pool2 = pool2.detach().to("cpu").squeeze(0)  # 1280
        text_latent = {
            'hid1': encoder_hidden_states1.cpu().numpy(),
            'hid2': encoder_hidden_states2.cpu().numpy(),
            'pool': pool2.cpu().numpy(),
        }
    npz_file_name = os.path.join(os.path.dirname(args.train_data_dir), 'null_token.npy')
    np.save(npz_file_name, text_latent)
    print(f"saved null token latent to {npz_file_name}")
    return


def save_metadata(meta_data, lock, json_path):
    lock_enable = False
    # with lock:
    try:
        # if os.path.exists(json_path):
        file = open(json_path, 'r+', encoding="utf-8")
        fd = file.fileno()
        fcntl.flock(fd, fcntl.LOCK_EX)
        # lock_enable = True
        data = json.load(file)
        for key, value in data.items():
            meta_data[key] = value

        json.dump(meta_data, file, indent=2)

        # if os.path.exists(json_path):
        #     with open(json_path, "r", encoding='utf-8') as file:
        #         data = json.load(json_path)
        #
        #     for key, value in data.items():
        #             meta_data[key] = value
                # else:
                #     print(f'some image in two processing  >>  {key}')

        # with open(json_path, "w", encoding="utf-8") as file:
        #     json.dump(meta_data, file, indent=2)

        # if lock_enable:
            # release_file_lock(lock_file)
    finally:
        # if lock_enable:
        file.close()
        fcntl.flock(fd, fcntl.LOCK_UN)



def _init(args):
    # check reso steps
    if args.bucket_reso_steps % 8 > 0:
        print(f"resolution of buckets in training time is a multiple of 8")
        exit(0)

    # load input json
    if os.path.exists(args.in_json):
        print(f"loading existing metadata: {args.in_json}")
        with open(args.in_json, "rt", encoding="utf-8") as f:
            metadata = json.load(f)
        image_paths: List[str] = [str(p.strip()) for p in metadata.keys()]
        print(f"found {len(image_paths)} images.")
    else:
        print(f"no metadata / メタデータファイルがありません: {args.in_json}")
        return

    # weight init
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat1
    args.weight_dtype = weight_dtype

    # bucket init
    max_reso = tuple([int(t) for t in args.max_resolution.split(",")])
    assert len(max_reso) == 2, \
        f"illegal resolution (not 'width,height') / 画像サイズに誤りがあります。'幅,高さ'で指定してください: " \
        f"{args.max_resolution}"
    bucket_manager = train_util.BucketManager(
        args.bucket_no_upscale, max_reso, args.min_bucket_reso, args.max_bucket_reso, args.bucket_reso_steps
    )

    if not args.bucket_no_upscale:
        bucket_manager.make_buckets()
    else:
        print(
            "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
        )

    # accelerate
    accelerator = Accelerator()
    args.device = accelerator.device
    lock = multiprocessing.Lock()

    return metadata, image_paths, bucket_manager, accelerator, lock


def _load_model(args):
    (text_encoder1, text_encoder2, vae, _, logit_scale, ckpt_info) = sdxl_model_util.load_models_from_sdxl_checkpoint(
        "sdxl", args.model_name_or_path, 'cpu', need_unet=False)
    text_encoder1.to(args.device)
    text_encoder2.to(args.device)
    vae.to(args.device)
    vae.eval()
    text_encoder1.eval()
    text_encoder2.eval()
    args.tokenizer_cache_dir = None
    args.max_token_length = None
    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    return vae, text_encoder1, text_encoder2, tokenizer1, tokenizer2


def main(args):
    # init
    metadata, image_paths, bucket_manager, accelerator, lock = _init(args)

    # load model
    vae, text_encoder1, text_encoder2, tokenizer1, tokenizer2 = _load_model(args)
    vae, text_encoder1, text_encoder2, tokenizer1, tokenizer2 = \
        accelerator.prepare(vae, text_encoder1, text_encoder2, tokenizer1, tokenizer2)
    tokenizers = [tokenizer1, tokenizer2]
    text_encoders = [text_encoder1, text_encoder2]

    # get null token
    if args.null_token:
        get_null_token()
        exit(0)

    # bucket processing
    img_ar_errors = []

    def process_batch(is_last):
        for bucket in bucket_manager.buckets:
            if (is_last and len(bucket) > 0) or len(bucket) >= args.batch_size:
                latents, hid1s, hid2s, pools = get_latents(vae, [(key, img) for key, img, _, _ in bucket], tokenizers,
                                                           text_encoders, args.weight_dtype, cache_dir=args.cache_dir)
                assert (
                        latents.shape[2] == bucket[0][1].shape[0] // 8 and latents.shape[3] == bucket[0][1].shape[
                    1] // 8
                ), f"latent shape {latents.shape}, {bucket[0][1].shape}"

                for (image_key, _, original_size, crop_left_top), latent, hid1, hid2, pool in zip(bucket, latents,
                                                                                                  hid1s, hid2s, pools):
                    # save img cache
                    npz_file_name = get_npz_filename_wo_ext(args.cache_dir, image_key, args.full_path, False, False)
                    train_util.save_latents_to_disk(npz_file_name, latent, original_size, crop_left_top)
                    # save text cache
                    text_latent = {
                        'hid1': hid1.cpu().numpy(),
                        'hid2': hid2.cpu().numpy(),
                        'pool': pool.cpu().numpy(),
                    }
                    npz_file_name = get_npz_filename_wo_ext(args.cache_dir, image_key, args.full_path, False, False,
                                                            is_text=True)
                    np.save(npz_file_name, text_latent)

                bucket.clear()

    # dataload speed up
    dataset = train_util.ImageLoadingDataset(image_paths)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.max_data_loader_n_workers,
        collate_fn=collate_fn_remove_corrupted,
        drop_last=False,
    )
    dataloader = accelerator.prepare(dataloader)

    bucket_counts = {}
    print(f'adding image ......')
    for data_entry in tqdm(dataloader, smoothing=0.0):
        try:
            if data_entry[0] is None:
                continue

            img_tensor, image_path = data_entry[0]
            if img_tensor is not None:
                image = transforms.functional.to_pil_image(img_tensor)
            else:
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                except Exception as e:
                    print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                    continue
        except IndexError as e:
            print('Some errors is happencing ...... ')
            print(f'Data:{data_entry}')
            print(f'Data Type:{type(data_entry)}')
            continue

        image_key = image_path if args.full_path else f'{os.path.basename(os.path.dirname(image_path))}xyz{os.path.splitext(os.path.basename(image_path))[0]}'
        if image_key not in metadata:
            metadata[image_key] = {}

        # According to the resolution defined by the bucket,
        # the resolution size with the closest aspect ratio is obtained
        reso, resized_size, ar_error = bucket_manager.select_bucket(image.width, image.height)
        img_ar_errors.append(abs(ar_error))
        bucket_counts[reso] = bucket_counts.get(reso, 0) + 1

        # The resolution recorded in the metadata is in latent, so 8 units can be discarded
        metadata[image_key]["train_resolution"] = (reso[0] - reso[0] % 8, reso[1] - reso[1] % 8)
        metadata[image_key]["npz_path"] = f'{args.cache_dir}/{os.path.splitext(os.path.basename(image_path))[0]}.npz'
        metadata[image_key][
            'text_latent'] = f'{args.cache_dir}/{os.path.splitext(os.path.basename(image_path))[0]}_text.npy'
        # del metadata[image_key]['caption']

        if not args.bucket_no_upscale:
            # upscaleを行わないときには、resize後のサイズは、bucketのサイズと、縦横どちらかが同じであることを確認する
            assert (
                    resized_size[0] == reso[0] or resized_size[1] == reso[1]
            ), f"internal error, resized size not match: {reso}, {resized_size}, {image.width}, {image.height}"
            assert (
                    resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
            ), f"internal error, resized size too small: {reso}, {resized_size}, {image.width}, {image.height}"

        assert (
                resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
        ), f"internal error resized size is small: {resized_size}, {reso}"

        # if exist skiping......
        if args.skip_existing:
            npz_files = [
                get_npz_filename_wo_ext(args.cache_dir, image_key, args.full_path, False, args.recursive) + ".npz"]
            if args.flip_aug:
                npz_files.append(
                    get_npz_filename_wo_ext(args.cache_dir, image_key, args.full_path, True, False,
                                            is_text=True) + ".npz"
                )

            found = True
            if len(npz_files) > 1:
                print(len(npz_files))
                exit(0)
            for npz_file in npz_files:
                if not os.path.exists(npz_file):
                    found = False
                    break

                latents, _, _ = train_util.load_latents_from_disk(npz_file)
                if latents is None:  # old version
                    found = False
                    break

                if latents.shape[1] != reso[1] // 8 or latents.shape[2] != reso[0] // 8:  # latentsのshapeを確認
                    found = False
                    break
            if found:
                continue

        # resize imgae
        try:
            image = np.array(image)
        except OSError as e:
            print("image is corrupted, skipping", image_path)
            del metadata[image_key]
            continue
        if resized_size[0] != image.shape[1] or resized_size[1] != image.shape[0]:
            image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)

        # get trim error
        trim_left = 0
        if resized_size[0] > reso[0]:
            trim_size = resized_size[0] - reso[0]
            image = image[:, trim_size // 2: trim_size // 2 + reso[0]]
            trim_left = trim_size // 2
        trim_top = 0
        if resized_size[1] > reso[1]:
            trim_size = resized_size[1] - reso[1]
            image = image[trim_size // 2: trim_size // 2 + reso[1]]
            trim_top = trim_size // 2

        original_size_wh = (resized_size[0], resized_size[1])
        # target_size_wh = (reso[0], reso[1])
        crop_left_top = (trim_left, trim_top)

        assert (
                image.shape[0] == reso[1] and image.shape[1] == reso[0]
        ), f"internal error, illegal trimmed size: {image.shape}, {reso}"

        # bucket: add image accordance to resolution
        bucket_manager.add_image(reso, (image_key, image, original_size_wh, crop_left_top))

        process_batch(False)

    # Deal with the rest
    process_batch(True)

    bucket_manager.sort()
    for i, reso in enumerate(bucket_manager.resos):
        count = bucket_counts.get(reso, 0)
        if count > 0:
            print(f"bucket {i} {reso}: {count}")
    img_ar_errors = np.array(img_ar_errors)
    print(f"mean ar error: {np.mean(img_ar_errors)}")

    # save metadata to json
    print(f"writing metadata: {args.out_json}")
    save_metadata(metadata, lock, args.out_json)
    print("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("in_json", type=str, help="metadata file to input / 読み込むメタデータファイル")
    parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
    parser.add_argument("model_name_or_path", type=str,
                        help="model name or path to encode latents / latentを取得するためのモデル")
    parser.add_argument("--v2", action="store_true",
                        help="not used (for backward compatibility) / 使用されません（互換性のため残してあります）")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",
    )
    parser.add_argument(
        "--max_resolution",
        type=str,
        default="512,512",
        help="max resolution in fine tuning (width,height) / fine tuning時の最大画像サイズ 「幅,高さ」（使用メモリ量に関係します）",
    )
    parser.add_argument("--min_bucket_reso", type=int, default=256,
                        help="minimum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--max_bucket_reso", type=int, default=1024,
                        help="maximum resolution for buckets / bucketの最小解像度")
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します",
    )
    parser.add_argument(
        "--bucket_no_upscale", action="store_true",
        help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
        help="use mixed precision / 混合精度を使う場合、その精度"
    )
    parser.add_argument(
        "--full_path",
        action="store_true",
        help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）",
    )
    parser.add_argument(
        "--flip_aug", action="store_true",
        help="flip augmentation, save latents for flipped images / 左右反転した画像もlatentを取得、保存する"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip images if npz already exists (both normal and flipped exists if flip_aug is enabled) / npzが既に存在する画像をスキップする（flip_aug有効時は通常、反転の両方が存在する画像をスキップ）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursively look for training tags in all child folders of train_data_dir / train_data_dirのすべての子フォルダにある学習タグを再帰的に探す",
    )
    parser.add_argument(
        "--image_list",
        type=str
    )
    parser.add_argument(
        "--cache_dir",
        type=str
    )
    parser.add_argument(
        "--null_token",
        action="store_true",
    )
    parser.add_argument(
        "--null_word_path",
        type=str,
    )
    parser.add_argument(
        "--device",
        type=str,
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
