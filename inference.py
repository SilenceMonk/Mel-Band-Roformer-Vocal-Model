import argparse
import yaml
import numpy as np
import time
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import torch.nn as nn
from utils import demix_track, get_model_from_config

import librosa


import warnings
warnings.filterwarnings("ignore")


def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()

    # --- 变更 2: 搜索 .wav 和 .mp3 文件 ---
    # 分别搜索两种格式，然后合并列表
    print(f"正在搜索 '{args.input_folder}' 及其子目录下的 .wav 和 .mp3 文件...")
    search_path_wav = os.path.join(args.input_folder, '**', '*.wav')
    search_path_mp3 = os.path.join(args.input_folder, '**', '*.mp3')
    
    all_mixtures_path = glob.glob(search_path_wav, recursive=True) + glob.glob(search_path_mp3, recursive=True)
    
    total_tracks = len(all_mixtures_path)
    if total_tracks == 0:
        print(f"在目录 '{args.input_folder}' 及其子目录中没有找到 .wav 或 .mp3 文件。")
        return
        
    print('Total tracks found: {}'.format(total_tracks))

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    first_chunk_time = None

    for track_number, path in enumerate(all_mixtures_path, 1):
        # 使用 os.path.normpath 来标准化路径显示
        print(f"\nProcessing track {track_number}/{total_tracks}: {os.path.normpath(path)}")

        relative_path = os.path.relpath(path, args.input_folder)
        # 获取不带扩展名的相对路径，用于构建输出文件名
        relative_path_no_ext, _ = os.path.splitext(relative_path)
        
        output_dir = os.path.join(args.store_dir, os.path.dirname(relative_path_no_ext))
        os.makedirs(output_dir, exist_ok=True) 

        base_filename = os.path.basename(relative_path_no_ext)

        # --- 变更 3: 使用 librosa 读取音频文件 ---
        # librosa.load 可以处理 wav, mp3 等多种格式
        # sr=None 会保留文件的原始采样率
        # mono=False 会保留原始的通道数（立体声/单声道）
        try:
            mix, sr = librosa.load(path, sr=None, mono=False)
            # librosa 返回的形状是 (channels, samples)，而模型代码期望 (samples, channels)
            # 所以我们需要进行转置
            mix = mix.T
        except Exception as e:
            print(f"  - Error loading file {path}: {e}")
            continue # 跳过损坏或无法读取的文件
        
        # --- 变更 4: 更新单声道的判断逻辑 ---
        # 因为 librosa 加载单声道文件后的 shape 是 (n_samples, 1)
        original_mono = False
        if mix.ndim == 1:
            original_mono = True
            mix = np.stack([mix, mix], axis=-1)
        elif mix.shape[1] == 1:
            original_mono = True
            # 将 (n_samples, 1) 形状的单声道数据复制为 (n_samples, 2)
            mix = np.concatenate([mix, mix], axis=1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        if first_chunk_time is not None:
            total_length = mixture.shape[1]
            num_chunks = (total_length + config.inference.chunk_size // config.inference.num_overlap - 1) // (config.inference.chunk_size // config.inference.num_overlap)
            estimated_total_time = first_chunk_time * num_chunks
            print(f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds")
            sys.stdout.write(f"Estimated time remaining: {estimated_total_time:.2f} seconds\r")
            sys.stdout.flush()

        res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)

        for instr in instruments:
            vocals_output = res[instr].T
            if original_mono:
                # 取一个声道作为输出即可
                vocals_output = vocals_output[:, 0]

            # 输出文件总是保存为 .wav 格式
            vocals_path = os.path.join(output_dir, f"{base_filename}_{instr}.wav")
            sf.write(vocals_path, vocals_output, sr, subtype='FLOAT')

        # --- 变更 5: 重新加载原始音频以计算伴奏 ---
        # 使用 librosa 加载，并确保与处理时的数据类型和形状一致
        original_mix, _ = librosa.load(path, sr=sr, mono=original_mono)
        # 如果是立体声，需要转置
        if not original_mono:
            original_mix = original_mix.T

        vocals_output = res[instruments[0]].T
        if original_mono:
            vocals_output = vocals_output[:, 0]
        
        instrumental = original_mix - vocals_output

        instrumental_path = os.path.join(output_dir, f"{base_filename}_instrumental.wav")
        sf.write(instrumental_path, instrumental, sr, subtype='FLOAT')

    time.sleep(1)
    print("\nElapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mel_band_roformer')
    parser.add_argument("--config_path", type=str, help="path to config yaml file")
    parser.add_argument("--model_path", type=str, default='', help="Location of the model")
    parser.add_argument("--input_folder", type=str, help="folder with songs to process")
    parser.add_argument("--store_dir", default="", type=str, help="path to store model outputs")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
      config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model = get_model_from_config(args.model_type, config)
    if args.model_path != '':
        print('Using model: {}'.format(args.model_path))
        model.load_state_dict(
            torch.load(args.model_path, map_location=torch.device('cpu'))
        )

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids)==int:
            device = torch.device(f'cuda:{device_ids}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not available. Run inference on CPU. It will be very slow...')
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
