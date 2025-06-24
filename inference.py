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
import torch.nn as nn
from utils import demix_track, get_model_from_config

import librosa
from pydub import AudioSegment

import warnings

warnings.filterwarnings("ignore")


# --- 新增: BPM 和调性分析的辅助函数 ---
def analyze_audio_properties(audio_mono, sr):
    """
    分析单声道音频的BPM和调性。

    Args:
        audio_mono (np.ndarray): 单声道音频波形数据。
        sr (int): 采样率。

    Returns:
        str: 格式化后的BPM和调性字符串，例如 "_120bpm_C-Major"。
             如果分析失败，则返回空字符串。
    """
    try:
        # 1. BPM 检测
        # librosa.beat.tempo 返回一个包含估计速度的数组，我们取第一个
        bpm = librosa.beat.tempo(y=audio_mono, sr=sr)[0]
        bpm_str = f"{int(round(bpm))}bpm"

        # 2. 调性检测
        # 首先计算色度图 (chromagram)
        chroma = librosa.feature.chroma_stft(y=audio_mono, sr=sr)
        # 使用 librosa 的内置函数估算调性
        key_note, key_mode = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(audio_mono), sr=sr
        )[-2:]
        pitches = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_name = pitches[int(round(key_note.mean()))]
        mode_name = "Major" if key_mode.mean() > 0 else "Minor"
        # 将 '#' 替换为 's' 以便在文件名中使用
        key_str = f"{key_name.replace('#', 's')}-{mode_name}"

        print(f"  - Detected: {bpm:.1f} BPM, Key: {key_str}")
        return f"_{bpm_str}_{key_str}"

    except Exception as e:
        print(f"  - Could not analyze BPM/key: {e}")
        return ""


def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()

    print(f"正在搜索 '{args.input_folder}' 及其子目录下的 .wav 和 .mp3 文件...")
    search_path_wav = os.path.join(args.input_folder, "**", "*.wav")
    search_path_mp3 = os.path.join(args.input_folder, "**", "*.mp3")

    all_mixtures_path = glob.glob(search_path_wav, recursive=True) + glob.glob(
        search_path_mp3, recursive=True
    )

    total_tracks = len(all_mixtures_path)
    if total_tracks == 0:
        print(f"在目录 '{args.input_folder}' 及其子目录中没有找到 .wav 或 .mp3 文件。")
        return

    print("Total tracks found: {}".format(total_tracks))

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    first_chunk_time = None

    for track_number, path in enumerate(all_mixtures_path, 1):
        print(
            f"\nProcessing track {track_number}/{total_tracks}: {os.path.normpath(path)}"
        )

        relative_path = os.path.relpath(path, args.input_folder)
        relative_path_no_ext, _ = os.path.splitext(relative_path)

        output_dir = os.path.join(args.store_dir, os.path.dirname(relative_path_no_ext))
        os.makedirs(output_dir, exist_ok=True)

        base_filename = os.path.basename(relative_path_no_ext)

        try:
            mix, sr = librosa.load(path, sr=None, mono=False)
            mix = mix.T
        except Exception as e:
            print(f"  - Error loading file {path}: {e}")
            continue

        # --- 新增: 调用分析函数并构建新的文件名 ---
        # librosa的分析函数在单声道上效果最好
        mix_mono_for_analysis = librosa.to_mono(mix.T)
        analysis_suffix = analyze_audio_properties(mix_mono_for_analysis, sr)
        # 将分析结果添加到基础文件名后
        output_base_filename = f"{base_filename}{analysis_suffix}"

        original_mono = False
        if mix.ndim == 1:
            original_mono = True
            mix = np.stack([mix, mix], axis=-1)
        elif mix.shape[1] == 1:
            original_mono = True
            mix = np.concatenate([mix, mix], axis=1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        if first_chunk_time is not None:
            total_length = mixture.shape[1]
            num_chunks = (
                total_length
                + config.inference.chunk_size // config.inference.num_overlap
                - 1
            ) // (config.inference.chunk_size // config.inference.num_overlap)
            estimated_total_time = first_chunk_time * num_chunks
            print(
                f"Estimated total processing time for this track: {estimated_total_time:.2f} seconds"
            )
            sys.stdout.write(
                f"Estimated time remaining: {estimated_total_time:.2f} seconds\r"
            )
            sys.stdout.flush()

        res, first_chunk_time = demix_track(
            config, model, mixture, device, first_chunk_time
        )

        for instr in instruments:
            vocals_output = res[instr].T
            if original_mono:
                vocals_output = vocals_output[:, 0]

            # --- 修改: 使用带有BPM和调性的新文件名来保存MP3 ---
            vocals_path_mp3 = os.path.join(
                output_dir, f"{output_base_filename}_{instr}.mp3"
            )

            vocals_int16 = (vocals_output * 32767).astype(np.int16)

            num_channels = 1 if original_mono else 2
            audio_segment = AudioSegment(
                vocals_int16.tobytes(),
                frame_rate=sr,
                sample_width=vocals_int16.dtype.itemsize,
                channels=num_channels,
            )

            audio_segment.export(vocals_path_mp3, format="mp3", bitrate="192k")
            print(f"  - Saved vocal to: {os.path.normpath(vocals_path_mp3)}")

    time.sleep(1)
    print("\nElapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mel_band_roformer")
    parser.add_argument("--config_path", type=str, help="path to config yaml file")
    parser.add_argument(
        "--model_path", type=str, default="", help="Location of the model"
    )
    parser.add_argument("--input_folder", type=str, help="folder with songs to process")
    parser.add_argument(
        "--store_dir", default="", type=str, help="path to store model outputs"
    )
    parser.add_argument(
        "--device_ids", nargs="+", type=int, default=0, help="list of gpu ids"
    )
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model = get_model_from_config(args.model_type, config)
    if args.model_path != "":
        print("Using model: {}".format(args.model_path))
        model.load_state_dict(
            torch.load(args.model_path, map_location=torch.device("cpu"))
        )

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids) == int:
            device = torch.device(f"cuda:{device_ids}")
            model = model.to(device)
        else:
            device = torch.device(f"cuda:{device_ids[0]}")
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = "cpu"
        print("CUDA is not available. Run inference on CPU. It will be very slow...")
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
