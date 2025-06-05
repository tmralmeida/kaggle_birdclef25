# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
import click
import glob
from dotenv import find_dotenv, load_dotenv
import librosa
import pandas as pd
import numpy as np
import random

from .io import create_dir, dump_json_file
import torchaudio


@click.command()
@click.argument("input_directory", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
@click.argument("spectrogram_style", type=click.Choice(["mel", "stft"]))
def main(input_directory, output_directory, spectrogram_style):
    all_specs = []
    processed_meta_df = pd.read_csv(os.path.join(input_directory, "train_filtered.csv"))

    paths = []
    for _, row in processed_meta_df.iterrows():
        primary_label = row["primary_label"]
        filename = os.path.splitext(os.path.basename(row["filename"]))[0]

        pattern = os.path.join(input_directory, primary_label, f"{filename}_clip*.ogg")
        clip_paths = glob.glob(pattern)

        for clip_path in clip_paths:
            paths.append(clip_path)

    random.seed(42)
    random.shuffle(paths)
    train_paths = paths[:int(len(paths) * 0.9)]

    # First pass: compute normalization stats
    print("Computing normalization stats...")

    for fpath in train_paths[:int(len(train_paths) * 0.75)]:
        print(fpath)
        try:
            clip, sr = torchaudio.load(fpath, normalize=True)
            clip = clip.numpy().squeeze()

            if spectrogram_style == "mel":
                spec = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=128, fmax=8000)
                spec = librosa.power_to_db(spec, ref=np.max)
            else:
                stft = librosa.stft(clip, n_fft=2048, hop_length=512)
                spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            all_specs.append(spec)
        except Exception as e:
            print(f"[ERROR: First pass] {fpath}: {e}")

    stacked = np.stack(all_specs)
    mean = float(np.mean(stacked))
    std = float(np.std(stacked))

    create_dir(output_directory)
    dump_json_file(
        {"mean": mean, "std": std}, os.path.join(output_directory, "normalization_stats.json")
    )

    print(f"Normalization stats saved. Mean: {mean}, Std: {std}")
    print("Normalizing data...")
    # Second pass: normalize and save
    for fpath in paths:
        try:
            clip, sr = torchaudio.load(fpath, normalize=True)
            clip = clip.numpy().squeeze()
            animal, fname = fpath.split("/")[-2:]
            if fpath in train_paths:
                out_dir = os.path.join(output_directory, "train", animal)
            else:
                out_dir = os.path.join(output_directory, "val", animal)
            if spectrogram_style == "mel":
                spec = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=128, fmax=8000)
                spec = librosa.power_to_db(spec, ref=np.max)
            else:
                stft = librosa.stft(clip, n_fft=2048, hop_length=512)
                spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            spec = (spec - mean) / std
            out_file = fname.replace(".ogg", ".npy")
            print("Saving:", out_file)
            create_dir(out_dir)
            np.save(os.path.join(out_dir, out_file), spec)
        except Exception as e:
            print(f"[ERROR: Second pass] {fpath}: {e}")


if __name__ == "__main__":
    LOGO_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGO_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
