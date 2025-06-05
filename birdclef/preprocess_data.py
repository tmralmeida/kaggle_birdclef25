# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import librosa
import soundfile as sf
from .io import create_dir


MIN_CLIP_DURATION = 3
MAX_CLIP_DURATION = 5
MINIMUM_QUALITY = 3.0


def split_audio(y, sr, min_clip_duration=5, max_clip_duration=10):
    max_samples_per_clip = sr * max_clip_duration
    min_samples_per_clip = sr * min_clip_duration
    clips = []
    for i in range(0, len(y), max_samples_per_clip):
        clip = y[i: i + max_samples_per_clip]

        if len(clip) >= min_samples_per_clip:
            if len(clip) < max_samples_per_clip:
                clip = librosa.util.fix_length(clip, size=max_samples_per_clip)
            clips.append(clip)
    return clips


@click.command()
@click.argument("input_directory", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
def main(input_directory, output_directory):
    meta_df = pd.read_csv(os.path.join(input_directory, "train.csv"))
    filtered_df = meta_df[(meta_df["rating"] >= MINIMUM_QUALITY) | (meta_df["rating"] == 0)]
    filtered_df.to_csv(os.path.join(output_directory, "train_filtered.csv"))
    filtered_meta_paths = filtered_df["filename"].values

    train_audio_path = os.path.join(input_directory, "train_audio")
    animals_files = os.listdir(train_audio_path)
    for animal in animals_files:
        animal_path = os.path.join(train_audio_path, animal)
        if os.path.isdir(animal_path):
            print(f"Animal: {animal}, Number of files: {len(os.listdir(animal_path))}")
            for _file in os.listdir(animal_path):
                meta_path = os.path.join(animal, _file)
                if meta_path in filtered_meta_paths:
                    file_path = os.path.join(animal_path, _file)
                    y, sr = librosa.load(file_path, sr=None)
                    if len(y) > 0:
                        print(
                            f"File: {_file}, Duration: {librosa.get_duration(y=y, sr=sr)} seconds"
                        )
                        clips = split_audio(
                            y,
                            sr,
                            min_clip_duration=MIN_CLIP_DURATION,
                            max_clip_duration=MAX_CLIP_DURATION,
                        )
                        for idx, clip in enumerate(clips):
                            # Prepare output path
                            output_animal_dir = os.path.join(output_directory, animal)
                            create_dir(output_animal_dir)
                            clip_filename = f"{os.path.splitext(_file)[0]}_clip{idx}.ogg"
                            output_path = os.path.join(output_animal_dir, clip_filename)

                            sf.write(output_path, clip, sr)
                            print(f"Saved clip to {output_path}")
                    else:
                        print(f"File: {_file} is empty or could not be loaded.")
                else:
                    print(f"File: {_file} is not in filtered metadata paths.")


if __name__ == "__main__":
    LOGO_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGO_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
