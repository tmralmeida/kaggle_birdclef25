# kaggle_birdclef25

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Solution for the BirdCLEF+ 2025 kaggle competition without much preparation.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         birdclef and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── birdclef   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes birdclef a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Preprocess and train

To preprocess the data (i.e., split the data into 5-s segments):

```
python -m birdclef.preprocess_data PATH_TO_RAW_DATA PATH_TO_OUTPUT_DIRECTORY_P1
```

To normalize the data and split the data into train/val splits:

```
python -m birdclef.normalize_data PATH_TO_OUTPUT_DIRECTORY_P1 PATH_TO_OUTPUT_DIRECTORY_P2 SPECTOGRAM_STYLE
```

where, `PATH_TO_OUTPUT_DIRECTORY_P1` is in [`mel`, `stft`].

To train the model:

```
python -m birdclef.train_model
```

## Note

I only had the time to try mel spectrograms. 
Future work can extend this to the fourier domain.