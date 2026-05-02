# Data Directory

This folder is the expected local home for the Kaggle dataset used by PancreaTrack.

## Expected file path

After downloading and extracting the dataset, place the CSV at:

`data/archive/Debernardi et al 2020 data.csv`

The app will look for that file by default when you run:

```bash
streamlit run app.py
```

## Why the dataset is not committed

The source data comes from Kaggle:

- [Urinary biomarkers for pancreatic cancer](https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer)

To keep the repository lightweight and avoid redistributing downloaded dataset files through git, the actual CSV is kept local and excluded from version control.

## Setup reminder

1. Download the dataset from Kaggle.
2. Extract the archive.
3. Place `Debernardi et al 2020 data.csv` inside `data/archive/`.

If the file is missing, the Streamlit app will show an upload fallback in the UI.
