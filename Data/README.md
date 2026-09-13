# Data Folder

This folder is intentionally empty in the repository.

## Required Dataset

Download the **Resume Dataset** from Kaggle:

https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Place the file here as:

```
data/Resume.csv
```

Expected columns:

| Column       | Description                              |
|--------------|-------------------------------------------|
| ID           | Unique resume identifier                  |
| Resume_str   | Plain text content of the resume          |
| Resume_html  | HTML formatted version of the resume      |
| Category     | Job category / role label                 |

The dataset is **not included** in this repository due to size and licensing.
After placing `Resume.csv` here, run the training notebook or script described
in the main `README.md` to generate the model artifacts inside `models/`.
