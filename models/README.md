# Models Folder

This folder is intentionally empty in the repository.

After training (see main `README.md`), the following files will be generated
here automatically:

| File                        | Description                                   |
|-----------------------------|------------------------------------------------|
| `job_category_model.pkl`    | Trained Logistic Regression classifier         |
| `tfidf_vectorizer.pkl`      | Fitted TF-IDF vectorizer used for the classifier|
| `label_encoder.pkl`         | Label encoder mapping categories to class IDs  |
| `metrics.json`              | Saved evaluation metrics from the last training run |

These files are excluded from version control via `.gitignore` because they
are derived artifacts that can always be regenerated from `data/Resume.csv`.
