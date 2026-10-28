# JobGuard AI — Fake Job Posting Detector

A beginner-friendly AI Application Lab project using Python, Pandas, NumPy, Streamlit and scikit-learn.

## Important note

The app starts in **Built-in demo examples** mode so the interface works immediately. The included examples are deliberately tiny and illustrative, not a scientifically valid training dataset. For a credible project evaluation, download a labelled dataset and use **Upload a CSV dataset** in the sidebar.

## Dataset

A common choice is the Real / Fake Job Posting Prediction dataset based on EMSCAD, available on Kaggle:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Use the CSV that contains a `fraudulent` column. Label meaning:
- `0` = legitimate
- `1` = fraudulent

The dataset may contain personal or company text; do not upload private data. Read the dataset's license and source terms.

## Run locally (Windows)

1. Install Python 3.10+ from https://www.python.org/downloads/ (check "Add Python to PATH").
2. Extract this ZIP and open the folder in VS Code.
3. Open Terminal in VS Code.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```

5. Install libraries:
   ```bash
   python -m pip install -r requirements.txt
   ```
6. Run:
   ```bash
   streamlit run app.py
   ```
7. Your browser should open a local address, normally `http://localhost:8501`.

## Make it public

1. Create/sign in to GitHub.
2. Create a new repository named `jobguard-ai`.
3. Upload `app.py`, `requirements.txt`, and optionally `README.md`. Do not upload `.venv`, credentials, or private data.
4. Open https://share.streamlit.io/ and connect GitHub.
5. Choose **Create app**, select the repository, branch `main`, and entrypoint `app.py`, then deploy.
6. Streamlit will give you a public `*.streamlit.app` URL. Test it in a private/incognito window and on a phone before submitting.

The hosted app starts with the demo examples. For real dataset-backed predictions, upload the labelled CSV through the sidebar. For a class demo, it is usually better to include the data file in a repository only if its size/license and your instructor's requirements allow it. Never claim demo-mode results are validated.

## Libraries and why

- `pandas`: read CSV files, clean missing values, combine columns.
- `numpy`: numerical computing (included as a standard data-science dependency).
- `scikit-learn`: TF-IDF, Logistic Regression, train/test split, and metrics.
- `streamlit`: user interface and hosting-friendly app structure.
- `matplotlib`: plotting dependency; Streamlit's built-in chart is used for the simple class-count plot.
- `re`: Python standard library for simple text patterns.

## Pipeline

CSV data → text cleaning → combine job fields → TF-IDF word/phrase vectors → Logistic Regression → prediction + evaluation.

## Syllabus mapping

- Unit 1 Experiment 1: preprocessing, Pandas/NumPy.
- Unit 1 Experiment 2: class counts and visualization / EDA.
- Unit 1 Experiment 3: supervised learning.
- Unit 1 Experiment 4: binary classification.
- Unit 3 Experiment 1/3: NLP and text processing concepts.

## Limits

A model can make mistakes, and a high score on one dataset does not guarantee performance on real listings. The keyword flags are only clues. Never pay a fee to get a job and verify the employer using independently located official contact details.
