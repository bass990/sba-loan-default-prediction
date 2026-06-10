# Hugging Face Spaces Deployment

**Author:** Mamadou Bassirou Diallo 
**App type:** Gradio  

## Deployed App Link

 
> App link: `https://huggingface.co/spaces/bass990/SBA-Loan-Default-Prediction`

## How to Deploy

1. Create a new Space at https://huggingface.co/spaces (choose **Gradio** SDK).
2. Upload the following files to the Space repository:
   - `app.py` (this directory)
   - `requirements.txt` (this directory)
   - `artifacts/xgb_model_1.json`
   - `artifacts/xgb_model_2.json`
   - `artifacts/label_encoders.pkl`
   - `artifacts/median_fill.pkl`
   - `artifacts/ensemble_metadata.json`
3. The Space will install dependencies from `requirements.txt` and launch `app.py`.

## App Interface

The app provides:
- A CSV file uploader
- A **Score** button that runs the two-model weighted ensemble
- A summary panel showing record count, default rate, weights, and threshold
- A results table with columns: `index`, `label`, `probability_0`, `probability_1`
- A **Download Results CSV** button

## Screenshots

> **TODO:** Add screenshots here after deployment.  
> 1. Screenshot of the app interface (empty state) 
![App form, empty state](screenshot_app_form.png)
> 2. Screenshot after a successful scoring run showing results table  
![App results, post-scoring](screenshot_app_results.png)

## Scoring Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `index` | int | Original record ID from input |
| `label` | int (0/1) | Predicted class using ensemble F1 threshold |
| `probability_0` | float | Ensemble probability for class 0 (non-default) |
| `probability_1` | float | Ensemble probability for class 1 (default) |
