# Email Phishing Detection

Group Members: Etienne Rousseau, Vishal Shenoy, Lucas Fedronic, Chadvik Maganti, Harley Salacup

Phishing attacks have become one of the most pervasive cybersecurity threats, targeting both individuals and organizations with deceptive emails. This repository contains three different approaches to detect phishing emails with Machine Learning - Random Forest, NLP, and XGBoost.

Datasets Used:
- [Numerical](https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset) - contains pre-extracted numerical features for over 500,000 emails.
- [Text](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) - contains over 80,000 emails with full body text, from a variety of sources.

Repo Organization:
- `\comparison_charts` - comparison metrics for the Random Forest, NLP, and XGBoost models.
- `\demo` - Streamlit app to run the NLP model.
- `\models` - source code, results, and charts for all model trainings and evaluations.
- `\nlp_eda` - exploratory data analysis (EDA) for the text dataset.
- `\numerical_eda` - exploratory data analysis (EDA) for the numerical dataset.
- `\numerical_eda` - exploratory data analysis (EDA) for the numerical dataset.

Running our Demo:

In order to run our demo, follow these steps:

1. Create and activate a virtual environment  
   ```bash
   cd demo
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app  
   ```bash
   streamlit run app.py
   ```
