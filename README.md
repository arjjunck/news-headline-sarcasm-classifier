# Sarcasm Detection using Custom Naive Bayes Models

## Project Overview

This project builds a **sarcasm detection system** using custom implementations of Naive Bayes models in Python.

The aim of the project is to classify news headlines as either **sarcastic** or **not sarcastic** using text preprocessing, manual feature representation, Naive Bayes variants, numeric meta-features, class imbalance handling and error analysis.

Unlike using a ready-made Scikit-learn classifier, this project implements much of the logic manually to demonstrate a deeper understanding of how Naive Bayes text classification works internally.

## Dataset

**Dataset file used:** `Sarcasm_Headlines_Dataset_v2.json`

The dataset contains news headlines with binary sarcasm labels.

### Dataset Columns

| Column | Description |
|---|---|
| `headline` | News headline text |
| `is_sarcastic` | Target label, where 1 = sarcastic and 0 = not sarcastic |
| `article_link` | URL of the source article |

### Dataset Summary

| Item | Value |
|---|---:|
| Total rows | 28,619 |
| Training set | 20,033 |
| Validation set | 4,293 |
| Test set | 4,293 |
| Split ratio | 70 / 15 / 15 |

The dataset was shuffled using a seed derived from the student ID to ensure reproducibility.

## Problem Type

This is a **binary NLP classification problem**.

The two classes are:

| Label | Meaning |
|---|---|
| `0` | Not sarcastic |
| `1` | Sarcastic |

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Regular Expressions
- Scikit-learn metrics

## Techniques Applied

- Text cleaning
- Tokenisation
- Negation handling
- Stopword removal
- Unigram and bigram generation
- Out-of-vocabulary handling
- Rare word filtering
- Manual count matrix creation
- TF-IDF calculation
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Gaussian Naive Bayes
- Numeric meta-feature extraction
- Missing value injection and imputation
- Outlier detection using IQR
- Outlier clamping
- Scaling comparison
- Random oversampling
- Threshold tuning
- Confusion matrix analysis
- Error analysis

## Project Workflow

```text
Load sarcasm headline dataset
        |
        v
Shuffle dataset using reproducible seed
        |
        v
Create 70/15/15 train-validation-test split
        |
        v
Preprocess text
- Lowercase conversion
- HTML removal
- Regex cleaning
- Tokenisation
- Negation handling
- Stopword removal
- N-gram generation
        |
        v
Build vocabulary
        |
        v
Apply rare word filtering
        |
        v
Create manual feature matrices
- Count matrix
- TF-IDF matrix
        |
        v
Train Naive Bayes variants
- Multinomial NB
- Bernoulli NB
- Gaussian NB
        |
        v
Add numeric meta-features
        |
        v
Handle missing values and outliers
        |
        v
Study scaling effects
        |
        v
Handle class imbalance
        |
        v
Tune decision threshold
        |
        v
Evaluate model performance
        |
        v
Analyse misclassified headlines
```

## Text Preprocessing

The preprocessing pipeline includes:

1. Convert text to lowercase
2. Remove HTML tags
3. Replace repeated punctuation with special tokens
4. Tokenise the text using regular expressions
5. Apply negation handling
6. Remove stopwords
7. Generate unigrams and bigrams
8. Replace out-of-vocabulary tokens with `<UNK>`

### Example

```text
Input:
I do not like this movie

Output:
['do', 'not', 'NOT_like', 'movie', 'do__not', 'not__NOT_like', 'NOT_like__movie']
```

## Special Token Handling

The preprocessing stage converts punctuation patterns into meaningful tokens:

| Pattern | Token |
|---|---|
| Multiple exclamation marks | `[EXC_SEQ]` |
| Multiple question marks | `[QUE_SEQ]` |
| Ellipsis | `[ELLIP]` |
| Mixed question/exclamation patterns | `[INT_MARK]` |

This helps preserve emotional or sarcastic signals that may be useful for classification.

## Feature Representation

### Count Matrix

A custom count matrix was created manually instead of using `CountVectorizer`.

The project used rare word filtering with the threshold:

```text
threshold = 6
```

The final count matrix shapes were:

| Dataset | Shape |
|---|---|
| Training matrix | 20,033 × 791 |
| Validation matrix | 4,293 × 791 |
| Test matrix | 4,293 × 791 |

### TF-IDF

TF-IDF was also implemented manually using:

```text
TF = token count in document / total tokens in document
IDF = log((N + 1) / (document frequency + 1)) + 1
```

## Top Tokens by Document Frequency

The top tokens in the training data included:

| Token | Document Frequency |
|---|---:|
| with | 1370 |
| new | 1148 |
| trump | 1021 |
| man | 955 |
| about | 766 |
| you | 701 |
| after | 700 |
| up | 651 |
| out | 630 |
| it | 585 |

## Models Implemented

### 1. Multinomial Naive Bayes

Multinomial Naive Bayes was implemented manually using token count features and smoothing.

Validation was performed using different smoothing values:

```text
Selected alphas: 0, 0.5, 2
```

### Multinomial NB Test Result

| Metric | Score |
|---|---:|
| Accuracy | 0.6152 |
| F1 Macro | 0.5956 |
| F1 Sarcasm | 0.5066 |

### 2. Bernoulli Naive Bayes

Bernoulli Naive Bayes was implemented using binary word presence features.

### Bernoulli NB Test Result

| Metric | Score |
|---|---:|
| Accuracy | 0.6157 |
| F1 Macro | 0.5969 |
| F1 Sarcasm | 0.5101 |

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes was applied to numeric meta-features extracted from the headlines.

The numeric features included:

- Word count
- Character length
- Uppercase word count
- Exclamation mark count
- Question mark count
- Negation count
- Digit count
- Sarcasm ratio

## Missing Value Handling

Artificial missing values were injected into the numeric feature dataset to study imputation methods.

Two imputation methods were compared:

1. Global mean imputation
2. Class-conditional mean imputation

### Imputation Results

| Method | Accuracy | F1 Macro | F1 Sarcasm |
|---|---:|---:|---:|
| Global Mean Imputation | 0.5577 | 0.5178 | 0.3792 |
| Class Mean Imputation | 0.5595 | 0.5255 | 0.3983 |

Class mean imputation performed slightly better.

## Outlier Detection and Treatment

Outliers were detected using the IQR method.

```text
Outliers detected: 9,030 out of 20,033 training samples
Outlier percentage: 45.1%
```

Instead of removing outliers, values were clamped to the IQR boundaries. This preserved the dataset size and class distribution.

### Outlier Treatment Results

| Method | Accuracy | F1 Macro | F1 Sarcasm |
|---|---:|---:|---:|
| With outliers | 0.5115 | 0.3881 | 0.6629 |
| IQR clamping | 0.5693 | 0.5409 | 0.4267 |

IQR clamping improved macro F1-score and overall balance.

## Scaling Study

Three scaling approaches were compared on numeric features:

1. No scaling
2. MinMax scaling
3. Standardisation

### Scaling Results

| Method | Accuracy | F1 Macro | F1 Sarcasm |
|---|---:|---:|---:|
| No scaling | 0.5693 | 0.5409 | 0.4267 |
| MinMax | 0.5155 | 0.3401 | 0.0000 |
| Standardised | 0.5693 | 0.5409 | 0.4267 |

No scaling and standardisation gave the same results in this experiment, while MinMax scaling performed poorly.

## Class Imbalance Handling

The training set class distribution was:

| Class | Count |
|---|---:|
| Not sarcastic | 10,558 |
| Sarcastic | 9,475 |

The imbalance ratio was:

```text
1.114
```

Random oversampling was applied to balance the sarcastic class.

After oversampling:

| Class | Count |
|---|---:|
| Not sarcastic | 10,558 |
| Sarcastic | 10,558 |

## Threshold Tuning

Threshold tuning was performed on the validation set.

```text
Best threshold: 0.50
Best validation F1 macro: 0.5519
```

### Validation Confusion Matrix

```text
[[1655,  559],
 [1287,  792]]
```

### Validation Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.5700 |
| Precision Macro | 0.5744 |
| Recall Macro | 0.5642 |
| F1 Macro | 0.5519 |
| F1 Minority/Sarcasm | 0.4618 |

## Error Analysis

The Multinomial Naive Bayes model misclassified:

```text
1,664 out of 4,293 test samples
Error rate: 38.8%
```

The project manually examined misclassified headlines to understand model weaknesses.

Some errors occurred because sarcastic headlines can be subtle and context-dependent, while some serious headlines contain words or structures that resemble satire.

## Custom Headline Testing

The model was tested on custom headlines.

| Headline | Expected | Predicted |
|---|---|---|
| local farmer breeding pit-bull sheep to ward off dog attacks | Sarcastic | Serious |
| government announces new funding for rural healthcare clinics | Serious | Serious |
| gp can see you in four months at the earliest | Mixed | Serious |

This shows that the model handles simple serious headlines better than subtle or context-heavy sarcasm.

## Key Findings

- Custom Naive Bayes models can classify sarcasm with moderate performance.
- Bernoulli Naive Bayes performed slightly better than Multinomial Naive Bayes in this run.
- Negation handling, punctuation tokens and bigrams helped preserve useful linguistic patterns.
- Class mean imputation performed better than global mean imputation.
- IQR clamping improved macro F1-score for numeric feature models.
- Random oversampling balanced the class distribution.
- Sarcasm detection remains difficult because sarcastic headlines often require real-world context.

## Limitations

- The model uses bag-of-words style features and does not understand deep context.
- Word order is only partially captured through bigrams.
- Sarcasm is often context-dependent and may require external knowledge.
- The model does not use modern embeddings such as Word2Vec, GloVe or BERT.
- The custom feature mapping approach is useful for learning but less practical than standard vectorisers.
- Some manually engineered numeric features are weak indicators of sarcasm.
- The dataset comes from news headlines and may not generalise to social media sarcasm.

## Future Improvements

- Compare with Scikit-learn `MultinomialNB` and `BernoulliNB`.
- Add TF-IDF with Logistic Regression.
- Add SVM for text classification.
- Use word embeddings.
- Fine-tune a transformer model such as BERT.
- Add more advanced preprocessing, including lemmatisation.
- Improve sarcasm-specific features.
- Build a Streamlit app for live headline testing.
- Save predictions and error examples to CSV.
- Add model explainability for top sarcastic and non-sarcastic tokens.

## Suggested Repository Structure

```text
sarcasm-detection-naive-bayes/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── sarcasm_detection_naive_bayes.ipynb
│
├── data/
│   └── Sarcasm_Headlines_Dataset_v2.json
│
└── images/
    ├── class_distribution.png
    └── confusion_matrix.png
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sarcasm-detection-naive-bayes.git
cd sarcasm-detection-naive-bayes
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 3. Add the dataset

Place the dataset file inside the project folder or inside the `data/` folder:

```text
data/Sarcasm_Headlines_Dataset_v2.json
```

### 4. Update the dataset path

Inside the notebook, update the dataset path if needed:

```python
df = pd.read_json("data/Sarcasm_Headlines_Dataset_v2.json", lines=True)
```

### 5. Run the notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/sarcasm_detection_naive_bayes.ipynb
```

## Requirements

Create a `requirements.txt` file with:

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

## Files to Remove Before Publishing

Remove these files if they appear:

```text
.DS_Store
.ipynb_checkpoints/
__pycache__/
```

Use this `.gitignore`:

```text
.DS_Store
.ipynb_checkpoints/
__pycache__/
*.pyc
.env
```

## Project Status

Completed as a custom NLP classification project using Naive Bayes variants and manual feature engineering.

## Author

**Arjun Krishna Krishnakumar**

Aspiring AI/ML and software developer with an interest in machine learning, natural language processing, text classification and practical AI applications.

## License

This project is for educational and portfolio purposes.
