
# 📰 Fake News Detection Using Apache Spark MLlib

This project builds a machine learning pipeline using **Apache Spark MLlib** to classify news articles as **FAKE** or **REAL** based on their content. It walks through each core stage: loading, cleaning, feature extraction, modeling, and evaluation — all in individual, modular Python scripts.

---

## 💻 Requirements

- Python 3.8+
- PySpark
- Pandas
- Faker (for data generation)

---

## 📁 Dataset Used

- **Filename:** `fake_news_sample.csv`
- **Records:** 500 articles (250 FAKE, 250 REAL)
- **Columns:**
  - `id`: Unique article ID
  - `title`: Headline
  - `text`: Full article text
  - `label`: FAKE or REAL

🛠️ **To generate this file**, run `data_generator.py` after installing `Faker`:

```bash
pip install faker
python data_generator.py
```

---

## 🧠 Task Descriptions

---

### 🧪 Task 1: Load & Explore the Dataset
**What it does:**
- Loads the CSV with Spark.
- Displays sample rows.
- Counts total articles.
- Extracts distinct labels.
- Saves results to `task1_output.csv`.

**Sample Output (`task1_output.csv`):**

| id   | title                            | text                                                                 | label |
|------|----------------------------------|----------------------------------------------------------------------|--------|
| R129 | New Report on Election           | Action whatever air commercial take same. Question mind any able rule by Effort level hope not trouble fast. Experts discuss the ongoing developments in election. | REAL   |
| R227 | New Report on Climate Change     | Case professional nearly service fact call box seek. Ready environmental join memory black draw. Experts discuss the ongoing developments in climate change. | REAL   |


---

### 🧹 Task 2: Text Preprocessing
**What it does:**
- Lowercases all article text.
- Tokenizes the text into words.
- Removes stopwords.
- Saves tokenized output to `task2_output.csv`.

**Sample Output (`task2_output.csv`):**

| id   | title                            | filtered_words                                         | label |
|------|----------------------------------|--------------------------------------------------------|--------|
| R129 | New Report on Election       | ['action', 'whatever', 'air', 'commercial', 'take', 'same.', 'question', 'mind', 'able', 'rule', 'by.', 'effort', 'level', 'hope', 'trouble', 'fast.', 'experts', 'discuss', 'ongoing', 'developments', 'election.'] | REAL   |
| R227 | New Report on Climate Change | ['case', 'professional', 'nearly', 'service', 'fact', 'call', 'box', 'seek.', 'ready', 'environmental', 'join', 'memory', 'black', 'draw.', 'experts', 'discuss', 'ongoing', 'developments', 'climate', 'change.']     | REAL   |

---

### 🧮 Task 3: Feature Extraction
**What it does:**
- Converts words into numerical TF vectors (`HashingTF`).
- Applies `IDF` weighting.
- Encodes labels into numeric format.
- Saves output to `task3_output.csv`.

**Sample Output (`task3_output.csv`):**

| id   | filtered_words_array                                 | features                   | label_index |
|------|------------------------------------------------------|----------------------------|--------------|
| R129 | ['action', 'whatever', 'air', 'commercial', 'take', 'same.', ...] | (10000,[88,343,474,...],[4.13,4.13,5.52,...]) | 1.0 |
| R227 | ['case', 'professional', 'nearly', 'service', 'fact', ...] | (10000,[821,1948,2059,...],[2.84,4.83,2.72,...]) | 1.0 |


---

### 🤖 Task 4: Model Training & Prediction
**What it does:**
- Splits dataset into train/test sets.
- Trains `LogisticRegression` model.
- Predicts on test set.
- Saves predictions to `task4_output.csv`.

**Sample Output (`task4_output.csv`):**

| id   | title                            | label_index | prediction |
|------|----------------------------------|-------------|------------|
| F002 | Mind Control Confirmed by Sources | 0.0              | 0.0         |
| F006 | Hoax Confirmed by Sources        | 0.0              | 0.0         |

---

### 📊 Task 5: Model Evaluation
**What it does:**
- Evaluates model using Accuracy and F1 Score.
- Saves results to `task5_output.csv`.

**Sample Output (`task5_output.csv`):**

| Metric   | Value |
|----------|-------|
| Accuracy | 1.0   |
| F1 Score | 1.0   |

---

## ▶️ How to Run the Code

### 📦 Install Required Libraries
```bash
pip install pyspark pandas faker
```

---

### 📄 Create the Python Files
```bash
touch task1.py
touch task2.py
touch task3.py
touch task4.py
touch task5.py
```

---

### 🚀 Run the Files in Order

```bash
python data_generator.py   # Generate dataset using Faker
python task1.py            # Task 1: Load and explore dataset
python task2.py            # Task 2: Clean and tokenize text
python task3.py            # Task 3: TF-IDF feature extraction
python task4.py            # Task 4: Train and predict with model
python task5.py            # Task 5: Evaluate accuracy and F1 score
```

Each task will generate its own `taskX_output.csv` file.

---

## ✅ Final Results

| 📈 Metric   | 🔢 Value |
|------------|----------|
| Accuracy   | 1.00     |
| F1 Score   | 1.00     |

These high scores are expected on this synthetic, clearly separable dataset.

---





