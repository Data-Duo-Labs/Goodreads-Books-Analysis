# 📚 Goodreads Market & Reader Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link-goes-here.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

An end-to-end data analysis and interactive dashboard exploring decades of literature to uncover what truly makes a book successful.

---

# 🎯 Project Overview

This collaborative project analyzes a massive dataset of Goodreads books to identify macro market trends, author performance, and reader preferences.
[[View StreamLit Dashboard](https://the-data-library.streamlit.app/)]

We engineered a custom **Popularity Score** that weights:

- Total ratings
- Best Book Ever votes
- Reader satisfaction

This allows books to be evaluated **beyond just their star rating**.

The final **interactive web dashboard built in Python using Streamlit**, designed to visualize the anatomy of a successful book.


---

# 🛠️ Tech Stack & Tools

**Language**
- Python

**Data Processing**
- Pandas  
- NumPy  

**Visualization**
- Matplotlib  
- Seaborn  

**Modeling / Scaling**
- Scikit-Learn (`MinMaxScaler`)

**Web Deployment**
- Streamlit

---

# 📊 Key Insights & Features

### Custom Metric Engineering
Created a normalized **`popularity_score`** balancing:

- Volume → total ratings
- Sentiment → liked percentage

---

### Franchise Deep Dives
Analyzed major series such as:

- Harry Potter
- The Hunger Games
- Other high-impact franchises

This allowed us to map **reader retention and franchise momentum over time**.

---

### Market Trends
Visualized a **70-year timeline (1950–2020)** showing the relationship between:

- Average ratings
- Market popularity
- Publishing volume

---

### The Anatomy of a Hit
Explored how book attributes influence reach:

- Page count
- Character volumes
- Literary awards
- Release seasonality

---

# 📂 Project Workflow & Documentation

We treated this project like a **structured data science sprint**, documenting each stage from ideation to deployment.

### 1️⃣ Ideation & Architecture
[[View Brainstorming & Workflow Notes (Google Docs)](https://docs.google.com/document/d/1FtraKUcTtWhhFXORc182eBlOinrglJ-6PxMv2xCjwF0/edit?usp=sharing)]

### 2️⃣ Data Cleaning & EDA
[[View EDA Tracker (Google Sheets)](https://docs.google.com/spreadsheets/d/1WmvuV2ILCUgYlSW53Rrdp1YgU4USCW46A0crQPzxyns/edit?usp=sharing)]

### 3️⃣ Analysis Notebooks

We split the exploratory analysis to explore different perspectives of the market.

**Mahak Vishwakarma**  
- `mahak_cleaned.ipynb`
- `mahak_analysis.ipynb`  
Focus: **Decades, Series, Authors, Publishers**

**Aaditya Malviya**
- `aaditya_cleaned.ipynb`
- `aaditya_analysis.ipynb`  
Focus: **Seasonality, Awards, Book Length, Characters**

---

# 📁 Repository Structure

```text
Goodreads-Market-Analysis/
│
├── data/
│   ├── books_1.Best_Books_Ever.csv   # Raw dataset
│   └── final_merge.csv               # Cleaned dataset used for the app
│
├── notebooks/
│   ├── 01_aaditya_cleaned.ipynb        # Data cleaning (Part 1)
│   ├── 02_mahak_cleaned.ipynb      # Data cleaning (Part 2)
│   ├── 03_aaditya_analysis.ipynb       # Awards & attributes analysis
│   └── 04_mahak_analysis.ipynb     # Series & market trends analysis
│
├── app.py                            # Streamlit dashboard source code
├── requirements.txt                  # Deployment dependencies
└── README.md                         # Project documentation
```

---

# 🚀 How to Run Locally

### Clone the repository

```bash
git clone https://github.com/mahakVishwa/Goodreads-Books-Analysis.git
cd Goodreads-Books-Analysis
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

---

# 👨‍💻 Contributors

Built by:

- **Mahak Vishwakarma** — ([GitHub Profile](https://github.com/mahakVishwa))
- **Aaditya Malviya** — [GitHub Profile](Aaditya's GitHub Link)
