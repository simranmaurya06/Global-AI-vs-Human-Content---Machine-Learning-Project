# Global AI vs Human Content Detection

A Machine Learning project that classifies whether a given text is AI-generated or human-written.

---

## Overview

With the increasing use of AI-generated content, distinguishing between human-written and AI-generated text has become important. This project applies Natural Language Processing (NLP) and Machine Learning techniques to perform this classification.

---

## Features

* Classifies text as AI-generated or human-written
* Simple and efficient prediction pipeline
* Modular and easy-to-understand code structure
* Can be extended with advanced models

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* NLP (TF-IDF / CountVectorizer)

---

## Project Structure

```
Global-AI-vs-Human-Content/
│
├── app.py
├── model.py
├── dataset.csv
├── requirements.txt
└── README.md
```

---

## Installation and Setup

1. Clone the repository:

```
git clone https://github.com/your-username/Global-AI-vs-Human-Content.git
```

2. Navigate to the project directory:

```
cd Global-AI-vs-Human-Content
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the application:

```
python app.py
```

---

## How It Works

* Input text is preprocessed (cleaning and tokenization)
* Text is converted into numerical features using TF-IDF or CountVectorizer
* A trained machine learning model performs classification
* The system outputs whether the content is AI-generated or human-written

---

## Future Improvements

* Integrate deep learning models (LSTM or Transformers)
* Improve dataset quality and size
* Add a web interface using Streamlit
* Deploy the project for public access

---

## Author

Simran Mourya

