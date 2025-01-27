# HateSpeechDetection

Hate Speech Detection Application using Transformer Models

Enhancing Hate Speech Detection in Tagalog and Taglish Text: An Ensemble Learning Framework Using Bernoulli Naive Bayes, LSTM, and mBERT
### Summary

This documentation covers the development of the thesis requirement in BSCS IV. This covers the development of the AI for the use of research.

### Requirements Needed:

- Python
- Git
- Pytorch
- Transformers
- Pandas
- Excel (csv)
- Rust

### References:

[Python](https://www.python.org/doc/)

[Pytorch](https://pytorch.org/docs/stable/index.html)

[Python Pandas](https://pandas.pydata.org/docs/)

[Transformers - Huggingface](https://huggingface.co/docs/transformers/index)

# Setting Up

---

## 1. Install Dependencies

### Python

## For Mac (via homebrew):

```
brew install python3
```

## For Windows:

Download Python through their official Website - [Python](https://www.python.org/downloads/)
### Rust

[Download Rust here](https://www.rust-lang.org/tools/install)

## 2. Set-up the environment

```
python -m venv <venv>
```

## 3. Activate the environment

## For Windows:

```
.\venv\Scripts\activate
```

## For Mac:

```
source venv/bin/activate

```

## 4. Install dependencies

`pip install -r requirements.txt`

---

# Run the Program

```
python mbert_test.py
```

## Deactivate

After running the app run this code

```
deactivate
```

## Other packages to install

- `scikit-learn`
- `skorch`
- `pip install -e "git+https://github.com/kostyachum/python-markdown-plain-text.git#egg=plain-text-markdown-extention"`
  - Needed to convert reddit comments into plain text

## Credits

[Tagalog Stopwords](https://github.com/stopwords-iso/stopwords-tl/blob/master/stopwords-tl.txt)
