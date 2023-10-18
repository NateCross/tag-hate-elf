# HateSpeechDetection

Hate Speech Detection Application using Transformer Models

### Summary

This documentation covers the development of the thesis requirement in BSCS IV. This covers the development of the AI for the use of research.

### Requirements Needed:

- Python
- Git
- Pytorch
- Transformers
- Pandas
- Excel (csv)

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

### Pytorch

## Install Pytorch:

```
pip install torch torchvision torchaudio
```

# Transformers

## Install Pytorch:

```
pip install transformers
```

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
- `pip install -e "git+https://github.com/kostyachum/python-markdown-plain-text.git#egg=plain-text-markdown-extention"`
  - Needed to convert reddit comments into plain text
