from lingua import Language, LanguageDetectorBuilder
import pandas as pd

languages = [Language.ENGLISH, Language.TAGALOG]

detector = LanguageDetectorBuilder.from_languages(*languages).build()

csv = pd.read_csv("./datasets/train.csv")

filipino_phrases = 0

for row in csv.itertuples():
    result = detector.compute_language_confidence(row.text, Language.TAGALOG)
    if result >= 0.70:
        filipino_phrases += 1
        print(row.text)

print(filipino_phrases)

