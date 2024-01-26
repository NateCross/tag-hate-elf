from lingua import Language, LanguageDetectorBuilder
import argparse
import pandas as pd

"""
Credit: https://stackoverflow.com/questions/761824/python-how-to-convert-markdown-formatted-text-to-text
"""
from markdown import Markdown
from io import StringIO


def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


# patching Markdown
Markdown.output_formats["plain"] = unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False


def unmark(text):
    return __md.convert(text)

"""
Credit: https://wisecode.blog/python-string-remove-urls
"""
import re

def remove_urls(text):
    url_pattern = re.compile(
			r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})(\S*)\/?',
			re.IGNORECASE
		)
    return url_pattern.sub('[LINK]', text)

def remove_urls_v2(text):
    url_pattern = re.compile(
			r'http\S+',
			re.IGNORECASE
		)
    return url_pattern.sub('[LINK]', text)

def remove_usernames(text):
    username_pattern = re.compile(
        r"/?u/[A-Za-z0-9_-]+",
        re.IGNORECASE,
    )
    return username_pattern.sub('[USERNAME]', text)

"""
This script is meant to filter out non-Tagalog or Taglish text in a csv
made from running `reddit-scrape.py`
"""

# OPTIONS
TAGALOG_THRESHOLD = 0.75

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_filename")
    args = parser.parse_args()

    csv_filename = args.csv_filename

    languages = [Language.ENGLISH, Language.TAGALOG]

    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    csv = pd.read_csv(csv_filename, lineterminator='\n')

    filipino_phrases = 0
    length = len(list(csv.itertuples()))

    progress = 0
    for row in csv.itertuples():
        # Some of the data may end up being recognized as floats
        # so we should convert to str to be properly filtered
        if not isinstance(row.body, str): text = str(row.body)
        else: text = row.body

        text = unmark(text)
        text = remove_urls_v2(text)
        text = remove_usernames(text)

        result = detector.compute_language_confidence(text, Language.TAGALOG)
        progress += 1
        print(f"{progress} / {length}")

        if result >= TAGALOG_THRESHOLD:
            filipino_phrases += 1
            csv.at[row.Index, 'body'] = text
        else:
            csv.drop(row.Index, inplace=True)

    print(filipino_phrases)

    split_filename = csv_filename.split(".")

    csv.to_csv(
        f"{split_filename[0]}-filtered.csv"
    )
