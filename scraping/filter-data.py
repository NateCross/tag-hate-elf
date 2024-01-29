# Import necessary modules
from lingua import Language, LanguageDetectorBuilder # For language detection
import argparse # For parsing command line arguments
import pandas as pd # For handling CSV files

"""
Credit: https://stackoverflow.com/questions/761824/python-how-to-convert-markdown-formatted-text-to-text
"""


# Markdown to plain text conversion utility
from markdown import Markdown
from io import StringIO


# Function to convert markdown formatted text to plain text
def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO() # Create a new StringIO if none is provided
    if element.text:
        stream.write(element.text)  # Write the text of the current element to the stream
    for sub in element:
        unmark_element(sub, stream) # Recursively process all child elements
    if element.tail:
        stream.write(element.tail)  # Write the tail text of the current element to the stream
    return stream.getvalue()    # Return the entire content of the stream as a string


# Patching the Markdown library to add a plain text output format
Markdown.output_formats["plain"] = unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False


# Wrapper function to convert markdown text to plain text
def unmark(text):
    return __md.convert(text)

"""
Credit: https://wisecode.blog/python-string-remove-urls
"""
import re

# Function to remove URLs from a text string and replace them with '[LINK]'
def remove_urls(text):
    url_pattern = re.compile(
			r'http\S+',
			re.IGNORECASE
		)
    return url_pattern.sub('[LINK]', text)

# Function to remove Reddit usernames from a text string and replace them with '[USERNAME]'
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_filename",
        help="The csv to filter",
    ) # Command line argument for the CSV filename
    parser.add_argument(
        "tagalog_threshold",
        help="Threshold for amount of Tagalog present, float from 0.0 to 1.0",
        type=float,
        default=0.5,
    ) # Command line argument for the CSV filename
    args = parser.parse_args()

    CSV_FILENAME = args.csv_filename
# Set the threshold for considering a text as Tagalog
    TAGALOG_THRESHOLD = args.tagalog_threshold if args.tagalog_threshold else 0.50


    # Define the languages to be considered by the language detector
    languages = [Language.ENGLISH, Language.TAGALOG]


    # Initialize the language detector
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    # Read the CSV file
    csv = pd.read_csv(CSV_FILENAME, lineterminator='\n')

    filipino_phrases = 0    # Counter for Filipino phrases
    length = len(list(csv.itertuples()))    # Total number of rows in the CSV

    progress = 0    # To track progress
    for row in csv.itertuples():    # Iterate over each row in the CSV
        # Some of the data may end up being recognized as floats
        # so we should convert to str to be properly filtered
        if not isinstance(row.body, str): text = str(row.body)
        else: text = row.body

        # Clean the text by removing markdown, URLs, & usernames
        text = unmark(text)
        text = remove_urls(text)
        text = remove_usernames(text)

        # Compute the confidence of the text being Tagalog
        result = detector.compute_language_confidence(text, Language.TAGALOG)
        progress += 1
        print(f"{progress} / {length}") # Print the current progress

        # If the confidence is above the threshold, consider it a Filipino phrase
        if result >= TAGALOG_THRESHOLD:
            filipino_phrases += 1
            csv.at[row.Index, 'body'] = text    # Update the text in the CSV
        else:
            csv.drop(row.Index, inplace=True)   # Drop rows that don't meet the threshold

    print(filipino_phrases) # Print the total number of Filipino phrases found

    # Save the filtered CSV
    split_filename = CSV_FILENAME.split(".")

    csv.to_csv(
        f"{split_filename[0]}-filtered.csv"
    )
