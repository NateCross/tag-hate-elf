import pandas as pd
from lingua import Language, LanguageDetectorBuilder  # Placeholder for actual import if using a different library
from markdown import Markdown
from io import StringIO
import re

# Directly set the CSV filename and Tagalog threshold in the script
CSV_FILENAME = 'testing.csv'  # Replace with the actual path to your CSV file
TAGALOG_THRESHOLD = 0.5  # Set your desired threshold for Tagalog content

# Function to convert markdown formatted text to plain text
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

# Patching the Markdown library to add a plain text output format
Markdown.output_formats["plain"] = unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False

# Wrapper function to convert markdown text to plain text
def unmark(text):
    return __md.convert(text)

# Function to remove URLs from a text string and replace them with '[LINK]'
def remove_urls(text):
    url_pattern = re.compile(r'http\S+', re.IGNORECASE)
    return url_pattern.sub('[LINK]', text)

# Function to remove Reddit usernames from a text string and replace them with '[USERNAME]'
def remove_usernames(text):
    username_pattern = re.compile(r"/?u/[A-Za-z0-9_-]+", re.IGNORECASE)
    return username_pattern.sub('[USERNAME]', text)

if __name__ == "__main__":
    # Ensure the Tagalog threshold is within the valid range
    if not (0.00 <= TAGALOG_THRESHOLD <= 1.00):
        print("ERROR: Tagalog threshold must be between 0.0 and 1.0")
        exit(1)

    # Initialize the language detector with the specified languages
    languages = [Language.ENGLISH, Language.TAGALOG]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    # Read the CSV file
    try:
        csv_data = pd.read_csv(CSV_FILENAME, lineterminator='\n')
    except FileNotFoundError:
        print("ERROR: File not found")
        exit(1)

    # Drop unnecessary columns
    csv_data = csv_data.drop(columns=['id', 'subreddit', 'author', 'score', 'timestamp', 'Unnamed: 0'])

    # Add a blank 'label' column for annotation
    csv_data['label'] = ''

    filipino_phrases = 0  # Counter for Filipino phrases
    for row in csv_data.itertuples():  # Iterate over each row in the CSV
        text = str(row.body)  # Convert to string to handle non-string data

        # Clean the text by removing markdown, URLs, & usernames
        text = unmark(text)
        text = remove_urls(text)
        text = remove_usernames(text)

        # Compute the confidence of the text being Tagalog
        result = detector.compute_language_confidence(text, Language.TAGALOG)
        
        # If the confidence is above the threshold, consider it a Filipino phrase
        if result >= TAGALOG_THRESHOLD:
            filipino_phrases += 1
            csv_data.at[row.Index, 'body'] = text  # Update the text in the CSV
        else:
            csv_data.drop(row.Index, inplace=True)  # Drop rows that don't meet the threshold

    print(f"Total: {filipino_phrases}")  # Print the total number of Filipino phrases found

    # Save the filtered CSV
    filtered_filename = CSV_FILENAME.replace('.csv', '-filtered.csv')
    csv_data.to_csv(filtered_filename, index=False)  # Prevent index from being saved as a column

    print(f"Filtered data saved to {filtered_filename}")
