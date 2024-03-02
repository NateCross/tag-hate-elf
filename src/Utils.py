from markdown import Markdown
from io import StringIO
import re
import emoji

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

# Function to remove emojis using the emoji python library
def remove_emojis(text):
    return emoji.replace_emoji(text, '')

# Replace escape sequences with a space
def remove_escape_sequences(text):
    escape_pattern = re.compile(r'[\r\n\t]', re.IGNORECASE)
    return escape_pattern.sub(' ', text)
