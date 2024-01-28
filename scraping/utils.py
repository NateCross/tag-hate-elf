"""
Credit: https://stackoverflow.com/questions/761824/python-how-to-convert-markdown-formatted-text-to-text
"""


# Import necessary modules
from markdown import Markdown   # Handles markdown text
from io import StringIO # For in-memory text stream


# Function to recursively convert markdown elements to plain text
def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO() # Create a new in-memory text stream if none is provided
    if element.text:
        stream.write(element.text)  # Write the current element's text to the stream
    for sub in element:
        unmark_element(sub, stream) # Recursively process all child elements
    if element.tail:
        stream.write(element.tail)  # Write the tail text (text after this element) to the stream
    return stream.getvalue()    # Return the entire content of the stream as a string


# patching Markdown class to support plain text output
Markdown.output_formats["plain"] = unmark_element   # Add a new output format 'plain' to the Markdown class, using the unmark_element function
__md = Markdown(output_format="plain")  # Create a Markdown instance configured to use the 'plain' output format
__md.stripTopLevelTags = False # Configure the Markdown instance not to strip top-level tags


# Function to convert markdown text to plain text
def unmark(text):
    return __md.convert(text)
