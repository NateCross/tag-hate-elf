# Import necessary libraries
import praw # Reddit API wrapper
import pandas as pd # Data manipulation library
import argparse # Command-line parsing library
from typing import TypedDict # Type hinting for data dictionaries
from datetime import datetime   # Date & time handling

import prawcore # PRAW core exceptions

# Define a type for rows of data to ensure consistent data structure
class DataRow(TypedDict):
    id: str
    author: str
    body: str
    score: int
    subreddit: str
    timestamp: str
    submission_name: str
    submission_text: str


# Setup command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "subreddit_name",
    help="The name of the subreddit to scrape",
)   # Subreddit name argument
parser.add_argument(
    "subreddit_filter",
    help="The filter to be used on the subreddit",
    choices=[
        "top",
        "hot",
        "controversial",
    ],
) # Filter type argument (e.g., top, hot, controversial)
args = parser.parse_args()

# Set default subreddit name & limit for data collection
SUBREDDIT_NAME = args.subreddit_name if args.subreddit_name else "Philippines"
LIMIT = 1000    # Reddit api queries only return a hard maximum of 1000
                # As such, even if the limit is increased, it can
#               # never go over 1000 entries even in the main site

# Define blacklists for authors & comment bodies to exclude from the data
AUTHOR_BLACKLIST = [
    'AutoModerator',
]
BODY_BLACKLIST = [
    '[deleted]',
    '[removed]',
]

# Define options for subreddit fetching, mainly the limit
OPTIONS = {
    'limit': LIMIT,
}

# Constants for file naming
CURRENT_DATETIME = datetime.today().strftime("%Y%m%d-%H%M%S")   # Current date and time for filename
FILENAME = f'data-{SUBREDDIT_NAME}-{CURRENT_DATETIME}-{args.subreddit_filter}.csv'  # Filename format

if __name__ == "__main__":
    # Initialize PRAW Reddit instance with credentials & user agent
    reddit = praw.Reddit(
        client_id="YJvLI6W6NMduk55T10M6Qw",
        client_secret="RP9c8mfnX2OGF2t6aHQKHhvxLV4UUg",
        user_agent="linux:praw:v7.7.2 (by u/Elairion)",
        ratelimit_seconds=6000, # Give heavy allowance for rate limits to avoid TooManyRequests error
    )

    data_collection: list[DataRow] = [] # List to hold all DataRow items

    # Get subreddit instance from PRAW
    subreddit_instance = reddit.subreddit(SUBREDDIT_NAME)

    # Select the subreddit section based on the filter argument (top, controversial, hot)
    result = {
        'top': subreddit_instance.top(**OPTIONS),
        'controversial': subreddit_instance.controversial(**OPTIONS),
        'hot': subreddit_instance.hot(**OPTIONS),
    }[args.subreddit_filter]

    progress = 0    # Track the number of processed posts

    try:
        for submission in result:   # Iterate through submissions in the selected subreddit section
            progress += 1
            print(f"POST: {progress} / {LIMIT}")

            submission.comments.replace_more(limit=None)    # Load all comments by replacing "MoreComments"
            comments = submission.comments.list()   # Flatten the comment tree into a list

            comments_progress = 0   # Track the number of processed comments
            comments_num = len(comments)    # Total number of comments for the current submission

            for comment in comments:    # Iterate through each comment
                comments_progress += 1
                print(f"COMMENTS: {comments_progress} / {comments_num}")    # Log comment processing progress

                # Get author name, or set as empty string if not available
                author = (
                    comment.author.name 
                    if isinstance(comment.author, praw.models.Redditor) 
                    else ''
                )
                body = comment.body # Comment text
                word_count = len(body.split())

                # Skip comment if the author is in the blacklist
                if author in AUTHOR_BLACKLIST: continue

                # Skip comment if body is in the blacklist
                if body in BODY_BLACKLIST: continue

                # Filter out comments with two words or less
                # if word_count <= 2: continue

                data_row: DataRow = {
                    'id': comment.id,
                    'subreddit': comment.subreddit.display_name,
                    'submission_name': submission.title,
                    'submission_text': submission.selftext,
                    'author': author,
                    'body': body,
                    'score': comment.score,
                    'timestamp': datetime.utcfromtimestamp(
                        comment.created_utc
                    ).strftime('%Y-%m-%d %H:%M:%S'),
                }
                data_collection.append(data_row)    # Add the data row to the collection
    except prawcore.exceptions.TooManyRequests:
        pass    # Handle rate limit exceptions gracefully

    # Convert the list of DataRow dictionaries to a Pandas DataFrame
    data_frame = pd.DataFrame(data_collection)

    data_frame.to_csv(FILENAME) # Save the DataFrame to a CSV file

    print("Saved data") # Log completion
