import praw
import pandas as pd
import argparse
from typing import TypedDict
from datetime import datetime

import prawcore

class DataRow(TypedDict):
    id: str
    author: str
    body: str
    score: int
    subreddit: str
    timestamp: str
    submission_name: str
    submission_text: str


parser = argparse.ArgumentParser()
parser.add_argument("subreddit_name")
parser.add_argument("subreddit_filter")
args = parser.parse_args()

# Options
SUBREDDIT_NAME = args.subreddit_name if args.subreddit_name else "CasualPH"
LIMIT = 1000
AUTHOR_BLACKLIST = [
    'AutoModerator',
]
BODY_BLACKLIST = [
    '[deleted]',
    '[removed]',
]

OPTIONS = {
    'limit': LIMIT,
}

# Constants
CURRENT_DATETIME = datetime.today().strftime("%Y%m%d-%H%M%S")
FILENAME = f'data-{SUBREDDIT_NAME}-{CURRENT_DATETIME}-{args.subreddit_filter}.csv'

if __name__ == "__main__":
    reddit = praw.Reddit(
        client_id="YJvLI6W6NMduk55T10M6Qw",
        client_secret="RP9c8mfnX2OGF2t6aHQKHhvxLV4UUg",
        user_agent="linux:praw:v7.7.2 (by u/Elairion)",
        ratelimit_seconds=6000, # Give heavy allowance for rate limits
    )

    data_collection: list[DataRow] = []

    subreddit_instance = reddit.subreddit(SUBREDDIT_NAME)

    result = {
        'top': subreddit_instance.top(**OPTIONS),
        'controversial': subreddit_instance.controversial(**OPTIONS),
        'hot': subreddit_instance.hot(**OPTIONS),
    }[args.subreddit_filter]

    progress = 0

    try:
        for submission in result:
            progress += 1
            print(f"POST: {progress} / {LIMIT}")

            submission.comments.replace_more(limit=None)
            comments = submission.comments.list()

            comments_progress = 0
            comments_num = len(comments)

            for comment in comments:
                comments_progress += 1
                print(f"COMMENTS: {comments_progress} / {comments_num}")

                author = (
                    comment.author.name 
                    if isinstance(comment.author, praw.models.Redditor) 
                    else ''
                )
                body = comment.body
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
                data_collection.append(data_row)
    except prawcore.exceptions.TooManyRequests:
        pass

    data_frame = pd.DataFrame(data_collection)

    data_frame.to_csv(FILENAME)

    print("Saved data")
