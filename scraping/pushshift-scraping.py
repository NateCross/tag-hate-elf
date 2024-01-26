from pmaw import PushshiftAPI

api = PushshiftAPI()
comments = api.search_comments(
    subreddit="philippines",
    limit=1000,
)
for comment in comments:
    print(comment)
