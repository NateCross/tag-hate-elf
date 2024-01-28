# Import necessary modules
from pmaw import PushshiftAPI   # Used to access Reddit data

# Allows us to interact with the API to fetch Reddit data & returns  a generator of comment objects that match the search criteria
api = PushshiftAPI()
comments = api.search_comments(
    subreddit="philippines",    # Specifies the subreddit where comments will be fetched
    limit=1000, # Limits the number of comments fetched
)

# Iterate over each comment objext in the comments generator & prints them
for comment in comments:
    print(comment)
