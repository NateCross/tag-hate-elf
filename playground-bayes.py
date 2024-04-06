from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The sky is blue and the sun is shining.",
    "Cats are wonderful pets to have around the house.",
    "Soccer is a popular sport played all around the world.",
    "Pizza topped with cheese and pepperoni is my favorite.",
    "Learning a new language can be challenging but rewarding.",
    "Reading books is a great way to expand your knowledge.",
    "The internet has revolutionized the way we communicate.",
    "Music has the power to evoke strong emotions in people.",
    "Exercise is important for maintaining good health.",
    "Science has made incredible advancements in recent years.",
    "Coffee is the fuel that keeps many people going throughout the day.",
    "Traveling allows you to experience different cultures and cuisines.",
    "The importance of education cannot be overstated.",
    "Technology continues to progress at a rapid pace.",
    "Dogs are known for their loyalty and companionship.",
    "Cooking homemade meals can be a fun and rewarding activity.",
    "Nature has a way of calming the mind and soothing the soul.",
    "Artistic expression comes in many forms, from painting to music.",
    "Happiness is often found in the simplest of things."
]

tfidf = TfidfVectorizer()

document_matrix = tfidf.fit_transform(texts)

# print(document_matrix)
