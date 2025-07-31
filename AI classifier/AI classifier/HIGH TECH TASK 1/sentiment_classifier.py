from textblob import TextBlob

# Initialize counters
positive_count = 0
negative_count = 0
neutral_count = 0

# File to save results
log_file = "sentiment_log.txt"

print("🔍 Simple Sentiment Classifier (Type 'exit' to quit)\n")

while True:
    user_input = input("Enter a sentence: ")

    if user_input.lower() == "exit":
        break

    # Analyze sentiment using TextBlob
    analysis = TextBlob(user_input)
    polarity = analysis.sentiment.polarity

    # Determine sentiment category
    if polarity > 0:
        sentiment = "Positive"
        positive_count += 1
    elif polarity < 0:
        sentiment = "Negative"
        negative_count += 1
    else:
        sentiment = "Neutral"
        neutral_count += 1

    # Show result
    print(f"Sentiment: {sentiment}\n")

    # Save to file
    with open(log_file, "a") as file:
        file.write(f"Input: {user_input} => Sentiment: {sentiment}\n")

# Final stats
print("\n📊 Session Summary:")
print(f"Positive: {positive_count}")
print(f"Negative: {negative_count}")
print(f"Neutral : {neutral_count}")
