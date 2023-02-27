import json

import requests
from prettytable import PrettyTable

# Ask for user input
subreddit = input("👋 Hello there! Please, enter a subreddit: r/")

# Make a request to the subreddit using pushshift.io
print(f"🔍 Searching for posts in r/{subreddit}...")
response = requests.get(f"https://api.pushshift.io/reddit/search/submission?subreddit={subreddit}&size=500")

# Convert the response to JSON
data = json.loads(response.text)

# Gather desired data
total_title_length = 0
total_selftext_length = 0
link_count = 0
image_count = 0
selftext_count = 0

for post in data["data"]:
  # Sum up title and selftext length
  total_title_length += len(post["title"])
  total_selftext_length += len(post["selftext"]) if "selftext" in post else 0

  # Check if the post contains a link
  if post["url"] != "":
    link_count += 1

  # Check if the post contains an image
  if "post_hint" in post and post["post_hint"] == "image":
    image_count += 1
  
  # Check if the post contains selftext
  if "selftext" in post and post["selftext"] != "":
    selftext_count += 1

# Calculate averages and percentages
data_count = len(data["data"])

average_title_length = total_title_length / data_count
average_selftext_length = total_selftext_length / data_count
link_percentage = link_count / data_count * 100
image_percentage = image_count / data_count * 100
selftext_percentage = selftext_count / data_count * 100

# Print the results
table = PrettyTable()
table.field_names = ["Metric", "Value"]
table.align["Metric"] = "l"
table.align["Value"] = "r"

table.add_row(["Average title length", f"{average_title_length:.1f} W"])
table.add_row(["Average selftext length", f"{average_selftext_length:.1f} W"])
table.add_row(["Link percentage", f"{link_percentage:.1f} %"])
table.add_row(["Image percentage", f"{image_percentage:.1f} %"])
table.add_row(["Selftext percentage", f"{selftext_percentage:.1f} %"])

print(table)

print("🚀 Done")
