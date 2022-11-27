import json
import os.path
import time
from datetime import datetime, timedelta

import ndjson
import requests
from progress.bar import Bar

# Create data output directory if it doesn't exist
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Ask for user input
subreddit = input('👋 Hello there! Please, enter a subreddit: r/')
time_period = int(input('🕒 For what time period do you want to fetch data? (in days): '))

# Store information for subsequent requests
params = { 'subreddit': subreddit, 'size': 500, 'sort': 'created_utc:desc' }
last_date = datetime.now()

# Create a progress bar and show it immediately
bar = Bar('🔭 Fetching posts', max=time_period)
bar.start()

# Make requests to pushshift.io until we have all the data
while datetime.now() - last_date < timedelta(days=time_period):
    # Make a request to the subreddit using pushshift.io
    # If we encounter an error (most likely rate limiting), we will simply try again after a short delay
    while True:
        response = requests.get('https://api.pushshift.io/reddit/search/submission', params=params)
        if response.status_code == 200:
            break
        else:
            time.sleep(10)

    data = json.loads(response.text)
    posts = data['data']

    # If there are no more posts, stop the loop
    if len(posts) == 0:
        break

    # Get the date of the last post
    new_date = datetime.fromtimestamp(posts[-1]['created_utc'])

    # Write the data to a file
    with open(f'{data_dir}/{subreddit}.ndjson', 'a') as f:
        ndjson.dump(posts, f)

    # Update progress bar by the day difference since the last post
    bar.goto((datetime.now() - new_date).days)

    # Update the last date and the params for the next request
    last_date = datetime.fromtimestamp(posts[-1]['created_utc'])
    params['before'] = posts[-1]['created_utc']

# Finish the progress bar
bar.finish()

# Print a message to the user
print(f'🚀 Done! You can find the data in data/{subreddit}.ndjson.')
