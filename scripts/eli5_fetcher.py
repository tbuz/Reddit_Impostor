import json
import os.path
import time
from datetime import datetime, timedelta

import ndjson
import requests
from dotenv import load_dotenv
from progress.bar import Bar


class PostDeletedException(Exception):
    "Raised when the post was deleted"
    pass

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
# Create data output directory if it doesn't exist
data_dir = '../data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Ask for user input
subreddit = 'explainlikeimfive'
time_period = int(input('🕒 For what time period do you want to fetch data? (in days): '))

# Store information for subsequent requests
params = { 'subreddit': subreddit, 'size': 500, 'sort': 'created_utc' }
last_date = datetime.now()

# Create a progress bar and show it immediately
bar = Bar('🔭 Fetching posts', max=time_period)
bar.start()

reddit_api_token = ''
reddit_api_token_expires_at = 0
reddit_api_user_agent_header = { 'User-Agent': f"RecentTrends:v0.0.1 (by /u/{os.environ.get('USERNAME')})" }

def get_reddit_api_token():
    global reddit_api_token, reddit_api_token_expires_at, reddit_api_user_agent_header
    
    if reddit_api_token_expires_at > time.time():
        return reddit_api_token
    auth = requests.auth.HTTPBasicAuth(os.environ.get('CLIENT_ID'), os.environ.get('SECRET_TOKEN'))
    data = {
        'grant_type': 'password',
        'username': os.environ.get('USERNAME'),
        'password': os.environ.get('PASSWORD')
    }

    # Send request to access token endpoint
    res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=reddit_api_user_agent_header)
    
    reddit_api_token = res.json()['access_token']
    reddit_api_token_expires_at = time.time() + (res.json()['expires_in'] / 2)
    return reddit_api_token


def get_comments_for_postid(id):
    global reddit_api_user_agent_header
    answers = []

    # Add authorization to our headers dictionary
    headers = {
        **reddit_api_user_agent_header,
        'Authorization': f"Bearer {get_reddit_api_token()}"
    }
    url = f'https://oauth.reddit.com/comments/{id}?sort=top&depth=1&limit=5'

    while True:
        response = requests.get(url, headers=headers)
        # If we get rate limited, wait 30 seconds and try again
        if response.status_code != 200:
            time.sleep(30)
        else:
            break

    try:
        data = json.loads(response.text)
    except:
        print('Reddit API returned an unexpected response body')

    # Check if the post has been removed
    if data[0]['data']['children'][0]['data']['removed_by_category'] is None:
        raise PostDeletedException

    for comment in data[1]['data']['children']:
        if comment['kind'] == 'more':
            continue
        answers.append({
            'body': comment['data']['body'],
            'ups': comment['data']['ups']
        })

    return answers


def isPostValid(post):
    if 'removed_by_category' in post and post['removed_by_category'] != None:
        return False
    if post['selftext'] != '':
        return False
    if "post_hint" in post and post["post_hint"] == "image":
        return False
    return True
        

# Make requests to pushshift.io until we have all the data
while datetime.now() - last_date < timedelta(days=time_period):
    while True:
        response = requests.get('https://api.pushshift.io/reddit/search/submission', params=params)
        # If we get rate limited, wait 30 seconds and try again
        if response.status_code != 200:
            time.sleep(30)
        else:
            break
    
    try:
        data = json.loads(response.text)
    except:
        print('Parsing JSON from pushshift response failed')

    posts = data['data']
    # If there are no more posts, stop the loop
    if len(posts) == 0:
        print('No more posts found')
        break

    # Get the date of the last post
    new_date = datetime.fromtimestamp(posts[-1]['created_utc'])
    
    postsWithAnswers = []
    for post in posts:
        if not isPostValid(post):
            continue
        try:
            answers = get_comments_for_postid(post['id'])
        except PostDeletedException:
            continue
        postsWithAnswers.append({ **post, 'answers': answers })
        # Currently pushshift allows 60 requests per minute
        # Therefore, we need to wait 1 seconds between requests
        time.sleep(1)

    # Write the data to a file
    with open(f'{data_dir}/{subreddit}.ndjson', 'a') as f:
        # If we already have some data, we need to insert a newline
        if os.path.getsize(f'{data_dir}/{subreddit}.ndjson') > 0:
            f.write('\n')
        ndjson.dump(postsWithAnswers, f)

    # Update progress bar by the day difference since the last post
    bar.goto((datetime.now() - new_date).days)

    # Update the last date and the params for the next request
    last_date = datetime.fromtimestamp(posts[-1]['created_utc'])
    params['before'] = posts[-1]['created_utc']

    # Currently pushshift allows 120 requests per minute (https://api.pushshift.io/meta)
    # Therefore, we need to wait 0.5 seconds between requests
    time.sleep(0.5)

# Finish the progress bar
bar.finish()

# Print a message to the user
print(f'🚀 Done! You can find the data in data/{subreddit}.ndjson.')
