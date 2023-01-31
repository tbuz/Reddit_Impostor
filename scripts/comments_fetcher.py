import json
import os.path
import time

import requests
from dotenv import load_dotenv
from progress.bar import Bar

# Variables
data_dir = 'data'
subreddit = 'explainlikeimfive'

# Define input, output
input_file = os.path.join(data_dir, f'{subreddit}.ndjson')
if not os.path.exists(input_file):
  print(f'Input file {input_file} does not exist.')
  exit(1)

output_file = os.path.join(data_dir, f'{subreddit}_comments.ndjson')
if os.path.exists(output_file):
  os.remove(output_file)

# Load env
dotenv_file = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_file)

# For Authentication
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
    return []

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

# Main Logic
num_input_lines = sum(1 for _ in open(input_file))

bar = Bar('🔭 Fetching comments', max=num_input_lines)
bar.start()

with open(input_file) as f:
  for line in f:
    post = json.loads(line)

    if not isPostValid(post):
      bar.next()
      continue
    
    answers = get_comments_for_postid(post['id'])
    if not answers:
      bar.next()
      continue

    post_with_answers = { **post, 'answers': answers }
    
    # Write to output file
    with open(output_file, 'a') as f:
      f.write(json.dumps(post_with_answers) + '\n')

    # Show progress
    bar.next()

    # Currently Reddit API allows 60 requests per minute
    # Therefore, we need to wait 1 seconds between requests
    time.sleep(1)

# Finish the progress bar
bar.finish()

# Print a message to the user
print(f'🚀 Done! You can find the data in {output_file}')
