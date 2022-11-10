import json
import ndjson
import datetime
import requests
from requests.adapters import HTTPAdapter, Retry


# Ask for user input
subreddit = input('👋 Hello there! Please, enter a subreddit: r/')
file = input('Please enter the name of the output file: ')

# Make a request to the subreddit using pushshift.io
print(f'🔍 Searching for posts in r/{subreddit}...')
isFirstRequest = True
params = {'subreddit': subreddit, 'after': '24h', 'sort': 'created_utc:desc', 'size': 500}
lastTimeStamp = int(datetime.datetime.utcnow().timestamp())

s = requests.Session()
retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[ 500, 502, 503, 504 ])
with open(f'{file}.ndjson', 'a') as outline:
    while(datetime.datetime.now() - datetime.datetime.utcfromtimestamp(lastTimeStamp) < datetime.timedelta(days=7)):
        
        s.mount('https://', HTTPAdapter(max_retries=retries))
        response = s.get(f'https://api.pushshift.io/reddit/search/submission', params=params)
        
        # Convert the response to JSON
        data = json.loads(response.text)
        ndjson.dump(data['data'], outline)
        print(f'Last timestamp: {datetime.datetime.utcfromtimestamp(lastTimeStamp)}')
        lastTimeStamp = data['data'][-1]['created_utc']
        params['before'] = lastTimeStamp
        if isFirstRequest:
            params.pop('after')
            isFirstRequest = False
        
print('Done!')