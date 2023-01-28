import os
import shutil

import zstandard as zstd
from download_utils import download_url

# Variables
data_dir = 'data'
temp_dir = 'temp'
base_url = 'https://files.pushshift.io/reddit/submissions'
fetch_from = (2022, 1)
fetch_to = (2022, 1)
subreddit = 'explainlikeimfive'

# Delete old files and create new directories
if os.path.exists(temp_dir):
  shutil.rmtree(temp_dir)
if os.path.exists(data_dir):
  shutil.rmtree(data_dir)
os.makedirs(temp_dir)
os.makedirs(data_dir)

# Main Logic
for year in range(fetch_from[0], fetch_to[0] + 1):
  for month in range(1, 13):
    if year == fetch_from[0] and month < fetch_from[1]:
      continue
    if year == fetch_to[0] and month > fetch_to[1]:
      break

    print(f'📅 Processing {year}-{month:02d}')
    
    # Download file
    url = f'{base_url}/RS_{year}-{month:02d}.zst'
    output_path = os.path.join(temp_dir, url.split('/')[-1])
    download_url(url, output_path, '📥 Downloading File')

    # Decompress file
    print('📦 Decompressing File')
    dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)
    with open(output_path, 'rb') as input_f:
      with open(output_path[:-4] + '.ndjson', 'wb') as output_f:
        dctx.copy_stream(input_f, output_f)

    # Extract subreddit
    print('🔎 Extracting Subreddit')
    with open(output_path[:-4] + '.ndjson', 'r') as input_f:
      with open(os.path.join(data_dir, f'{subreddit}.ndjson'), 'a') as output_f:
        for line in input_f:
          if line.find(f'"subreddit":"{subreddit}"') != -1:
            output_f.write(line)          
    
    # Delete temp files
    os.remove(output_path)
    os.remove(output_path[:-4] + '.ndjson')

    print()

print(f'🚀 Done! You can find the data in {data_dir}/{subreddit}.ndjson')
