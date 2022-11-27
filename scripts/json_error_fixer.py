import json

# Ask for user input
file_name = input('👋 Hello there! Please, enter a file name: ')

data_dir = 'data'
fixed_file_name = 'fixed_' + file_name
error_count = 0

# Open the file
with open(f'{data_dir}/{file_name}', 'r') as f:
  for line in f:
    # Check if the line contains '}{"all_awardings":' (the error)
    if '}{"all_awardings":' in line:
      error_count += 1
      index = line.find('}{"all_awardings":')
      # Split the line into two parts
      line1 = line[:index+1]
      line2 = line[index+1:]
      # Write both parts to a new file
      with open(f'{data_dir}/{fixed_file_name}', 'a') as f2:
        f2.write(line1 + '\n')
        f2.write(line2)
    else:
      # Write the line to a new file
      with open(f'{data_dir}/{fixed_file_name}', 'a') as f2:
        f2.write(line)



# Print a message to the user
print(f'🚀 Done! Fixed {error_count} errors. You can find the data in data/{fixed_file_name}.')
