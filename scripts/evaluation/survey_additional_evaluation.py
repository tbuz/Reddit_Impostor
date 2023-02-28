import pandas as pd
import numpy as np
import csv

# GROUP A
dfA = pd.read_csv('surveyA.csv')

# GROUP B 
dfB = pd.read_csv('surveyB.csv')

realSurveyA = [
"Most drivers of the Honda Fit are in fact not fit",
"Stoplights seem to work more efficiently when they are malfunctioning and drivers are temporarily treating them like stop signs",
"Chocolate bars are really bean bars and coffee is really bean juice",
"Sword is an S word",
"Influencers are treated like celebrities but are really salespeople on commission",
"Writing something down is not always about reading it again later to remember, but using a different facet of your brain to also record that memory and make it easier to retrieve when you need it",
"Most people know how long they can hold their breath, few people know how long they can hold their feelings",
"Eventhough it's so close to our eyes, we'll never be able to physically see our brain",
"When you stare at somebody's reflection, your reflection is staring at them",
"Brushing teeth is actually just polishing your bones",
"The brain is an organ that can complain about itself",
"The real gauge of a relationship is whether you annoy each other by how you eat",
"Driving a car is actually some Wile E. Coyote type madness. We’re riding a series of explosions down a freeway at nearly 100 mile an hour",
"Spoons can be used as catapults, but catapults can’t be used as spoons",
"You are always looking at your nose",
]
realSurveyB = [
"Napoleon never rode a bicycle",
"Technically life is a death sentence",
"Construction workers get a free pass regarding carrying a knife in public",
"There's a perception that you can't just invent a word, and yet hundreds of thousands of them have been invented",
"A photo is the ultimate #ShortVideo",
"All network and digital database is a mini matrix constructed by humans",
"One second is technically a fraction of a second",
"One unforeseen aspect of becoming a parent is that suddenly you are open to a whole new genre of nightmares that involve losing your kid",
"Gravity is a dog’s best friend in the kitchen and dining table",
"A simple flip of a sandwich can alter your entire perception/experience of that sandwich when eaten",
"Some types of colourblind people may never learn that red grapes look more purple than red",
"Accidentally washing a pair of pants with a $5 bill in one of the pockets is money laundering",
"Mickey Mouse is doing its best to save the reputation of the mice",
"Parents used to tell kids not to believe in anything the TV said and nowadays, parents believe in everything social media says",
"Every person you will ever meet knows something you don’t",
]


neoA = [
"We could be the last generation to die before immortality is invented",
"You’d have to go in the right direction to get into the right direction",
"There's probably an unbroken line between cavemen and computers",
"If your parents didn't have kids, you won't either",
"If your parachute fails, you have the rest of your life to fix it",
"The world will be a very sad place if there wasn't any hope",
"We’re all dying. Just some faster than others",
"Your tongue rests on the roof of your mouth",
"People who make ads are literally paid actors",
"You can only see as far as your eyes will take you",
]

neoB = [
"We are living in interesting times",
"Your stomach thinks all potato is mashed",
"If there are infinite parallel universes, then there is one where no parallel universes exist",
"Most people aren't afraid of death, they're afraid of what happens after it",
"There is no way to know for sure that everything we believe is not simply a product of our limited perspective",
"When your alarm goes off, it actually goes on",
"When somebody says \"I'm down for anything\", it's usually followed by an overly enthusiastic 'up'",
"Socks aren’t underwear",
"We are not our bodies, we are just the host that our bodies use to sustain themselves",
"The number of people older than you never increases",
]

chatA = [
"If time flies when we're having fun, does that mean it's just slow and boring when we're doing chores?",
"The internet never forgets, yet our memories are surprisingly selective",
"If we can make plans to meet up with someone tomorrow, why can't we make plans to dream about them tonight?",
"Why do we call it a \"keyboard\" when it's really a \"typing board\"?",
"If you think about it, our fingerprints are just nature's way of making sure we always leave our mark",
"If we can put a man on the moon, why can't we find a solution to the snooze button?",
"Life is like a camera, just focus on the good times, develop from the negatives, and if things don't work out, take another shot",
"If we can train dogs to do tricks, why can't we train our minds to do the same?",
"Why do we park in a driveway and drive on a parkway?",
"If we call our significant other \"my better half,\" does that mean they complete us or just make up 50% of us?",
]

chatB = [
"Why do we call it \"taking a dump\" when we're actually leaving one behind?",
"Why do we have to turn off the lights to go to sleep when our eyes are already closed?",
"If we can have a \"piece of mind,\" does that mean we can have a \"slice of happiness\" or a \"serving of joy\"?",
"Why do we say \"heads up\" when we really mean \"heads down\"?",
"If time is money, does that mean we're all just broke?",
"Why do we call it \"fast food\" when it's actually just slow poison?",
"If we're all just stars in the making, why do we act like we're already burnt out?",
"Why do we say \"I slept like a baby\" when babies are notorious for waking up every few hours?",
"If love is a battlefield, does that mean our hearts are the casualties?",
"If we can have a \"clean slate,\" does that mean our brains are really just chalkboards?",
]

gpt2A = [
"Cats are the only animals who are good at hiding",
"If everyone were born blind, no one would know how to read",
"When you get a tattoo on your back, it’s a reminder of the pain you’ve had to deal with",
"When you get older, you realize that you’re not the same person you used to be",
"It is socially unacceptable to wear a shirt with a hole in the front",
"There isn't a single person in the world who has never seen a picture of themselves with their eyes closed",
"Cats are probably more afraid of humans than humans are of cats",
"In the future we’d probably be able to use the same voice actor for both the “real world” and “virtual world”",
"The fact that the word \"cringe\" has become a word of endearment is pretty cringe",
"We’re all just living in a dream",
]

gpt2B = [
"A single-use item can be used to make more than a dozen different items",
"You can't tell how good or bad a song is until you listen to the lyrics",
"If you’ve ever been to the moon, you know how hard it is to get out",
"The real villains in the Harry Potter series are the Ministry of Magic and Voldemort himself",
"There could very well be a person out there that has the same username as you, but you don't know it because they're not in your social media",
"When you go to a restaurant you are paying a restaurant for your food",
"A man wearing a mask is a man with a mask",
"Cats must think that we are their pets. We are their pets",
"If every movie was a true story, then no one would ever see it",
"The people who are the most likely to have a bad day are those who don’t have a bad day",
]

criterias = [
  "I like this Showerthought",           
  "It makes a true/valid/logical statement",
  "It is creative",
  "It is funny",
  "It is clever",
  "I believe this Showerthought has been written by a real person"
]

titleRow = [
  ('Model',)
]
rows = [
  ('Real', ),
  ('GPT-Neo', ),
  ('ChatGPT',),
  ('GPT-2', ),
]
conditions = [
  # Trained a Machine Learning model at least once (e.g., in a university project)
  (1,'without condition'), # blank tuple for running the test in general without any conitions
  ('How experienced are you in using Machine Learning models?', "Working with Meaching Learning models regularly (e.g., part of study focus or job)"),
  ('How experienced are you in using Machine Learning models?', "Trained a Machine Learning model at least once (e.g., in a university project)"),
  ('How experienced are you in using Machine Learning models?', 'Using a product with features of which you know that they use AI / Machine Learning'),
  ('How experienced are you in using Machine Learning models?', 'No experience'),
  ('Are you familiar with the r/Showerthoughts community?', 'Interact (post, up/downvote, or comment) regularly'),
  ('Are you familiar with the r/Showerthoughts community?', 'Interact (post, up/downvote, or comment) rarely'),
  ('Are you familiar with the r/Showerthoughts community?', 'Visited sometime in the past'),
  ('Are you familiar with the r/Showerthoughts community?', 'No never heard of it'),
  ('How old are you?', '<20'),
  ('How old are you?', '20-25'),
  ('How old are you?', '26-30'),
  ('How old are you?', '31-40'),
  ('How old are you?', '41-55'),
  ('How often do you visit Reddit?', 'Never'),
  ('How often do you visit Reddit?', 'Rarely'),
  ('How often do you visit Reddit?', 'Weekly'),
  ('How often do you visit Reddit?', 'Daily'),
]
for idx, condition in enumerate(conditions):
  workingDfA = dfA if idx == 0 else dfA[dfA[condition[0]] == condition[1]]
  workingDfB = dfB if idx == 0 else dfB[dfB[condition[0]] == condition[1]]
  #dfB = dfB[dfB[conditions[0][0]] == conditions[0][1]]
  for criteria in criterias:
    realILike = []
    for item in realSurveyA:
      realILike = [*realILike, *(workingDfA[f"{item} [{criteria}]"].values)]
    for item in realSurveyB:
      realILike = [*realILike, *(workingDfB[f"{item} [{criteria}]"].values)]

      
    neoILike = []
    for item in neoA:
      neoILike = [*neoILike, *(workingDfA[f"{item} [{criteria}]"].values)]
    for item in neoB:
      neoILike = [*neoILike, *(workingDfB[f"{item} [{criteria}]"].values)]
      
    chatILike = []
    for item in chatA:
      chatILike = [*chatILike, *(workingDfA[f"{item} [{criteria}]"].values)]
    for item in chatB:
      chatILike = [*chatILike, *(workingDfB[f"{item} [{criteria}]"].values)]

    gpt2ILike = []
    for item in gpt2A:
      gpt2ILike = [*gpt2ILike, *(workingDfA[f"{item} [{criteria}]"].values)]
    for item in gpt2B:
      gpt2ILike = [*gpt2ILike, *(workingDfB[f"{item} [{criteria}]"].values)]
    
    temp = criteria + f" " + condition[1].replace(',', ' ')
    titleRow[0] = titleRow[0] + (temp, )
    rows[0] = rows[0] + (np.average(realILike) if len(realILike) else 0,)
    rows[1] = rows[1] + (np.average(neoILike) if len(neoILike) else 0,)
    rows[2] = rows[2] + (np.average(chatILike) if len(chatILike) else 0,)
    rows[3] = rows[3] + (np.average(gpt2ILike) if len(gpt2ILike) else 0,)

with open ('survey_additional_evaluation.csv', 'w') as f:
  wr = csv.writer(f)
  wr.writerows(titleRow)
  wr.writerows(rows)
