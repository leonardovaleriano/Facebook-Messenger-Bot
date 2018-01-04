import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

person_name = input('Enter your full name: ')
fb_data = input('Do you have Facebook data to parse through (y/n)?')
google_data = input('Do you have Google Hangouts data to parse through (y/n)?')
linkedin_data = input('Do you have LinkedIn data to parse through (y/n)?')


def get_google_hangouts_data():
    # Putting all the file names in a list
    all_files = []
    # Edit these file and directory names if you have them saved somewhere else
    for filename in os.listdir('GoogleTextForm'):
        if filename.endswith(".txt"):
            all_files.append('GoogleTextForm/' + filename)

    # The key is the other person's message, and the value is my response
    response_dictionary = dict()

    # Going through each file, and recording everyone's messages to me, and my responses
    for current_file in all_files:
        my_message, other_people_message, current_speaker = "", "", ""
        opened_file = open(current_file, 'r', encoding="utf8")
        all_lines = opened_file.readlines()
        for index, lines in enumerate(all_lines):
            # The sender's name is separated by < and >
            left_bracket = lines.find('<')
            right_bracket = lines.find('>')

            # Find messages that I sent
            if lines[left_bracket+1:right_bracket] == person_name:
                if not my_message:
                    # Want to find the first message that I send (if I send multiple in a row)
                    start_message_index = index - 1
                my_message = my_message + " " + lines[right_bracket+1:]

            elif my_message:
                # Now go and see what message the other person sent by looking at previous messages
                for counter in range(start_message_index, 0, -1):
                    current_line = all_lines[counter]
                    # In case the message above isn't in the right format
                    if current_line.find('<') < 0 or current_line.find('>') < 0:
                        my_message, other_people_message, current_speaker = "", "", ""
                        break
                    if not current_speaker:
                        # The first speaker not named me
                        current_speaker = current_line[current_line.find('<')+1:current_line.find('>')]
                    elif current_speaker != current_line[current_line.find('<')+1:current_line.find('>')]:
                        # A different person started speaking, so now I know that the first person's message is done
                        other_people_message = clean_message(other_people_message)
                        my_message = clean_message(my_message)
                        response_dictionary[other_people_message] = my_message
                        break
                    other_people_message = current_line[current_line.find('>')+1:] + " " + other_people_message
                my_message, other_people_message, current_speaker = "", "", ""
    return response_dictionary


def get_facebook_data():
    response_dictionary = dict()
    fb_file = open('fbMessages.txt', 'r', encoding="utf8")
    all_lines = fb_file.readlines()
    my_message, other_people_message, current_speaker = "", "", ""
    for index, lines in enumerate(all_lines):
        right_bracket = lines.find(']') + 2
        just_message = lines[right_bracket:]
        colon = just_message.find(':')
        # Find messages that I sent
        if just_message[:colon] == person_name:
            if not my_message:
                # Want to find the first message that I send (if I send multiple in a row)
                start_message_index = index - 1
            my_message = my_message + " " + just_message[colon+2:]

        elif my_message:
            # Now go and see what message the other person sent by looking at previous messages
            for counter in range(start_message_index, 0, -1):
                current_line = all_lines[counter]
                right_bracket = current_line.find(']') + 2
                just_message = current_line[right_bracket:]
                colon = just_message.find(':')
                if not current_speaker:
                    # The first speaker not named me
                    current_speaker = just_message[:colon]
                elif current_speaker != just_message[:colon] and other_people_message:
                    # A different person started speaking, so now I know that the first person's message is done
                    other_people_message = clean_message(other_people_message)
                    my_message = clean_message(my_message)
                    response_dictionary[other_people_message] = my_message
                    break
                other_people_message = just_message[colon+2:] + " " + other_people_message
            my_message, other_people_message, current_speaker = "", "", ""
    return response_dictionary


def get_linkedin_data():
    df = pd.read_csv('Inbox.csv')
    date_time_converter = lambda x: datetime.strptime(x,'%B %d, %Y, %I:%M %p')
    response_dictionary = dict()
    people_contacted = df['From'].unique().tolist()
    for person in people_contacted:
        received_messages = df[df['From'] == person]
        sent_messages = df[df['To'] == person]
        if len(sent_messages) == 0 or len(received_messages) == 0:
            # There was no actual conversation
            continue
        combined = pd.concat([sent_messages, received_messages])
        combined['Date'] = combined['Date'].apply(date_time_converter)
        combined = combined.sort(['Date'])
        other_people_message, my_message = "", ""
        first_message = True
        for index, row in combined.iterrows():
            if row['From'] != person_name:
                if my_message and other_people_message:
                    other_people_message = clean_message(other_people_message)
                    my_message = clean_message(my_message)
                    response_dictionary[other_people_message.rstrip()] = my_message.rstrip()
                    other_people_message, my_message = "", ""
                other_people_message = other_people_message + row['Content'] + " "
            else:
                if first_message:
                    first_message = False
                    # Don't include if I am the person initiating the convo
                    continue
                my_message = my_message + str(row['Content']) + " "
    return response_dictionary


def clean_message(message):
    # Remove new lines within message
    cleaned_message = message.replace('\n', ' ').lower()
    # Deal with some weird tokens
    cleaned_message = cleaned_message.replace("\xc2\xa0", "")
    # Remove punctuation
    cleaned_message = re.sub('([.,!?])', '', cleaned_message)
    # Remove multiple spaces in message
    cleaned_message = re.sub(' +', ' ', cleaned_message)
    return cleaned_message

combined_dictionary = {}
if google_data == 'y':
    print('Getting Google Hangout Data')
    combined_dictionary.update(get_google_hangouts_data())
if fb_data == 'y':
    print('Getting Facebook Data')
    combined_dictionary.update(get_facebook_data())
if linkedin_data == 'y':
    print('Getting LinkedIn Data')
    combined_dictionary.update(get_linkedin_data())
print('Total len of dictionary', len(combined_dictionary))

print('Saving conversation data dictionary')
np.save('conversationDictionary.npy', combined_dictionary)

conversation_file = open('conversationData.txt', 'w', encoding="utf8")
for key, value in combined_dictionary.items():
    if not key.strip() or not value.strip():
        # If there are empty strings
        continue
    conversation_file.write(key.strip() + " " + value.strip())
