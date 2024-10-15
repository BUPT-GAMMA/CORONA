from openai import OpenAI
import openai
import os
import time
import pickle
import pandas as pd
import requests
import re
import json
from sentence_transformers import SentenceTransformer
# client = OpenAI()
api_key = ""
url = ''
file_path = ''
g_model_type = "gpt-3.5-turbo-0613" 
# # "claude", "chatglm-6b", "hambuger-13b", "baichuan-7B", "gpt-4", "gpt-4-0613"
s_model = SentenceTransformer("all-MiniLM-L6-v2")

# Netflix
def construct_prompting(item_attribute, item_list): 
    # make history string
    history_string = "User history:\n" 
    for index in item_list:
        row = item_attribute.loc[item_attribute['id'] == index]
        if row.empty:
            print(f"Item ID index: {index} is not found.")
            continue
        year = row['year'].values[0]
        title = row['title'].values[0]
        history_string += "["
        history_string += str(index)
        history_string += "] "
        history_string += str(year) + ", "
        history_string += title + "\n"
    # output format
    # output_format = "Please output the following infomation of user, output format:\n{\'age\':age, \'gender\':gender, \'liked genre\':liked genre, \'disliked genre\':disliked genre, \'liked directors\':liked directors, \'country\':country\, 'language\':language}\nPlease do not fill in \'unknown\', but make an educated guess based on the available information and fill in the specific content.\nplease output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments.\n\n"
    # output_format = "Please make an educated guess based on the available information and fill in the specific content about this user.\nPlease think it step by step. First, summarize the included information from the user interaction history in 'summary'. Then, provide the specific user content in 'content'. Finally, give a brief explanation of your guessed content in 'explanation'.\nFinal output format:{'summary': interaction history summary, 'age':user age, 'gender':user gender, 'liked genre':user's liked genre, 'disliked genre':user's disliked genre, 'liked directors':user's liked directors, 'country':user's country, 'language':user's language, 'explanation': why you made such a guess}, nothing more nothing less. Attention: Do not fill in 'unknown', but make an educated guess!!! Strictly follow the output format!!"
    output_format = (
        f"Please infer the user's information in the format below. Base your answers on the movie history and do not use 'unknown' as a response. "
        f"You must infer from the patterns and information given.\n"
        f"Final output format:\n"
        f"{{\n"
        f"  'summary': interaction history summary, \n"
        f"  'age': user's likely age, \n"
        f"  'gender': user's likely gender, \n"
        f"  'liked genre': user's likely liked genre, \n"
        f"  'disliked genre': user's likely disliked genre, \n"
        f"  'liked directors': user's likely liked directors, \n"
        f"  'country': user's likely country, \n"
        f"  'language': user's likely language, \n"
        f"  'explanation': why you made such guesses based on the history\n"
        f"}}\n"
    )

    # make prompt
    # prompt = "You are required to generate user profile based on the history of user, that each movie with title, year, genre.\n"
    prompt = "You are required to generate a user profile based on the interaction history below. Each movie includes a title, year, and genre. You will use this history to make reasonable inferences about the user's characteristics.\n"
    prompt += history_string
    prompt += output_format
    return prompt

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Write a haiku about recursion in programming."
#         }
#     ]
# )
# print(completion.choices[0].message)
def LLM_request(model_type, prompt, retries=3):
    headers = {
        'Authorization': 'Bearer ' + api_key,
        'Content-Type': 'application/json',
    }
    data = {
        'model': model_type,
        'messages': [{"role": "user", "content": prompt}],
        'temperature': 0.7,
    }
    for i in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            message = response.json()
            content = message['choices'][0]['message']['content']
            return content
        except requests.exceptions.RequestException as e:
            print(f"Request failed, retrying {i+1}/{retries} times: {e}")
            time.sleep(2)
    content = None
    return content

    # print(f"content: {content}, model_type: {model_type}")

# LLM_request("gpt-4o", "Write a haiku about recursion in programming.")

### step1: generate user profiling ################################################################################## 
### read item_attribute
toy_item_attribute = pd.read_csv(file_path + '/netflix_image_text/item_attribute.csv', names=['id','year', 'title'])
### write augmented dict
augmented_user_profiling_dict = {}  
if os.path.exists(file_path + "/augmented_user_profiling_dict"): 
    print(f"The file augmented_user_profiling_dict exists.")
    augmented_user_profiling_dict = pickle.load(open(file_path + '/augmented_user_profiling_dict','rb')) 
else:
    print(f"The file augmented_user_profiling_dict does not exist.")
    pickle.dump(augmented_user_profiling_dict, open(file_path + '/augmented_user_profiling_dict','wb'))

### read adjacency_list
adjacency_list_dict = {}
train_mat = pickle.load(open(file_path + '/train_mat','rb'))
for index in range(train_mat.shape[0]):
    data_x, data_y = train_mat[index].nonzero()
    adjacency_list_dict[index] = data_y
test = json.load(open(file_path + '/test.json'))
test_index = [int(key) for key in test.keys()]
count = 0
for index in test_index:
    if index not in augmented_user_profiling_dict.keys() and len(test[str(index)]) != 0:
        print(index, flush=True)
        response = LLM_request('gpt-4o-2024-08-06', construct_prompting(toy_item_attribute, adjacency_list_dict[index]))
        augmented_user_profiling_dict[index] = response
        count += 1
        if count % 10 == 0 and count != 0:
            print(f'Test count: {count}', flush=True)
            pickle.dump(augmented_user_profiling_dict, open(file_path + '/augmented_user_profiling_dict','wb'))
pickle.dump(augmented_user_profiling_dict, open(file_path + '/augmented_user_profiling_dict','wb'))
start_id = 0
count = 0
fail_count = 0
for index in range(start_id, len(adjacency_list_dict.keys())):
    # if count > 10:
    #     break
    if index not in augmented_user_profiling_dict.keys() or augmented_user_profiling_dict[index] == None:
        print(index)
        response = LLM_request(g_model_type, construct_prompting(toy_item_attribute, adjacency_list_dict[index]))
        augmented_user_profiling_dict[index] = response
        count += 1
        if response == None:
            print(f'Index {index} request failed. Filled in blank.')
            fail_count += 1
            if fail_count > 10:
                raise Exception('Too many failed case.')
        if count % 20 == 0 and count != 0:
            print(f'All count: {count}')
            pickle.dump(augmented_user_profiling_dict, open(file_path + '/augmented_user_profiling_dict','wb'))
pickle.dump(augmented_user_profiling_dict, open(file_path + '/augmented_user_profiling_dict','wb'))
print('Augmentation finished.')  
user_profiles = [augmented_user_profiling_dict[index] for index in sorted(augmented_user_profiling_dict.keys())]
user_embedding = s_model.encode(user_profiles)
pickle.dump(user_embedding, open(file_path + '/augmented_user_init_embedding_correct','wb'))
#     # # make prompting
#     re = LLM_request(toy_item_attribute, adjacency_list_dict, index, g_model_type, augmented_user_profiling_dict, error_cnt)
# "claude", "chatglm-6b", "hambuger-13b", "baichuan-7B", "gpt-4", "gpt-4-0613"
### step1: generate user profiling ################################################################################## 