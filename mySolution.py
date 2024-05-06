# import all the important libraries 
import transformers
from transformers import RobertaTokenizer
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn import metrics
import os
import re
import json
import argparse
from tqdm import tqdm

# import all the classes 

class ParagraphDataset(Dataset):
    # Initialize the dataset
    def __init__(self, dataframe, tokenizer, max_len):
        
        self.tokenizer = tokenizer  
        self.data = dataframe  
        self.texts = dataframe.texts  
        #self.targets = self.data.changes  
        self.max_len = max_len  

    # Return the length of the dataset based on the number of items in 'texts'
    def __len__(self):
        return len(self.texts)

    # Get a specific item from the dataset based on the index 'idx'
    def __getitem__(self, idx):
        texts = self.texts[idx]  
        
        # Tokenize the text using the provided tokenizer
        encoding = self.tokenizer.encode_plus(
            texts,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = encoding['input_ids']  
        mask = encoding['attention_mask']  
        
        # Return a dictionary containing token IDs, attention masks, target values, and the original text
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            #'targets': torch.tensor(self.targets[idx]),
            'texts': texts
        }
    
    
# Define the RoBERTaClass
class RoBERTaClass(torch.nn.Module):
    def __init__(self):
        super(RoBERTaClass, self).__init__()
        self.dropout = 0.5
        self.hidden_embd = 768
        self.output_layer = 1
        # Declare the layers here
        self.l1 = transformers.RobertaModel.from_pretrained('roberta-base')
        self.l2 = torch.nn.Dropout(self.dropout)
        self.l3 = torch.nn.Linear(self.hidden_embd, self.output_layer)

    def forward(self, ids, mask):
        # Use the transformer, then the dropout and the linear in that order.
        _, output_1 = self.l1(ids, attention_mask = mask, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
    

# neccessary functions for the solution

def list_converter(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    paragraph_list = content.split('\n')

    return paragraph_list

# Function to read a JSON file and return its content as a dictionary
def truth_labels(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        content = file.read()

    return json.loads(content)

def load_file(text_file_path):
    paragraph_list = list_converter(text_file_path)
    #label_data = truth_labels(json_file_path)

    return paragraph_list



# Function to concatenate adjacent paragraphs in a dataframe
def concatenate_paragraphs(df):
    new_texts = []
    #new_changes = []
    for i in range(len(df['texts'])):
        combined_texts = []
        #changes = []
        paragraphs = df['texts'][i]
        for j in range(len(paragraphs) - 1):
            combined_text = paragraphs[j] + '[CLS]' + paragraphs[j + 1]
            #change = df['changes'][i][j]
            combined_texts.append(combined_text)
            #changes.append(change)
        new_texts.extend(combined_texts)
        #new_changes.extend(changes)

    return new_texts



def create_csv(text_file_path):
    paragraphs = load_file(text_file_path)

    # Since you're only handling a single file, the structure of paragraphs and labels is simpler
    texts = [paragraphs]
    #changes = [labels['changes']]

    df = pd.DataFrame({'texts': texts})
    
    
    try:
        concatenated_texts = concatenate_paragraphs(df)
    except IndexError as e:
        print(f"Error occurred while processing file: {text_file_path}")
        raise e
    
    

    # Create a single dataframe with the concatenated paragraphs
    df = pd.DataFrame({'texts': concatenated_texts})

   

    return df


def create_dataloader(df, tokenizer, batch_size=16):
    dataset = ParagraphDataset(df, tokenizer, MAX_LEN)
    loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 0}
    data_loader = DataLoader(dataset, **loader_params)
    return data_loader

def predict(model, data_loader):
    model.eval()
    predictions = []
    texts = []

    with torch.no_grad():
        for data in data_loader:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            text_data = data['texts']

            outputs = model(ids, mask)
            prediction_batch = torch.sigmoid(outputs).cpu().detach().numpy()
            # Convert boolean predictions to 0s and 1s
            prediction_batch = (prediction_batch > 0.5).astype(int)

            predictions.extend(prediction_batch.tolist())
            texts.extend(text_data)

    return predictions, texts

import json
import os


def save_predictions_to_json(problem_id, predictions, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the file path
    output_file_path = os.path.join(output_dir, f'solution-problem-{problem_id}.json')
    
    # Create the data dictionary
    data = {'changes': predictions}
    
    # Write the data to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(data, f)

    print(f'File saved: {output_file_path}')


# extra helper functions 
def parse_args():
    parser = argparse.ArgumentParser(description='PAN24 Multi-Author Writing Style Analysis task.')

    parser.add_argument('--input', type=str, help='The input data for three sub tasks (expected in txt format in `easy/medium/hard` subfolders).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in json files in `easy/medium/hard` subfolders.', required=True)

    return parser.parse_args()


# function where everything combines 
def run_solution(problems_folder, output_folder, tokenizer, model):
    os.makedirs(output_folder, exist_ok=True)
    print(f'Write outputs to {output_folder}.')

    for file in tqdm(os.listdir(problems_folder)):
        if file.startswith('problem-') and file.endswith('.txt'):
            problem_id = file.split('.')[0].split('-')[1]
            text_file_path = os.path.join(problems_folder, file)
            #json_file_path = os.path.join(problems_folder, f'truth-problem-{problem_id}.json')
            df = create_csv(text_file_path)
            data_loader = create_dataloader(df, tokenizer, batch_size=16)
            predictions, _ = predict(model, data_loader)
            predictions = [item for sublist in predictions for item in sublist]
            save_predictions_to_json(problem_id, predictions, output_dir=output_folder)


if __name__ == '__main__':
    args = parse_args()
    # Here, you should enter the path to your trained models.
    model_easy = "easy_model.pt"
    model_medium = "medium_model.pt"
    model_hard = "hard_model.pt"

    
    # Setting the tokenizer to be used
    MAX_LEN = 128 
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Setting up the device for GPU usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.to(device)

    for subtask in ["easy", "medium", "hard"]:
        if subtask == "easy":
            PATH = model_easy
        elif subtask == "medium":
            PATH = model_medium
        elif subtask == "hard":
            PATH = model_hard
            

        model = torch.load(PATH, map_location=device)
        model.to(device)
        model.eval()
        run_solution(args.input + f"/{subtask}", args.output + f"/{subtask}", tokenizer, model)

