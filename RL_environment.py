#!/usr/bin/env python
# coding: utf-8

# ## Reinforcement Learning Project

# In[152]:


import numpy as np
from typing import Dict, Tuple, List
import csv 
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib
import matplotlib.pyplot as plt
import math
import random
import os 
from PIL import Image 
from torch.utils.data import Dataset, DataLoader, SequentialSampler, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import json
from collections import Counter 
from tqdm import tqdm
from torch.nn.functional import softmax
import time
import concurrent.futures
from threading import Lock


# ### Prepare dataset

# In[153]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[154]:


class DataProcessor:
    def __init__(self, folder_path, disease_list):
        self.folder_path = folder_path
        self.disease_list = disease_list
        self.folders_with_diseases_labels = {}
        self.folder_name_with_diseases = []
        self.label_counts = None

    def read_data(self):
        for root, dirs, files in os.walk(os.path.join(self.folder_path, 'imgs')):
            for folder_name in dirs:
                folder_path = os.path.join(root, folder_name)
                
                detection_file_path = os.path.join(folder_path, 'detection.json')
                with open(detection_file_path, 'r') as detection_file:
                    detection_data = json.load(detection_file)

                    disease_labels = [label.lower() for item in detection_data for label in item.keys() if label in self.disease_list]
                    
                    for idx, label in enumerate(disease_labels):
                        if label == "effusion":
                            disease_labels[idx] = "pleural effusion" 
                        elif label == "cardiomegaly":
                            disease_labels[idx] = "cardiomyopathy"
                             
                    disease_labels = set(disease_labels) 
                    
                    # merge labels for images with multiple labels
                    if disease_labels:
                        merged_label = '-'.join(sorted(disease_labels))
                        self.folders_with_diseases_labels[folder_name] = merged_label
                        self.folder_name_with_diseases.append(folder_name)

    def delete_folders(self):
        # frequency of each merged label
        self.label_counts = Counter(self.folders_with_diseases_labels.values())

        # delete folders with label counts <= 3
        folders_to_delete = [folder_name for folder_name, label in self.folders_with_diseases_labels.items() if self.label_counts[label] <= 3]

        for folder_name in folders_to_delete:
            del self.folders_with_diseases_labels[folder_name]
            self.folder_name_with_diseases.remove(folder_name)
            
    def get_training_data(self):
        training_data = []
        for folder_name, label in self.folders_with_diseases_labels.items():
            folder_path = os.path.join(self.folder_path, 'imgs', folder_name)
            image_path = os.path.join(folder_path, 'source.jpg') 
            img = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img = transform(img)
            training_data.append((img, label))
        return training_data


folder_path = 'Slake1.0'
disease_list = ['Pneumothorax', 'Pneumonia', 'Effusion', 'Lung Cancer', "Cardiomegaly"]
data_processor = DataProcessor(folder_path, disease_list)
data_processor.read_data()
data_processor.delete_folders()
data = data_processor.get_training_data()


# In[155]:


data_labels=[x[1] for x in data]
SUPPORTED_CONDITIONS = set(data_labels) 
num_classes=len(SUPPORTED_CONDITIONS)


# In[156]:


#split training-test set
training_data, validation_data = train_test_split(data, test_size=0.2, random_state=42, shuffle = True, stratify=np.array(data_labels))

train_labels = [x[1] for x in training_data]
train_label_counts = dict(Counter(train_labels))
train_weight_samples = [1/train_label_counts[x] for x in train_labels]


# In[157]:


train_sampler = WeightedRandomSampler(train_weight_samples, num_samples=len(train_labels), replacement=True)
train_dataloader = DataLoader(training_data, sampler=train_sampler, batch_size=1)

val_sampler = SequentialSampler(validation_data)
val_dataloader = DataLoader(validation_data, sampler=val_sampler, batch_size=1)


# ### Prepare training environment

# In[158]:


#CNN Model
class FineTunedAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedAlexNet, self).__init__()
        
        alexnet = models.alexnet(pretrained=True)

        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc(x)
        return x

cnn_model = FineTunedAlexNet(num_classes=num_classes).to(device)
cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device(device)))
cnn_model.eval()


# In[159]:


# RL Environment 
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class Env: 
    _condition_symptom_probabilities: Dict[str, Dict[str, float]] # conditions with symptoms and their probabilities
    _actions: list[str] # symptoms
    _init_state: np.array
    _current_state: np.array
    _img: torch.tensor
    _condition: str # the condition which the simulated patient has
    _symptoms_of_condition: Dict[str, float] # symptoms of the condition which the simulated patient has
    _supported_conditions: list[str]
    _cnn_model: FineTunedAlexNet
    
    def __init__(self,
                 img: torch.tensor, 
                 condition: str,
                 cnn_model: FineTunedAlexNet
                ) -> None:  
        self._supported_conditions= ["pneumonia", "pneumothorax", "lung cancer", "pleural effusion", "cardiomyopathy"]
        self._img = img
        self._cnn_model = cnn_model 
        #if(condition is None): 
        #    condition = random.sample(self._supported_conditions,1)[0]
        self._condition = condition

        # init condition_symptom_probabilities from HealthKnowledgeGraph.csv
        self._condition_symptom_probabilities = self.load_condition_symptom_probabilities()
        # init condition_symptom_probabilities from slake knowledge graph 
        #self._condition_symptom_probabilities= dict()
        #with open('Slake1.0/KG/en_disease.csv', newline='') as csvfile:
        #    reader = csv.reader(csvfile, delimiter='#')
        #    reader.__next__() # skip header 
        #    for row in reader:
        #        if(row[1]!="symptom"):
        #            continue
        #        if(row[0] not in self._supported_conditions):
        #            continue
        #        self._condition_symptom_probabilities[row[0]] = dict()
        #        n_symptoms=len(row[2].split(','))
        #        uniform_prob = 1/(2**n_symptoms)
        #        for symptom in row[2].split(','):
        #            #assign uniform conditional probability because no conditional probability are available 
        #            self._condition_symptom_probabilities[row[0]][symptom.strip()] = uniform_prob

        # check if condition is valid
        if(self._condition not in self._condition_symptom_probabilities.keys()):
            raise ValueError('Unknow Condition: ' + condition + '. Please choose one of the following: ' + str(self._condition_symptom_probabilities.keys()))
        
        # init symptoms_of_condition for easier access
        self._symptoms_of_condition = self._condition_symptom_probabilities[self._condition]

        # init actions
        self._actions = list()
        for condition in self._condition_symptom_probabilities.keys(): 
            for symptom in list(self._condition_symptom_probabilities[condition]):
                if symptom not in self._actions:
                    self._actions.append(symptom) 


        
        #compute visual prior
        logits = self._cnn_model(img[None,:].to(device))[0]
        logits = logits.cpu()
        #sort logits to the same order as in self._condition_symptom_probabilities. CNN_model output: Lung Cancer: idx 0, Pneumothorax: idx 1, Pneumonia: idx 2, Effusion: idx 3, Cardiomegaly: idx 4 
        #logit_indicies = {
        #    "lung cancer": 0,
        #    "pneumothorax": 1, 
        #    "pneumonia": 2, 
        #    "pleural effusion": 3, 
        #    "cardiomyopathy": 4
        #}  
        logit_indicies = {
            "cardiomyopathy": 0,
            "pneumothorax": 1,
            "pneumonia": 2, 
            "lung cancer": 3,
            "pleural effusion": 4, 
        }
        condition_logit_idx = [logit_indicies[c] for c in self._condition_symptom_probabilities.keys()]
        visual_prior = softmax(torch.tensor([logits[idx] for idx in condition_logit_idx]))
        
        #visual_prior = np.ones(shape=(len(self._condition_symptom_probabilities.keys()))) #TODO: replace with cnn output 
        # init init_state = vector with cnn output (probabilities per condition) and history of asked symptoms (0=not asked, 1=symptom is present, -1=symptom is not present)
        self._init_state = np.concatenate((visual_prior,np.zeros((len(self._actions)))), axis=0)
        self._current_state = self._init_state

    def load_condition_symptom_probabilities(self) -> Dict[str, Dict[str, float]]:
        condition_symptom_probabilities = dict()

        with open('HealthKnowledgeGraph.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # skip header
            header = next(reader)  
            for row in reader:
                # to make it case insensitive
                condition = row[0].lower() 
                
                # if condition is in the supported conditions list then add the symptoms to the list
                if condition not in self._supported_conditions:
                    continue

                symptoms_and_probs = row[1].split(',')
                symptom_probabilities = dict()
                for symptom_prob in symptoms_and_probs:
                    # example for symptom_prob: pain (0.318)
                    symptom, prob = map(str.strip, symptom_prob.split('('))
                    # to remove the last parentheses ')'
                    prob = float(prob[:-1])  
                    symptom_probabilities[symptom] = prob
                condition_symptom_probabilities[condition] = symptom_probabilities 
        return condition_symptom_probabilities
    
    def posterior_of_condition(self, condition: str, useAddition=False) -> float: 
        #TODO: What is the correct likelihood calculation? If we use multiplication as in P(x,y)=P(x)*P(y), the likelihood gets smaller 
        #and nothing prevents the model from asking symptoms which are not related to the condition.
        if(useAddition):
            likelihood=0
        else:
            likelihood=1
        for idx, symptom in enumerate(self._actions):
            patient_answer = self._current_state[idx+len(self._condition_symptom_probabilities.keys())]
            #if (patient_answer==1) and (symptom not in self._condition_symptom_probabilities[condition].keys()):
            #    likelihood*= 0
            #elif (patient_answer==-1) and (symptom not in self._condition_symptom_probabilities[condition].keys()):
            #    likelihood*=1
            if (symptom not in self._condition_symptom_probabilities[condition].keys()):
                #TODO: Do we have to punish the model if a symptom is positive and is not related to the condition?
                continue 
            elif patient_answer==1:
                if(useAddition):
                    likelihood+=self._condition_symptom_probabilities[condition][symptom]
                else:
                    likelihood*=self._condition_symptom_probabilities[condition][symptom]
            elif patient_answer==-1:
                if(useAddition):
                    likelihood+=(1-self._condition_symptom_probabilities[condition][symptom]) 
                else:
                    likelihood*=(1-self._condition_symptom_probabilities[condition][symptom]) 

        prior = self._current_state[list(self._condition_symptom_probabilities.keys()).index(condition)]
        if(useAddition):
            result = likelihood+prior
        else:
            result = likelihood*prior
        return result
    
    def reward(self) -> float:
        #TODO: Is it a problem when the reward gets smaller and smaller?
        punishment=0
        for i in range(len(self._actions)):
            patient_answer = self._current_state[i+len(self._condition_symptom_probabilities.keys())]
            if (patient_answer!=0):
                punishment+=0.3
        return self.posterior_of_condition(self._condition, useAddition=True)-punishment
    
    def has_symptom(self, symptom: str) -> bool:
        if symptom not in self._symptoms_of_condition:
            return False
        else:
            phi = np.random.uniform()
            return phi <= self._symptoms_of_condition[symptom]

    def step(self, action_idx: int) -> Transition: 
        action = self._actions[action_idx]
        old_state = self._current_state.copy()
        self._current_state[len(self._condition_symptom_probabilities.keys()) + action_idx] = 1 if self.has_symptom(action) else -1

        # only give reward if it's a symptom of the condition
        #if(action in self._symptoms_of_condition):
        #    reward = self.reward()
        #else:
        #    reward = 0 
        return Transition(old_state, action_idx, self._current_state, self.reward())
    
    def reset(self) -> np.array:
        self._current_state = self._init_state
        return self._current_state
 


# In[160]:


# Experience Replay for RL
class ReplayMemory():
    def __init__(self, capacity):
        self.memory=deque([], maxlen=capacity)
    def push(self, transition):
        self.memory.append(transition)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


# In[161]:


# RL model
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128, dtype=torch.double)
        self.layer2 = nn.Linear(128, 128, dtype=torch.double)
        self.layer3 = nn.Linear(128, n_actions, dtype=torch.double)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# In[162]:


# BATCH_SIZE is the number of transitions sampled form the replay buffer
#GAMMA is the discount factor as mentioned in the previous section
#SIZE is the number of transitions sampled from the replay buffer
#EPS START is the starting value of epsilon
#EPS DECAY controls the rate of exposential decay of epsilon, higher means a slower decay
#EPS END is the final value of epsilon
#TAU is the update rate of the target network 
#LR the learning rate of the Adams optimizar
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TAU = 0.005
LR = 1e-5

#Get number of actions from dummy env 
img = Image.open("Slake1.0/imgs/xmlab333/source.jpg").convert('RGB') # TODO: use dummy img
transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
img = transform(img) 
myEnv=Env(img, 'pneumonia', cnn_model) 
n_actions = len(myEnv._actions)
n_observations = len(myEnv._current_state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0 


# In[225]:


def optimize_model(losses, gradient_norms):
    policy_net.train()
    if len(memory) < BATCH_SIZE: 
        return
    transitions = memory.sample(BATCH_SIZE)
    transitions = transitions
    #converts batch_array of Transitions to Transition of batch_arrays
    batch = Transition(*zip(*transitions))

    next_state_batch = torch.tensor(batch.next_state).to(device)
    state_batch = torch.tensor(batch.state).to(device)
    action_batch = torch.tensor(batch.action).to(device)
    reward_batch = torch.tensor(batch.reward).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch[None,:])

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]

    #TODO: state_action_values grow infinitely
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    #print("state_action_values: ", state_action_values)
    #print("action_batch: ", action_batch)
    #print("reward batch: ", reward_batch)
    #print("state batch[0]: ", state_batch[0])

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)

    losses.append((loss.detach()).cpu())

    parameters = [p for p in policy_net.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else: 
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).cpu() for p in parameters]), 2.0).item()
    gradient_norms.append(total_norm)
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()


# In[218]:


def select_action(myEnv, state, epsilon):
    randnum = np.random.rand(1) 
    if randnum < epsilon:
        action_idx = np.random.randint(len(myEnv._actions))
        #print("random")
    else:  
        with torch.no_grad():
            action_idx = np.argmax(policy_net(torch.tensor(state).to(device)).cpu()).item() 
            #print("non random")
            #print("network: ", policy_net(torch.tensor(state).to(device)).cpu())
    
    return action_idx


# ### Start with training


# In[190]:


lock = Lock() 
def training_episode(img: torch.tensor, condition: str, epsilon: float):
    myEnv=Env(img, condition, cnn_model)
    state = myEnv.reset()
    for _ in range(len_episode):  
        action_idx = select_action(myEnv, state, epsilon)
        #print(action_idx)
        transition = myEnv.step(action_idx)
        last_reward=transition.reward
        state = transition.next_state
        with lock: 
            rewards.append(transition.reward)
            memory.push(transition) 
    with lock:
        for _ in range(len_episode):  
            optimize_model(losses, gradient_norms)
    
            #Soft update of target network weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict) 
        total_reward_per_episode.append(last_reward) 
        


# In[ ]:


num_epochs = 16000
len_episode = 10
i_decay=1
epsilon = EPS_START
losses=[]
gradient_norms=[]
rewards=[]
epsilons=[]
total_reward_per_episode=[]


# In[ ]:


for _ in tqdm(range(num_epochs)):
    for batch in train_dataloader:  
        if epsilon > EPS_END:
            epsilon = EPS_START * math.exp(-i_decay/EPS_DECAY)
            i_decay+=1 
        epsilons.append(epsilon)
        condition = batch[1][0]
        img = batch[0][0] 
        training_episode(img, condition, epsilon)
        #pool.map(training_episode, (img, condition, epsilon))
print("complete") 


# In[ ]:

 


# In[ ]:


torch.save(policy_net.state_dict(), 'RL_model.pth')


# In[227]:


# helper function
def averagewindow(R, d=1): 
    n = len(R)
    t = []
    y = []
    for i in range(0,int(n/d)):
        t.append(np.mean(range(i*d,(i+1)*d)))
        y.append(np.mean(R[i*d:min(n,(i+1)*d)]))
    return t,y


# In[228]:


plt.plot(epsilons)
plt.savefig('epsilons.png')


# In[229]:


t,y = averagewindow(losses, d=50)
plt.plot(t,y)
plt.title("losses")
plt.savefig('losses.png')


# In[230]:


t,y = averagewindow(gradient_norms, d=50)
plt.plot(t,y)
plt.title("gradient norms")
plt.savefig('gradient norms.png')


# In[231]:


t,y = averagewindow(rewards, d=800)
plt.plot(t,y)
plt.title("Rewards")
plt.savefig('rewards.png')


# In[232]:


t,y = averagewindow(total_reward_per_episode, d=50)
plt.plot(t,y)
plt.title("Total reward per episode")


# ### Evaluation

# In[210]:


def topKAccuracy(k=3):
    policy_net.eval()
    epsilon = 0
    N_samples=0
    N_correct_samples=0
    for batch in val_dataloader:
        N_samples+=1
        condition = batch[1][0]
        img = batch[0][0] 
        print("Condition: ", condition)
        myEnv=Env(img, condition, cnn_model) 
        state = myEnv.reset()
        # ask patient 10 symptoms
        for _ in range(len_episode):
            action_idx = select_action(myEnv, state, epsilon)
            print("Action: ", myEnv._actions[action_idx])
            transition = myEnv.step(action_idx)
            print("Reward: ", transition.reward)
            state = transition.next_state
        #calculate posterior for every conditions
        posterior_of_conditions = []
        for condition in myEnv._supported_conditions:
            posterior = myEnv.posterior_of_condition(condition, useAddition=False)
            #set posterior to 0 if no symptom is related to the condition and therefore the likelihood stays
            #TODO: Delete when cnn is integrated
            if(posterior==1):
                posterior=0
            posterior_of_conditions.append((posterior, condition))
        #sort posteriors by value
        posterior_of_conditions.sort(key=lambda x: x[0])
        #get rank of posterior of correct condition
        rank = 1+next(i for i, val in enumerate(posterior_of_conditions)
                                  if val[1] == condition)
        if(rank <= k):
            N_correct_samples+=1 
        print(posterior_of_conditions) 
    return N_correct_samples / N_samples


# In[211]:


print(topKAccuracy(k=3)) #Random model would have 0.6


