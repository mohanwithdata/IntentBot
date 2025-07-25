import os, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.chatbot_model import ChatbotModel
from utils.preprocessing import tokenize_and_lemmatize, bag_of_words

class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.documents, self.vocabulary = [], []
        self.intents, self. intents_responses = [], {}
        self.function_mappings = function_mappings
        self.X, self.y = None, None

    def parse_intents(self):
        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)

        for intent in intents_data['intents']:
            if intent['tag'] not in self.intents:
                self.intents.append(intent['tag'])
                self.intents_responses[intent['tag']] = intent['responses']
            for pattern in intent['patterns']:
                pattern_words = tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, intent['tag']))
        
        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags, indices = [], []
        for words, intent in self.documents:
            bag = bag_of_words(words, self.vocabulary)
            bags.append(bag)
            indices.append(self.intents.index(intent))

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        if self.X is None or self.y is None:
            raise ValueError("Data not prepared. Call 'parse_intents()' and 'prepare_data()' before training.")

        input_size = self.X.shape[1]
        output_size = len(self.intents)

        self.model = ChatbotModel(input_size, output_size)

        X_tensor =  torch.tensor(self.X, dtype=torch.float32)
        y_tensor =  torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss =  criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epoch} - Loss: {total_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(dimensions_path), exist_ok=True)
        
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

    def load_model(self, model_path, dimensions_path):
        import json
        from models.chatbot_model import ChatbotModel

        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        
        self.input_size = dimensions['input_size']
        self.output_size = dimensions['output_size']

        self.model = ChatbotModel(self.input_size, self.output_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_message(self, message):
        words = tokenize_and_lemmatize(message)
        bag = bag_of_words(words, self.vocabulary)
        input_tensor = torch.tensor([bag], dtype=torch.float32)
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_tensor)

        predicted_index = torch.argmax(output).item()
        predicted_tag = self.intents[predicted_index]

        if self.function_mappings and predicted_tag in self.function_mappings:
            self.function_mappings[predicted_tag]()

        return random.choice(self.intents_responses[predicted_tag])                                       