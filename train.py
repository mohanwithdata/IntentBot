import nltk
nltk.data.path.append('/home/codespace/nltk_data')
nltk.download('punkt')
nltk.download('wordnet')

from assistant.chatbot import ChatbotAssistant
from functions.stock_functions import get_stocks
from models.chatbot_model import ChatbotModel


assistant = ChatbotAssistant('data/intents.json', function_mappings={'stocks': get_stocks})
print("Parsing intents...")

assistant.parse_intents()
print("Preparing data...")


assistant.prepare_data()

input_size = assistant.X.shape[1]
output_size = len(assistant.intents)
assistant.model = ChatbotModel(input_size, output_size)

print("Training model...")

assistant.train_model(batch_size=8, lr=0.001, epochs=100)
print("Saving model...")

assistant.save_model('models/trained_model.pth', 'outputs/dimensions.json')

