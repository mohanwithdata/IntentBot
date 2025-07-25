from assistant.chatbot import ChatbotAssistant
from functions.stock_functions import get_stocks

assistant = ChatbotAssistant('data/intents.json', function_mappings={'stocks': get_stocks})
assistant.parse_intents()
assistant.load_model('models/trained_model.pth', 'outputs/dimensions.json')

print("Chatbot is ready! Type '/quit' to exit.")

while True:
    msg = input("You: ")
    if msg == '/quit':
        break
    print("Bot:", assistant.process_message(msg))