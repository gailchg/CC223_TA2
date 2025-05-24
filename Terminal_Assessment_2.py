from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

# Load BlenderBot model & tokenizer 
model_name = "facebook/blenderbot-1B-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Store chat history
chat_history = []

def chatbot_response(user_input):
    global chat_history
    
    # Add the new user input to chat history
    chat_history.append(user_input)

    # Limit chat history to the last 5 messages to avoid exceeding token length limits
    chat_history = chat_history[-5:] 

    # Encode the conversation history into input tensors
    inputs = tokenizer("\n".join(chat_history), return_tensors="pt", truncation=True, max_length=512)

    # Generate response
    reply_ids = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    # Add bot response to chat history
    chat_history.append(response)

    return response

# Main program 
if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' to end the chat.")

    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if user wants to exit the chat
        if user_input.lower() == "exit":
            break
        
        # Generate and display bot response
        response = chatbot_response(user_input)
        print("Bot:", response)


        