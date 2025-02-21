from telegram import Update, ForceReply
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import os
from dotenv import load_dotenv
load_dotenv()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm Luna chatbot. How can I assist you today?", reply_markup=ForceReply(selective=True))

async def switch_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    available_models = ["mistral-large", "openai", "openai-large", "qwen-coder", "llama", "deepseek-r1", "claude-hybridspace"]
    
    if not context.args or context.args[0].lower() not in [m.lower() for m in available_models]:
        model_list = "\n".join(f"- {model}" for model in available_models)
        await update.message.reply_text(
            f"Please specify a valid model name:\n{model_list}\n\nUsage: /model model_name"
        )
        return

    selected_model = context.args[0].lower()
    context.user_data['selected_model'] = selected_model
    await update.message.reply_text(f"Switched to {selected_model} model!")

async def chat_with_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text

    # Extract conversation history from the message if present
    if 'history' not in context.user_data:
        # Initialize history with system prompt
        context.user_data['history'] = [{
            "role": "system",
            "content": "You are Luna, a helpful and friendly AI assistant. You provide clear, concise, and accurate responses while maintaining a pleasant conversational tone."
        }]
    
    history = context.user_data['history']

    # Append the latest user message to the history
    history.append({"role": "user", "content": user_input})

    # Get selected model or use default
    selected_model = context.user_data.get('selected_model', 'mistral-large')

    # Construct the payload with the entire conversation history
    payload = {
        "messages": history,
        "model": selected_model,
        "jsonMode": False,
        "seed": None,
        "stream": False
    }

    # Use the base URL
    api_url = "https://text.pollinations.ai/"

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        
        # Print response for debugging
        # print(f"Response status: {response.status_code}")
        # print(f"Response content: {response.text}")
        
        # Use response.text directly since the API returns plain text
        bot_response = response.text if response.text else "I'm sorry, I couldn't process that."
            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        bot_response = "I'm sorry, there was an error connecting to the service."

    # Append bot response to history
    history.append({"role": "assistant", "content": bot_response})

    # Update the user's conversation history
    context.user_data['history'] = history

    # Send the bot's response to the user
    await update.message.reply_text(bot_response, reply_markup=ForceReply(selective=True))

if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("model", switch_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_bot))
    print("Bot is running...")
    app.run_polling()
