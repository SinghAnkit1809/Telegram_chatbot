from telegram import Update, ForceReply, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import requests
import os
import random
from image_gen import generate_image

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm Luna chatbot. How can I assist you today?", reply_markup=ForceReply(selective=True))

async def switch_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    available_models = ["mistral-large", "openai", "openai-large", "qwen-coder", "llama", "deepseek-r1", "claude-hybridspace"]
    
    if context.args and context.args[0].lower() in [m.lower() for m in available_models]:
        selected_model = context.args[0].lower()
        context.user_data['selected_model'] = selected_model
        await update.message.reply_text(f"Switched to {selected_model} model!")
    else:
        # Create buttons for model selection
        model_buttons = [
            [InlineKeyboardButton(model, callback_data=f'chatmodel:{model}')] 
            for model in available_models
        ]
        await update.message.reply_text(
            "Choose a chat model:",
            reply_markup=InlineKeyboardMarkup(model_buttons)
        )

async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle chat model selection callback"""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith('chatmodel:'):
        selected_model = query.data.split(':')[1]
        context.user_data['selected_model'] = selected_model.lower()
        await query.edit_message_text(
            text=f"Switched to {selected_model} model!",
            reply_markup=None  # Remove the buttons
        )

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

async def imagine(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /imagine command"""
    if not context.args:
        await update.message.reply_text("Please provide a prompt after the command.\nExample: /imagine a sunset")
        return

    prompt = ' '.join(context.args)
    context.user_data['image_prompt'] = prompt
    
    # Create aspect ratio buttons
    size_buttons = [
        [InlineKeyboardButton("1:1 (Square)", callback_data='size:512x512')],
        [InlineKeyboardButton("9:16 (Portrait)", callback_data='size:576x1024')],
        [InlineKeyboardButton("16:9 (Landscape)", callback_data='size:1024x576')],
        [InlineKeyboardButton("2:1 (Wide)", callback_data='size:1024x512')]
    ]
    
    await update.message.reply_text(
        "Choose an aspect ratio:",
        reply_markup=InlineKeyboardMarkup(size_buttons)
    )

async def handle_image_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks for image generation"""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith('size:'):
        # Store size and show model selection
        width, height = query.data.split(':')[1].split('x')
        context.user_data['image_size'] = (width, height)
        
        # Model selection buttons
        models = ["Flux", "Flux-pro", "Flux-Realism", "Flux-anime", "Flux-3d", "Flux-cablyai", "Turbo"]
        model_buttons = [InlineKeyboardButton(m, callback_data=f'model:{m}') for m in models]
        
        await query.edit_message_text(
            text="Now choose a model:",
            reply_markup=InlineKeyboardMarkup([model_buttons[i:i+2] for i in range(0, len(model_buttons), 2)])
        )
    elif query.data.startswith('model:'):
        # Generate image with all parameters
        model = query.data.split(':')[1]
        prompt = context.user_data['image_prompt']
        width, height = context.user_data['image_size']
        
        image_url = generate_image(
            prompt=prompt,
            width=width,
            height=height,
            model=model,
            seed=random.randint(0, 1000000)
        )
        
        if image_url:
            await query.edit_message_reply_markup(reply_markup=None)  # Remove buttons
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=image_url,
                caption=f"🖼️ {prompt}\nModel: {model} | Size: {width}x{height}"
            )
        else:
            await query.edit_message_text("Failed to generate image. Please try again.")

if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable missing!")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("model", switch_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_bot))
    app.add_handler(CommandHandler("imagine", imagine))
    app.add_handler(CallbackQueryHandler(handle_image_callback))
    app.add_handler(CallbackQueryHandler(handle_model_callback, pattern='^chatmodel:'))
    print("Bot is running...")
    app.run_polling()
