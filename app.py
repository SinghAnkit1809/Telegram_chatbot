#telegram bot app.py
from telegram import Update, ForceReply, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from admin import setup_admin_handlers, is_feature_enabled, check_and_show_broadcast
from task_queue import task_queue, start_cleanup
import time
import io
import requests
import os
import random
from image_gen import generate_image
from video_gen import VideoGenerator
import asyncio
import json

def get_config():
    try:
        with open('admin_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Use the default structure from admin.py
        return {
            "features": {
                "chat": True,
                "imagine": True,
                "video": True,
                "model_switch": True
            },
            "broadcast_message": "",
            "broadcast_active": False,
            "broadcast_id": None,
            "broadcast_seen": {}
        }


async def check_membership(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if user is member of required channel"""
    user_id = update.effective_user.id
    try:
        # Use your channel's @username (including the @)
        channel_username = "@LunarckAI"
        member = await context.bot.get_chat_member(channel_username, user_id)
        return member.status in ['member', 'administrator', 'creator']
    except Exception as e:
        print(f"Membership check error: {e}")
        return False

async def send_join_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send channel join requirement message"""
    keyboard = [
        [InlineKeyboardButton("Join Lunarck AI", url="https://t.me/LunarckAI")],
        [InlineKeyboardButton("I Joined ✅", callback_data="verify_join")]
    ]
    if update.message:
        await update.message.reply_text(
            "🔒 You must join our official channel to use this bot!\n\n"
            "Please join the channel below then verify:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    elif update.callback_query:
        query = update.callback_query
        await query.edit_message_text(
            "🔒 You must join our official channel to use this bot!\n\n"
            "Please join the channel below then verify:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data.setdefault("registered_chats", set()).add(update.effective_chat.id)
    if not await check_membership(update, context):
        return await send_join_prompt(update, context)
    
    # Send welcome message
    await update.message.reply_text("Hello! I'm Luna chatbot. How can I assist you today?")
    
    # Check if there's an active broadcast message
    config = get_config()
    if config.get("broadcast_active") and config.get("broadcast_message"):
        await update.message.reply_text(
            f"📢 BROADCAST MESSAGE\n\n{config['broadcast_message']}"
        )


async def switch_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_membership(update, context):
        return await send_join_prompt(update, context)
    available_models = ["mistral-large", "openai", "openai-large", "qwen-coder", "llama", "deepseek-r1", "claude-hybridspace"]
    
    if context.args and context.args[0].lower() in [m.lower() for m in available_models]:
        selected_model = context.args[0].lower()
        context.user_data['selected_model'] = selected_model
        await update.message.reply_text(f"Switched to {selected_model} model!")
    else:
        # Create buttons in 2 columns for better width
        model_buttons = [
            [InlineKeyboardButton(m1, callback_data=f'chatmodel:{m1}'),
             InlineKeyboardButton(m2, callback_data=f'chatmodel:{m2}')] 
            for m1, m2 in zip(available_models[::2], available_models[1::2])
        ]
        # Add last model if odd count
        if len(available_models) % 2 != 0:
            model_buttons.append([InlineKeyboardButton(available_models[-1], callback_data=f'chatmodel:{available_models[-1]}')])
        
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
        # Edit original message to remove buttons
        await query.edit_message_text(
            text=f"✅ Switched to {selected_model} model!",
            reply_markup=None
        )

async def chat_with_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data.setdefault("registered_chats", set()).add(update.effective_chat.id)
    if not is_feature_enabled("chat"):
        await update.message.reply_text("⚠️ Chat functionality is currently disabled.")
        return
        
    if not await check_membership(update, context):
        return await send_join_prompt(update, context)
    
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

    if not is_feature_enabled("imagine"):
        await update.message.reply_text("⚠️ Image generation is currently disabled by the administrator.")
        return
    
    if not await check_membership(update, context):
        return await send_join_prompt(update, context)
    
    #await check_and_show_broadcast(update, context)

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
        # Show processing message
        await query.edit_message_text(
            "🎨 Generating your image...",
            reply_markup=None
        )
        
        image_url = generate_image(
            prompt=prompt,
            width=width,
            height=height,
            model=model,
            seed=random.randint(0, 1000000)
        )
        
        if image_url:
            # Remove buttons from both messages
            await query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=image_url,
                caption=f"🖼️ {prompt}\nModel: {model} | Size: {width}x{height}"
            )
        else:
            await query.edit_message_text("Failed to generate image. Please try again.")

async def video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_feature_enabled("video"):
        await update.message.reply_text("⚠️ Video generation is currently disabled.")
        return
        
    if not await check_membership(update, context):
        return await send_join_prompt(update, context)

    if not context.args:
        await update.message.reply_text(
            "Please provide a story prompt after the command.\n"
            "Example: /video a magical forest adventure"
        )
        return
    
    topic = ' '.join(context.args)
    
    # Send initial status message
    status_msg = await update.message.reply_text(
        "🎥 Video generation has been queued. You'll be notified when it's ready.\n"
        "You can continue using other bot features while your video is being generated."
    )
    
    try:
        # Create progress callback
        async def progress_update(message):
            try:
                await context.bot.edit_message_text(
                    chat_id=status_msg.chat_id,
                    message_id=status_msg.message_id,
                    text=f"🎥 Video generation in progress...\n\n{message}"
                )
            except Exception as e:
                print(f"Progress update error: {e}")
        
        # Initialize generator and start task
        generator = VideoGenerator()
        # Generate a unique task ID
        task_id = f"video_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Call _generate_video_task with task_id as a keyword argument
        task_id = await generator._generate_video_task(
            topic=topic,
            progress_callback=progress_update,
            task_id=task_id
        )
        
        # Start monitoring task status
        asyncio.create_task(
            check_video_task_status_loop(
                context,
                task_id=task_id,
                chat_id=update.effective_chat.id,
                message_id=status_msg.message_id,
                topic=topic
            )
        )
        
    except Exception as e:
        error_msg = f"❌ Video generation failed: {str(e)}"
        print(error_msg)
        await update.message.reply_text(error_msg)

# Add new function for status checking loop
async def check_video_task_status_loop(context, task_id, chat_id, message_id, topic):
    """Continuously check video task status"""
    try:
        max_attempts = 20  # Maximum number of attempts (10 minutes total with 30s interval)
        attempts = 0
        
        while attempts < max_attempts:
            task_status = task_queue.get_task_status(task_id)
            status = task_status.get('status', 'not_found')
            
            if status == 'completed':
                video_data = task_status.get('result')
                if video_data and isinstance(video_data, (bytes, bytearray)):
                    try:
                        # Send video with more detailed error handling
                        await context.bot.send_video(
                            chat_id=chat_id,
                            video=io.BytesIO(video_data),
                            caption=f"✅ Your video about '{topic}' is ready!",
                            supports_streaming=True
                        )
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=message_id,
                            text="✅ Video generation completed!"
                        )
                        print(f"Video sent successfully for task {task_id}")
                        return
                    except Exception as e:
                        print(f"Error sending video: {str(e)}")
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=message_id,
                            text=f"❌ Error sending video: {str(e)}"
                        )
                        return
                else:
                    print(f"Invalid video data received: {type(video_data)}")
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text="❌ Video generation failed: Invalid video data received"
                    )
                    return
            
            elif status == 'failed':
                error = task_status.get('error', 'Unknown error')
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=f"❌ Video generation failed: {error}"
                )
                return
            
            # Update progress message with attempt counter
            attempts += 1
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=f"🎥 Video generation in progress...\nTime elapsed: {attempts * 30} seconds"
            )
            
            await asyncio.sleep(30)
        
        # If we exit the loop without returning, we timed out
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text="❌ Video generation timed out. Please try again."
        )
        
    except Exception as e:
        print(f"Error in status check loop: {str(e)}")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=f"❌ Error checking video status: {str(e)}"
            )
        except:
            pass

# Add new callback handler
async def verify_join_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if await check_membership(update, context):
        await query.edit_message_text("✅ Verification successful! You can now use the bot.")
    else:
        await query.answer("❌ You haven't joined yet. Please join the channel first!", show_alert=True)

if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable missing!")
    
    # Initialize asyncio loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Start the task queue
    loop.run_until_complete(task_queue.start())
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Start the cleanup task with the loop
    start_cleanup(loop)
    
    # Add handlers
    app.add_handler(CallbackQueryHandler(verify_join_callback, pattern='^verify_join$'))
    setup_admin_handlers(app)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("model", switch_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_bot))
    app.add_handler(CommandHandler("imagine", imagine))
    app.add_handler(CallbackQueryHandler(handle_model_callback, pattern='^chatmodel:'))
    app.add_handler(CallbackQueryHandler(handle_image_callback))
    app.add_handler(CommandHandler("video", video))
    
    print("Bot is running...")
    try:
        app.run_polling()
    finally:
        # Cleanup
        loop.run_until_complete(task_queue.stop())
        loop.close()