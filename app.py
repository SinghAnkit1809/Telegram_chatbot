#telegram bot app.py
from telegram import Update, ForceReply, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from admin import setup_admin_handlers, is_feature_enabled, check_and_show_broadcast
from task_queue import task_queue, start_cleanup
from audio_gen import AudioGenerator
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
        try:
            model = query.data.split(':')[1]
            prompt = context.user_data['image_prompt']
            width, height = context.user_data['image_size']
            
            await query.edit_message_text(
                text="🎨 Generating your image...",
                reply_markup=None
            )
            
            print(f"Generating image with params: prompt={prompt}, width={width}, height={height}, model={model}")
            
            image_url = generate_image(
                prompt=prompt,
                width=int(width),
                height=int(height),
                model=model,
                seed=random.randint(0, 1000000)
            )
            
            print(f"Generated image URL: {image_url}")
            
            if image_url:
                # First try to download the image to verify it's accessible
                try:
                    img_response = requests.get(image_url, timeout=20)
                    img_response.raise_for_status()
                    
                    # Then send it as a file buffer (more reliable than URL)
                    await context.bot.send_photo(
                        chat_id=query.message.chat_id,
                        photo=io.BytesIO(img_response.content),
                        caption=f"🖼️ {prompt}\nModel: {model} | Size: {width}x{height}"
                    )
                    
                    # Update status afterward
                    await query.edit_message_text(
                        text="✅ Image generated successfully!",
                        reply_markup=None
                    )
                    
                except requests.exceptions.RequestException as e:
                    # Handle download error
                    print(f"Error downloading image: {str(e)}")
                    await query.edit_message_text(
                        text=f"❌ Error downloading image: Image URL inaccessible",
                        reply_markup=None
                    )
                    
                except Exception as e:
                    # Handle sending error
                    print(f"Error sending image: {str(e)}")
                    # Try sending just the URL as a fallback
                    await query.edit_message_text(
                        text=f"✅ Image generated! View here: {image_url}\n\nModel: {model} | Size: {width}x{height}",
                        reply_markup=None
                    )
                    
            else:
                await query.edit_message_text(
                    text="❌ Failed to generate image. Please try again.",
                    reply_markup=None
                )
                
        except Exception as e:
            print(f"Error in image generation: {str(e)}")
            await query.edit_message_text(
                text=f"❌ Error generating image: {str(e)}",
                reply_markup=None
            )
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
        "⏳ Checking video generation queue..."
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
        
        try:
            # Call _generate_video_task with task_id as a keyword argument
            # Now returns (task_id, position)
            task_result = await generator._generate_video_task(
                topic=topic,
                progress_callback=progress_update,
                task_id=task_id
            )
            
            # Unpack the result
            if isinstance(task_result, tuple) and len(task_result) == 2:
                task_id, queue_position = task_result
            else:
                task_id = task_result
                queue_position = 1  # Default if old format
            
            # Update message with queue position
            if queue_position > 1:
                await context.bot.edit_message_text(
                    chat_id=status_msg.chat_id,
                    message_id=status_msg.message_id,
                    text=f"🎥 Video generation has been queued.\n\n🔢 Your position: {queue_position} in queue\n\nYou'll be notified when your video is ready. You can continue using other bot features while waiting."
                )
            else:
                await context.bot.edit_message_text(
                    chat_id=status_msg.chat_id,
                    message_id=status_msg.message_id,
                    text=f"🎥 Video generation has started processing.\n\nYou'll be notified when it's ready. You can continue using other bot features while waiting."
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
            
        except ValueError as e:
            if "Queue is full" in str(e):
                await context.bot.edit_message_text(
                    chat_id=status_msg.chat_id,
                    message_id=status_msg.message_id,
                    text=f"⚠️ Video generation queue is full (max 20 tasks).\nPlease try again later."
                )
            else:
                raise
            
    except Exception as e:
        error_msg = f"❌ Video generation request failed: {str(e)}"
        print(error_msg)
        await context.bot.edit_message_text(
            chat_id=status_msg.chat_id,
            message_id=status_msg.message_id,
            text=error_msg
        )
# Add new function for status checking loop
async def check_video_task_status_loop(context, task_id, chat_id, message_id, topic):
    """Continuously check video task status with queue position updates"""
    try:
        max_attempts = 60  # Increased for longer wait in queue
        attempts = 0
        last_position = None
        
        while attempts < max_attempts:
            task_status = task_queue.get_task_status(task_id)
            status = task_status.get('status', 'not_found')
            
            if status == 'completed':
                video_data = task_status.get('result')
                if video_data and isinstance(video_data, (bytes, bytearray)):
                    try:
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=message_id,
                            text="✅ Video generation completed! Sending video now..."
                        )
                        video_stream = io.BytesIO(video_data)
                        # Send video with more detailed error handling
                        await context.bot.send_video(
                            chat_id=chat_id,
                            video=io.BytesIO(video_data),
                            caption=f"✅ Your video about '{topic}' is ready!",
                            supports_streaming=True
                        )
                        try:
                            await context.bot.edit_message_text(
                                chat_id=chat_id,
                                message_id=message_id,
                                text="✅ Video generation and delivery completed!"
                            )
                        except Exception as update_error:
                            print(f"Non-critical status update error: {str(update_error)}")
                            
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
                
            elif status == 'queued':
                position = task_status.get('position', 0)
                total = task_status.get('total_in_queue', 0)
                
                # Only update message if position changed
                if position != last_position:
                    last_position = position
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=f"⏳ Video waiting in queue...\nPosition: {position} of {total}\nEstimated wait: {position * 2}-{position * 4} minutes"
                    )
                    # Sleep longer if in queue to reduce API load
                    await asyncio.sleep(15)
                    continue
            
            elif status == 'processing':
                # Update progress message with attempt counter
                time_elapsed = attempts * 15  # Using 15-second intervals now
                minutes = time_elapsed // 60
                seconds = time_elapsed % 60
                
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=f"🎥 Video generation in progress...\nTime elapsed: {minutes}m {seconds}s"
                )
            
            # Increment attempt counter
            attempts += 1
            # Longer sleep interval to reduce API load
            await asyncio.sleep(15)
        
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

async def audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /audio command"""
    if not await check_membership(update, context):
        return await send_join_prompt(update, context)
    
    if not context.args:
        await update.message.reply_text(
            "Please provide the text to convert to speech.\n"
            "Example: /audio Hello, this is a test message."
        )
        return
    
    text = ' '.join(context.args)
    context.user_data['audio_text'] = text
    
    # Get all available voices
    audio_gen = AudioGenerator()
    all_voices = audio_gen.get_all_voices()
    
    # Create buttons categorized by language
    english_buttons = []
    indian_buttons = []
    
    for i, voice in enumerate(all_voices):
        voice_name = voice["name"]
        voice_id = voice["voice"]
        button = InlineKeyboardButton(voice_name, callback_data=f'audio:{voice_id}')
        
        if i < 5:  # First 5 are English
            english_buttons.append(button)
        else:  # Rest are Indian
            indian_buttons.append(button)
    
    # Arrange buttons in rows
    keyboard = []
    keyboard.append([InlineKeyboardButton("🇺🇸 English Voices", callback_data="audio:none")])
    keyboard.extend([english_buttons[i:i+2] for i in range(0, len(english_buttons), 2)])
    keyboard.append([InlineKeyboardButton("🇮🇳 Indian Voices", callback_data="audio:none")])
    keyboard.extend([indian_buttons[i:i+2] for i in range(0, len(indian_buttons), 2)])
    
    await update.message.reply_text(
        "Choose a voice for your audio:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_audio_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks for audio generation"""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith('audio:'):
        voice_id = query.data.split(':')[1]
        
        # Skip if this is a category header
        if voice_id == "none":
            return
        
        try:
            text = context.user_data.get('audio_text', "Hello, this is a test message.")
            
            # Update message to show generation status
            await query.edit_message_text(
                text="🔊 Generating your audio...",
                reply_markup=None
            )
            
            # Get voice name for display
            audio_gen = AudioGenerator()
            all_voices = audio_gen.get_all_voices()
            voice_name = next((v["name"] for v in all_voices if v["voice"] == voice_id), voice_id)
            
            # Generate audio
            audio_data = await audio_gen.generate_audio(text, voice_id)
            
            if audio_data:
                # Send the audio file
                await context.bot.send_audio(
                    chat_id=query.message.chat_id,
                    audio=io.BytesIO(audio_data),
                    title=f"TTS Audio - {voice_name}",
                    caption=f"🎵 Text-to-Speech Audio\nVoice: {voice_name}"
                )
                
                # Update status message
                await query.edit_message_text(
                    text="✅ Audio generated successfully!",
                    reply_markup=None
                )
            else:
                await query.edit_message_text(
                    text="❌ Failed to generate audio. Please try again.",
                    reply_markup=None
                )
                
        except Exception as e:
            print(f"Error in audio generation: {str(e)}")
            await query.edit_message_text(
                text=f"❌ Error generating audio: {str(e)}",
                reply_markup=None
            )

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
    app.add_handler(CommandHandler("audio", audio))
    app.add_handler(CallbackQueryHandler(handle_audio_callback, pattern='^audio:'))
    
    print("Bot is running...")
    try:
        app.run_polling()
    finally:
        # Cleanup
        loop.run_until_complete(task_queue.stop())
        loop.close()