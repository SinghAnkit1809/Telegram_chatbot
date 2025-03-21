from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, ContextTypes, CallbackQueryHandler
import json
import time
import os
from functools import wraps
import asyncio

# Constants
ADMIN_USERNAME = "@ankitSingh1809"
CONFIG_FILE = "admin_config.json"

# Default configuration
DEFAULT_CONFIG = {
    "features": {
        "chat": True,
        "imagine": True,
        "video": True,
        "model_switch": True,
        "audio": True
    },
    "broadcast_message": "",
    "broadcast_active": False,
    "broadcast_id": None,  # To track current broadcast
    "broadcast_seen": {}  # To track which users have seen the broadcast
}
USERS_FILE = "registered_users.json"
def load_registered_users():
    """Load registered users from file or return empty set if file doesn't exist"""
    try:
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
            return set(users)  # Convert list back to set
    except FileNotFoundError:
        return set()

# Initialize or load config
def get_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Ensure audio feature exists (in case of older config files)
            if "audio" not in config["features"]:
                config["features"]["audio"] = True
                save_config(config)
            return config
    except FileNotFoundError:
        # Use the default structure
        return DEFAULT_CONFIG.copy()

# Save config
def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# Admin check decorator
def admin_only(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user = update.effective_user
        # Check if user is admin
        if user.username == ADMIN_USERNAME.replace("@", ""):
            return await func(update, context, *args, **kwargs)
        else:
            await update.message.reply_text("⛔ This command is restricted to admin use only.")
            return None
    return wrapped

# Check if feature is enabled
def is_feature_enabled(feature_name):
    config = get_config()
    return config["features"].get(feature_name, True)

# Command handlers
@admin_only
async def admin_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin dashboard with all controls"""
    config = get_config()
    
    # Create feature toggle buttons
    feature_buttons = []
    for feature, enabled in config["features"].items():
        status = "✅ ON" if enabled else "❌ OFF"
        feature_buttons.append([
            InlineKeyboardButton(f"{feature.capitalize()}: {status}", 
                               callback_data=f"toggle:{feature}")
        ])
    
    # Add broadcast button
    feature_buttons.append([
        InlineKeyboardButton("📢 Send Broadcast", callback_data="admin:broadcast")
    ])
    
    # Add stats button
    feature_buttons.append([
        InlineKeyboardButton("📊 View Stats", callback_data="admin:stats")
    ])
    
    await update.message.reply_text(
        "🔐 Admin Dashboard\n\nManage bot features and settings:",
        reply_markup=InlineKeyboardMarkup(feature_buttons)
    )

@admin_only
async def broadcast_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Please provide a message to broadcast.\nExample: /broadcast Hello everyone!"
        )
        return
    
    message = ' '.join(context.args)
    registered_chats = load_registered_users()
    
    sent_count = 0
    failed_count = 0
    for chat_id in registered_chats:
        try:
            await context.bot.send_message(chat_id=chat_id, text=message)
            sent_count += 1
        except Exception as e:
            print(f"Failed to send broadcast to {chat_id}: {e}")
            failed_count += 1
    
    await update.message.reply_text(
        f"Broadcast complete. Sent: {sent_count} chats, Failed: {failed_count} chats."
    )


async def handle_admin_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle admin dashboard callbacks"""
    query = update.callback_query
    user = update.effective_user
    
    # Admin check for callbacks
    if user.username != ADMIN_USERNAME.replace("@", ""):
        await query.answer("⛔ Admin access required", show_alert=True)
        return
    
    await query.answer()
    config = get_config()
    
    if query.data.startswith("toggle:"):
        # Feature toggle logic
        feature = query.data.split(":")[1]
        
        # Toggle the feature
        config["features"][feature] = not config["features"][feature]
        save_config(config)
        
        # Update message with new status
        status = "✅ ON" if config["features"][feature] else "❌ OFF"
        await query.edit_message_text(
            f"Feature '{feature}' has been turned {status}.\n\n"
            f"Return to dashboard with /admin",
            reply_markup=None
        )
    
    elif query.data == "admin:broadcast":
        # Show broadcast form
        await query.edit_message_text(
            "Use /broadcast command followed by your message to send a broadcast.\n"
            "Example: /broadcast Hello everyone! Check out our new features!"
        )
    
    elif query.data == "admin:stats":
        # Show basic stats (you can expand this)
        await query.edit_message_text(
            "📊 Bot Statistics\n\n"
            "Active features:\n" + 
            "\n".join([f"- {f}: {'✅' if s else '❌'}" for f, s in config["features"].items()]) +
            "\n\nReturn to dashboard with /admin"
        )
    
    elif query.data.startswith("broadcast:"):
        action = query.data.split(":")[1]
        
        if action == "confirm":
            message = context.user_data.get('broadcast_message', '')
            if message:
                # Get all known chat IDs from context.bot
                try:
                    # Send message to all users
                    sent_count = 0
                    failed_count = 0
                    
                    # You need to maintain a list of chat_ids of users who have interacted with your bot
                    # This could be stored in a simple file or database
                    # For example, from your bot's chat_data or application's chat list
                    for chat_id in context.application.chat_data.keys():
                        try:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=f"📢 BROADCAST MESSAGE\n\n{message}"
                            )
                            sent_count += 1
                        except Exception as e:
                            print(f"Failed to send broadcast to {chat_id}: {e}")
                            failed_count += 1
                    
                    await query.edit_message_text(
                        f"✅ Broadcast sent!\n\n"
                        f"Successfully sent to: {sent_count} users\n"
                        f"Failed: {failed_count} users"
                    )
                except Exception as e:
                    await query.edit_message_text(f"❌ Error sending broadcast: {str(e)}")
            else:
                await query.edit_message_text("❌ No broadcast message found. Please try again.")
        
        elif action == "cancel":
            await query.edit_message_text("Broadcast cancelled.")

# Check for broadcasts and show to users
async def check_and_show_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check and show broadcast if user hasn't seen it"""
    if not update.effective_user:
        return
        
    user_id = str(update.effective_user.id)
    
    # Skip for admin
    if update.effective_user.username == ADMIN_USERNAME.replace("@", ""):
        return
        
    config = get_config()
    current_broadcast_id = config.get("broadcast_id")
    
    if (config.get("broadcast_active") and 
        config.get("broadcast_message") and 
        current_broadcast_id):
        
        # Check if user has seen this broadcast
        seen_users = config.get("broadcast_seen", {})
        if user_id not in seen_users.get(current_broadcast_id, []):
            # Show the broadcast
            await update.message.reply_text(
                f"📢 BROADCAST MESSAGE\n\n{config['broadcast_message']}"
            )
            
            # Mark as seen
            if current_broadcast_id not in seen_users:
                seen_users[current_broadcast_id] = []
            seen_users[current_broadcast_id].append(user_id)
            config["broadcast_seen"] = seen_users
            save_config(config)

# Function to disable broadcasting
@admin_only
async def stop_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop current broadcast"""
    config = get_config()
    config["broadcast_active"] = False
    save_config(config)
    
    await update.message.reply_text("✅ Broadcasting has been stopped.")

# Register admin handlers
def setup_admin_handlers(app):
    app.add_handler(CommandHandler("admin", admin_dashboard))
    app.add_handler(CommandHandler("broadcast", broadcast_message))
    app.add_handler(CommandHandler("stopbroadcast", stop_broadcast))
    app.add_handler(CallbackQueryHandler(handle_admin_callback, pattern='^(toggle:|admin:|broadcast:)'))