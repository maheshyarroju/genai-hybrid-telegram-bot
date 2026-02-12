# Used for environment variables and file/folder operations
import os

# Loads environment variables from a .env file
from dotenv import load_dotenv

# Telegram update object (message, user, chat etc.)
from telegram import Update

# Telegram bot framework components
from telegram.ext import (
    Application,       # Main bot application
    CommandHandler,    # Handles commands like /help, /ask, /image
    ContextTypes,      # Provides context inside handlers
    MessageHandler,    # Handles non-command messages (like photo uploads)
    filters,           # Used to filter message types (PHOTO, TEXT, etc.)
)

# Import Mini-RAG module (text-based retrieval system)
from rag import MiniRAG

# Import Vision module (image captioning system)
from vision import ImageDescriber


# Load environment variables from .env
load_dotenv()

# Get Telegram bot token securely from environment variable
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# ------------------ INITIALIZE MODULES ------------------

# Initialize Mini-RAG and build FAISS index from local documents
rag = MiniRAG()
rag.build_index()

# Initialize Image captioning model (BLIP)
vision = ImageDescriber()


# ------------------ BOT HELP TEXT ------------------

HELP_TEXT = """
🤖 GenAI Hybrid Telegram Bot

Commands:
/help - show usage instructions
/ask <query> - ask from knowledge base (Mini-RAG)
/image - upload an image and I will describe it
/summarize - summarize last response (RAG or image)

Examples:
/ask What is the WFH policy?
/image
/summarize
"""


# ------------------ USER STATE MANAGEMENT ------------------

# Tracks current mode per user
# Example: {12345: "image"} means user is about to upload an image
user_mode = {}  # {user_id: "image" or None}

# Stores last 3 interactions per user for lightweight context awareness
# Example: {12345: [{"type": "rag", "user": "...", "bot": "..."}]}
user_history = {}

# Stores the last bot output per user (used for /summarize command)
# Example: {12345: {"type": "rag", "content": "..."}}
last_output = {}


def add_history(user_id, interaction_type, user_text, bot_text):
    """
    Stores user interaction history (only last 3 interactions).
    This helps make the bot slightly context-aware.
    """
    if user_id not in user_history:
        user_history[user_id] = []

    # Add current interaction
    user_history[user_id].append({
        "type": interaction_type,
        "user": user_text,
        "bot": bot_text
    })

    # Keep only last 3 interactions
    user_history[user_id] = user_history[user_id][-3:]


def build_history_context(user_id):
    """
    Builds a small text context from the last 3 interactions.
    This is a lightweight way to show 'memory' without using an LLM.
    """
    if user_id not in user_history or len(user_history[user_id]) == 0:
        return ""

    lines = []
    for i, h in enumerate(user_history[user_id], 1):
        lines.append(f"{i}. User: {h['user']}")
        lines.append(f"   Bot: {h['bot']}")

    return "\n".join(lines)


# ------------------ COMMAND HANDLERS ------------------

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles /help command and shows usage instructions."""
    await update.message.reply_text(HELP_TEXT)


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /ask <query> command.
    Retrieves top relevant chunks from the knowledge base using Mini-RAG.
    """
    user_id = update.message.from_user.id

    # Extract user query from command text
    query = update.message.text.replace("/ask", "").strip()

    # Validate query
    if not query:
        await update.message.reply_text("❌ Please use: /ask <your question>")
        return

    # Retrieve top chunks (with source metadata)
    results = rag.retrieve(query, top_k=3)

    # Create small history context (optional feature)
    history_context = build_history_context(user_id)

    # Build answer response
    answer = "📌 Answer from knowledge base:\n\n"
    for i, r in enumerate(results, 1):
        answer += f"{i}) {r['chunk']}\n\n"

    # Add sources for transparency
    answer += "📚 Sources used:\n"
    for r in results:
        snippet = r["chunk"][:120].replace("\n", " ") + "..."
        answer += f"- {r['source']} → \"{snippet}\"\n"

    # If user has previous interactions, show note
    if history_context:
        answer += "\n🧠 (History-aware mode enabled: last 3 interactions considered)"

    # Send response back to Telegram user
    await update.message.reply_text(answer)

    # Store output for /summarize command
    last_output[user_id] = {"type": "rag", "content": answer}

    # Store in interaction history
    add_history(user_id, "rag", query, answer)


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /image command.
    Sets user mode so the next photo upload is processed.
    """
    user_id = update.message.from_user.id

    # Set mode to image so bot expects a photo next
    user_mode[user_id] = "image"

    await update.message.reply_text(
        "🖼️ Great! Now upload an image and I will describe it."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles photo uploads.
    Only processes image if user previously typed /image.
    """
    user_id = update.message.from_user.id

    # Validate user state: must use /image first
    if user_mode.get(user_id) != "image":
        await update.message.reply_text(
            "⚠️ Please type /image first, then upload the image."
        )
        return

    # Get highest quality photo from Telegram (last element is biggest)
    photo = update.message.photo[-1]
    file = await photo.get_file()

    # Create temp folder to store downloaded images
    os.makedirs("temp_images", exist_ok=True)
    image_path = f"temp_images/{user_id}.jpg"

    # Download image locally
    await file.download_to_drive(image_path)

    await update.message.reply_text("⏳ Processing your image...")

    try:
        # Generate caption + tags using BLIP
        caption, tags = vision.describe(image_path)

        # Prepare response message
        reply = f"📝 Caption:\n{caption}\n\n🏷️ Tags:\n- " + "\n- ".join(tags)

        await update.message.reply_text(reply)

        # Store output for /summarize
        last_output[user_id] = {"type": "image", "content": reply}

        # Store in history
        add_history(user_id, "image", "Uploaded an image", reply)

    except Exception as e:
        # Handle errors gracefully
        await update.message.reply_text(f"❌ Error while describing image: {e}")

    # Reset user mode after processing
    user_mode[user_id] = None


async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /summarize command.
    Summarizes the last response using a lightweight rule-based approach.
    """
    user_id = update.message.from_user.id

    # Check if there is anything to summarize
    if user_id not in last_output:
        await update.message.reply_text(
            "❌ Nothing to summarize yet. Ask something or upload an image first."
        )
        return

    content = last_output[user_id]["content"]

    # Lightweight summarization (no external LLM used)
    lines = content.split("\n")
    important = []

    for line in lines:
        line = line.strip()

        # Skip header lines
        if line.startswith("📝 Caption:") or line.startswith("📌 Answer"):
            continue

        # Pick only a few meaningful lines
        if len(line) > 0 and len(important) < 6:
            important.append(line)

    # Build summary response
    summary = "✅ Summary:\n" + "\n".join(important[:6])

    await update.message.reply_text(summary)

    # Store summary in history
    add_history(user_id, "summarize", "/summarize", summary)


# ------------------ MAIN ENTRY POINT ------------------

def main():
    """
    Creates Telegram bot application, registers handlers,
    and starts polling for messages.
    """
    # Create Telegram application using token
    app = Application.builder().token(TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(CommandHandler("summarize", summarize_command))

    # Register photo handler
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("✅ Hybrid bot running...")
    app.run_polling()


# Run program
if __name__ == "__main__":
    main()
