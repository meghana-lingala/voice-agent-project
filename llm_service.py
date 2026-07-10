import os
import re
import traceback
from openai import OpenAI
from dotenv import load_dotenv
import database

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize client. We check if key is present to prevent exceptions at import time.
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SYSTEM_PROMPT = (
    "You are a professional logistics support assistant helping users with shipment inquiries. "
    "Provide clear, short, and professional responses. Do not provide information unrelated to logistics. "
    "Follow these strict behavioral pathways matching these scenarios:\n\n"
    "Scenario 1 (Shipment Tracking):\n"
    "- If shipment data is found, state the status, location, and ETA concisely.\n"
    "- If the tracking ID was not found or is invalid, you MUST say exactly: "
    "\"I couldn't find a shipment matching that ID. Please double-check the number and try again.\"\n\n"
    "Scenario 2 (Placing an Order / Pickup Scheduling):\n"
    "- If the user wants to book a courier or place an order, audit the conversation history for 4 mandatory metrics: "
    "Pickup Address, Destination Address, Package Weight, and Target Time Slot.\n"
    "- If any details are missing, politely prompt the user to provide them (e.g., \"I can help with that! What is the pickup address?\"). "
    "Only ask for one missing metric at a time.\n"
    "- Once you have successfully gathered all 4 mandatory metrics, you MUST call the `create_new_shipment_record` tool.\n\n"
    "Scenario 3 (Delivery Delays):\n"
    "- Handle delayed package context with an empathetic, professional logistics delay apology.\n\n"
    "Scenario 4 (General Logistics FAQs):\n"
    "- Provide standard clean timeline windows for normal parcel operations: "
    "Standard shipping (3-5 business days), Express shipping (1-2 business days), Same-Day shipping.\n\n"
    "Scenario 5 (Unknown/Ambiguous Requests / Session Completion):\n"
    "- If the user says they are finished, says goodbye, or has no more questions, you MUST call the `end_current_session` tool instantly.\n"
    "- Do NOT call `end_current_session` if the user is simply acknowledging or thanking you (e.g. saying 'Thank you' or 'Okay') in response to a question you asked (such as asking for a tracking ID or address details). Keep the session active and wait for the requested details.\n"
    "- If the query is ambiguous or unrelated to supply chain management, logistics, tracking shipments, scheduling order pickups, or canceling orders, "
    "you MUST politely deflect using this standardized response exactly: "
    "\"I'm sorry, but as the LogiRoute assistant, I can only help you with shipping, tracking, and courier operations. How can I assist with your logistics needs today?\"\n\n"
    "Scenario 6 (Cancel Order Request):\n"
    "- If the user wants to cancel their shipment/order, you must verify if a tracking ID is present in the context or has been provided.\n"
    "- If the tracking ID is missing, politely prompt the user to provide the tracking ID (e.g., \"I can help you cancel that order. Could you please provide the tracking ID?\").\n"
    "- Once the tracking ID is provided, you MUST call the `cancel_shipment_order` tool instantly.\n\n"
    "Language Policy:\n"
    "- You MUST respond in the exact same language as the user's input/query (e.g. English, Telugu, Hindi). If the user speaks/writes in Telugu, you MUST reply in Telugu. If they speak/write in Hindi, you MUST reply in Hindi. If they speak/write in English, you MUST reply in English."
)

# Define the tools available for the assistant
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "end_current_session",
            "description": "Call this function when the customer explicitly states they are finished, says goodbye, or has no more questions.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_new_shipment_record",
            "description": "Creates a new shipment record in the system when all 4 mandatory parameters are collected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pickup_address": {
                        "type": "string",
                        "description": "The address where the package will be picked up."
                    },
                    "destination_address": {
                        "type": "string",
                        "description": "The destination address where the package will be delivered."
                    },
                    "package_weight": {
                        "type": "string",
                        "description": "The weight of the package (e.g., '1.5 kg')."
                    },
                    "pickup_time": {
                        "type": "string",
                        "description": "The scheduled time slot or status for pickup (e.g., 'Tomorrow at 2 PM')."
                    }
                },
                "required": ["pickup_address", "destination_address", "package_weight", "pickup_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_shipment_order",
            "description": "Cancels an existing shipment order if it has not yet been picked up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_id": {
                        "type": "string",
                        "description": "The tracking ID of the shipment to cancel (e.g., 'SH123')."
                    }
                },
                "required": ["tracking_id"]
            }
        }
    }
]

def sanitize_response(response: str) -> str:
    """
    Filters and sanitizes the generated response string.
    """
    if not response or not response.strip():
        return "I'm unable to understand your request. Could you please provide more details?"

    # Split response into sentences using a regex that checks for sentence-ending punctuation (.!?)
    # followed by whitespace and a capital letter, digit, or end of string, avoiding decimal splits.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', response.strip())
    
    # Filter empty parts
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return "I'm unable to understand your request. Could you please provide more details?"

    # Voice Readability: Limit to maximum of 3 sentences
    if len(sentences) > 3:
        response = " ".join(sentences[:3])
    else:
        response = " ".join(sentences)

    return response

def generate_response(user_text: str, session_id: str, context_data: dict = None, mode: str = "general") -> dict:
    """
    Generates a response from the OpenAI gpt-4o-mini model, supporting tool calls.
    Returns a dict with 'ai_response' and 'tool_calls' keys.
    """
    if not client:
        return {
            "ai_response": "System error: OpenAI client is not initialized. Please configure OPENAI_API_KEY.",
            "tool_calls": None
        }

    # Retrieve conversation history from database
    history = []
    try:
        history = database.get_conversation_history(session_id)
    except Exception:
        # Fallback gracefully if database history retrieval fails
        pass

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add historical messages (limit to last 10 turns)
    for msg in history:
        messages.append(msg)

    # Incorporate tracking context
    if mode == "tracking":
        if context_data:
            context_str = (
                f"Shipment details found for tracking ID '{context_data.get('tracking_id')}':\n"
                f"Status: {context_data.get('status')}\n"
                f"Current Location: {context_data.get('current_location')}\n"
                f"ETA (Days): {context_data.get('eta_days')}\n"
                f"Pickup Address: {context_data.get('pickup_address')}\n"
                f"Destination Address: {context_data.get('destination_address')}\n"
                f"Package Weight: {context_data.get('package_weight')}\n"
                f"Pickup Time: {context_data.get('pickup_time')}\n"
            )
            messages.append({
                "role": "system",
                "content": f"Use the following shipment data to answer the user's inquiry concisely:\n{context_str}"
            })
        else:
            # Strictly return the requested string when tracking ID not found
            return {
                "ai_response": "I couldn't find a shipment matching that ID. Please double-check the number and try again.",
                "tool_calls": None
            }

    # Add the current user query
    messages.append({"role": "user", "content": user_text})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=150
        )
        message = completion.choices[0].message
        raw_res = message.content or ""
        tool_calls = message.tool_calls

        sanitized_res = sanitize_response(raw_res) if raw_res.strip() else ""
        return {
            "ai_response": sanitized_res,
            "tool_calls": tool_calls
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "ai_response": "I'm experiencing a brief connection delay. Please give me a moment and try repeating your request.",
            "tool_calls": None
        }
