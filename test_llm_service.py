import unittest
from unittest.mock import patch, MagicMock
import llm_service

class TestLLMService(unittest.TestCase):

    @patch('llm_service.client')
    @patch('database.get_conversation_history', return_value=[])
    def test_generate_response_general(self, mock_history, mock_client):
        # Setup mock OpenAI completion response
        mock_completion = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Hello! I am your logistics assistant. How can I help you today?"
        mock_message.tool_calls = None
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_completion
        
        response = llm_service.generate_response(
            user_text="Hi, what is this service?",
            session_id="session_1",
            mode="general"
        )
        
        self.assertEqual(response["ai_response"], "Hello! I am your logistics assistant. How can I help you today?")
        self.assertIsNone(response["tool_calls"])
        
        # Verify the OpenAI api call params
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        
        # Verify system prompt and user text are passed
        messages = kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("logistics support assistant", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Hi, what is this service?")

    @patch('llm_service.client')
    @patch('database.get_conversation_history', return_value=[])
    def test_generate_response_tracking_found(self, mock_history, mock_client):
        mock_completion = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Your shipment SH123 is currently In Transit at Hyderabad. It is expected to arrive in 1 day."
        mock_message.tool_calls = None
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_completion

        context_data = {
            "tracking_id": "SH123",
            "status": "In Transit",
            "current_location": "Hyderabad",
            "eta_days": 1,
            "pickup_address": "Hitech City Office",
            "destination_address": "Delhi Hub",
            "package_weight": "1.5 kg",
            "pickup_time": "Completed"
        }

        response = llm_service.generate_response(
            user_text="Where is my shipment SH123?",
            session_id="session_2",
            context_data=context_data,
            mode="tracking"
        )

        self.assertEqual(response["ai_response"], "Your shipment SH123 is currently In Transit at Hyderabad. It is expected to arrive in 1 day.")
        self.assertIsNone(response["tool_calls"])
        
        mock_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_client.chat.completions.create.call_args
        messages = kwargs["messages"]
        
        # Check system prompt with details
        self.assertEqual(messages[1]["role"], "system")
        self.assertIn("Shipment details found for tracking ID 'SH123'", messages[1]["content"])
        self.assertIn("Status: In Transit", messages[1]["content"])

    def test_generate_response_tracking_not_found(self):
        # Tracking ID not found should immediately return the exact error message (without calling OpenAI client)
        response = llm_service.generate_response(
            user_text="Track my order SH999",
            session_id="session_3",
            context_data=None,
            mode="tracking"
        )

        self.assertEqual(response["ai_response"], "I couldn't find a shipment matching that ID. Please double-check the number and try again.")
        self.assertIsNone(response["tool_calls"])

    @patch('llm_service.client')
    @patch('database.get_conversation_history', return_value=[])
    def test_generate_response_error_handling(self, mock_history, mock_client):
        # Setup API error exception
        mock_client.chat.completions.create.side_effect = Exception("API rate limit reached")
        
        response = llm_service.generate_response(
            user_text="Where is my shipment?",
            session_id="session_error",
            mode="general"
        )
        
        self.assertEqual(response["ai_response"], "I'm experiencing a brief connection delay. Please give me a moment and try repeating your request.")
        self.assertIsNone(response["tool_calls"])

    @patch('llm_service.client')
    @patch('database.get_conversation_history', return_value=[])
    def test_generate_response_out_of_scope(self, mock_history, mock_client):
        mock_completion = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "I'm sorry, but as the LogiRoute assistant, I can only help you with shipping, tracking, and courier operations. How can I assist with your logistics needs today?"
        mock_message.tool_calls = None
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_completion
        
        response = llm_service.generate_response(
            user_text="Can you write a Python script to sort a list?",
            session_id="session_oos",
            mode="general"
        )
        
        self.assertEqual(
            response["ai_response"],
            "I'm sorry, but as the LogiRoute assistant, I can only help you with shipping, tracking, and courier operations. How can I assist with your logistics needs today?"
        )
        self.assertIsNone(response["tool_calls"])

    @patch('llm_service.client')
    @patch('database.get_conversation_history', return_value=[])
    def test_generate_response_tool_calling_reset(self, mock_history, mock_client):
        # Setup mock for tool call
        mock_completion = MagicMock()
        mock_message = MagicMock()
        mock_message.content = ""
        
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "end_current_session"
        mock_tool_call.function.arguments = "{}"
        
        mock_message.tool_calls = [mock_tool_call]
        mock_completion.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_completion
        
        response = llm_service.generate_response(
            user_text="goodbye",
            session_id="session_tool_1",
            mode="general"
        )
        
        self.assertEqual(response["ai_response"], "")
        self.assertEqual(len(response["tool_calls"]), 1)
        self.assertEqual(response["tool_calls"][0].function.name, "end_current_session")

    def test_sanitize_response_truncation(self):
        # Over 3 sentences should be truncated
        long_response = (
            "Your shipment is arriving tomorrow. The delivery agent is Bob. "
            "Please be available to receive it. We hope you liked our service."
        )
        sanitized = llm_service.sanitize_response(long_response)
        self.assertEqual(
            sanitized,
            "Your shipment is arriving tomorrow. The delivery agent is Bob. Please be available to receive it."
        )

    def test_sanitize_response_decimal_safety(self):
        # Period in decimal number (1.5) should not trigger sentence split
        decimal_response = "The package weight is 1.5 kg. It is currently in Transit. Thank you."
        sanitized = llm_service.sanitize_response(decimal_response)
        self.assertEqual(
            sanitized,
            "The package weight is 1.5 kg. It is currently in Transit. Thank you."
        )

    def test_sanitize_response_empty_guardrail(self):
        # Empty/blank responses should fall back to guardrail string
        self.assertEqual(
            llm_service.sanitize_response(""),
            "I'm unable to understand your request. Could you please provide more details?"
        )
        self.assertEqual(
            llm_service.sanitize_response("   \n   "),
            "I'm unable to understand your request. Could you please provide more details?"
        )

if __name__ == '__main__':
    unittest.main()
