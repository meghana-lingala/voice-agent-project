import unittest
import os
import sqlite3
from unittest.mock import patch
import database

class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.test_db = "test_voice_agent.db"
        # Patch the database file path to our test db
        self.db_patcher = patch('database.DB_FILE', self.test_db)
        self.db_patcher.start()
        
        # Ensure clean state
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
        # Initialize test database
        database.init_db()

    def tearDown(self):
        self.db_patcher.stop()
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except PermissionError:
                pass

    def test_init_db_creates_tables_and_mock_data(self):
        # Establish connection to verify tables exist
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        # Check shipment_tracking table
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='shipment_tracking'")
        self.assertEqual(cursor.fetchone()[0], 1)
        
        # Check conversation_logs table
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='conversation_logs'")
        self.assertEqual(cursor.fetchone()[0], 1)
        
        # Check mock data was loaded
        cursor.execute("SELECT count(*) FROM shipment_tracking")
        self.assertEqual(cursor.fetchone()[0], 3)
        
        # Verify specific profiles
        cursor.execute("SELECT status, current_location FROM shipment_tracking WHERE tracking_id='SH123'")
        row = cursor.fetchone()
        self.assertEqual(row[0], 'In Transit')
        self.assertEqual(row[1], 'Hyderabad')
        
        conn.close()

    def test_get_shipment_details_exists(self):
        details = database.get_shipment_details("sh123")  # lowercase should work
        self.assertIsNotNone(details)
        self.assertEqual(details["tracking_id"], "SH123")
        self.assertEqual(details["status"], "In Transit")
        self.assertEqual(details["current_location"], "Hyderabad")

    def test_get_shipment_details_not_exists(self):
        details = database.get_shipment_details("SH999")
        self.assertIsNone(details)

    def test_insert_new_order_success(self):
        # Insert a new order
        success = database.insert_new_order(
            tracking_id="SH111",
            p_addr="Bangalore Office",
            d_addr="Chennai Hub",
            weight="2.0 kg",
            p_time="Pending"
        )
        self.assertTrue(success)
        
        # Retrieve and verify details
        details = database.get_shipment_details("SH111")
        self.assertIsNotNone(details)
        self.assertEqual(details["status"], "Created")
        self.assertEqual(details["current_location"], "Origin Depot")
        self.assertEqual(details["pickup_address"], "Bangalore Office")
        self.assertEqual(details["destination_address"], "Chennai Hub")
        self.assertEqual(details["package_weight"], "2.0 kg")
        self.assertEqual(details["pickup_time"], "Pending")

    def test_insert_new_order_duplicate_fails(self):
        # Insert duplicate order (SH123 already exists from mock data)
        success = database.insert_new_order(
            tracking_id="SH123",
            p_addr="Some Office",
            d_addr="Some Hub",
            weight="3.0 kg",
            p_time="Pending"
        )
        self.assertFalse(success)

    def test_log_interaction(self):
        database.log_interaction(
            session_id="test_session_123",
            audio_file="speech.wav",
            transcript="Where is my shipment SH123?",
            response="Your shipment is in Transit in Hyderabad.",
            lang="en",
            engine="openai/gpt-4o-mini",
            latency=500,
            tracking_id="SH123"
        )
        
        # Verify conversation log entry
        conn = sqlite3.connect(self.test_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversation_logs WHERE session_id='test_session_123'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        
        self.assertEqual(row["audio_filename"], "speech.wav")
        self.assertEqual(row["user_transcript"], "Where is my shipment SH123?")
        self.assertEqual(row["ai_response"], "Your shipment is in Transit in Hyderabad.")
        self.assertEqual(row["detected_language"], "en")
        self.assertEqual(row["engine_used"], "openai/gpt-4o-mini")
        self.assertEqual(row["api_latency_ms"], 500)
        self.assertEqual(row["tracking_id_ref"], "SH123")
        
        conn.close()

if __name__ == '__main__':
    unittest.main()
