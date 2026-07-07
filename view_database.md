# Voice Agent Database Table View

This file is a human-readable snapshot of the SQLite tables. Open it in markdown preview mode in VS Code to see clean tables.

## Table: shipment_tracking

| tracking_id | status | current_location | eta_days | pickup_address | destination_address | package_weight | pickup_time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SH123 | In Transit | Hyderabad | 1 | Hitech City Office | Delhi Hub | 1.5 kg | Completed |
| SH456 | Out for Delivery | Ghatkesar | 0 | Secunderabad Depot | Ghatkesar Residential | 0.8 kg | Completed |
| SH789 | Delayed | Delhi Hub | 4 | Mumbai Port | Gachibowli Tech Park | 12.0 kg | Completed |

## Table: conversation_logs (Selected Columns)

| id | session_id | timestamp | user_transcript | ai_response | tracking_id_ref |
| --- | --- | --- | --- | --- | --- |
| 1 | default_session | 2026-06-30 09:08:30 |  hi I would like to know where my shipment SH123 is right now | Your shipment SH123 is currently in transit, located in Hyderabad, with an estimated time of arrival (ETA) in 1 day. | SH123 |
| 2 | default_session | 2026-06-30 09:09:57 |  Hi, I want you to tell me a joke. | I’m here to assist with logistics inquiries. How can I help you with your shipment or delivery needs? |  |
| 3 | default_session | 2026-06-30 09:10:18 | నమస్కారం, నా ఆర్డర్ ఎక్కడ ఉంది? | మీ ఆర్డర్ స్థితి తెలుసుకోవడానికి, దయచేసి ట్రాకింగ్ ID అందించండి. |  |

