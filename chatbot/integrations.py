import logging
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import hashlib
import hmac

logger = logging.getLogger(__name__)

class WhatsAppService:
    """Service for WhatsApp Business API integration"""
    
    def __init__(self):
        self.api_url = os.getenv("WHATSAPP_API_URL", "https://graph.facebook.com/v18.0")
        self.access_token = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
        self.phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
        self.verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
        
        # Message templates for different types of responses
        self.message_templates = {
            "greeting": "Hello! I'm your health assistant. How can I help you today?",
            "disease_info": "Here's information about {disease_name}:\n\n{symptoms}\n\nPrevention: {prevention}",
            "vaccination": "Vaccination schedule for {age_group}:\n\n{schedule}",
            "emergency": "🚨 EMERGENCY ALERT 🚨\n\nFor immediate medical assistance:\n• Call 108 (Emergency Ambulance)\n• Contact nearest hospital\n• Visit emergency room immediately",
            "fallback": "I understand you have a health-related question. Could you please rephrase your question or be more specific?"
        }
    
    async def process_webhook(self, data: Dict) -> Dict:
        """Process incoming WhatsApp webhook data"""
        try:
            # Verify webhook signature if provided
            if not await self._verify_webhook_signature(data):
                return {"status": "error", "message": "Invalid signature"}
            
            # Extract message data
            if "entry" in data:
                for entry in data["entry"]:
                    if "changes" in entry:
                        for change in entry["changes"]:
                            if change.get("field") == "messages":
                                await self._process_message_change(change["value"])
            
            return {"status": "success", "message": "Webhook processed"}
            
        except Exception as e:
            logger.error(f"Error processing WhatsApp webhook: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _verify_webhook_signature(self, data: Dict) -> bool:
        """Verify webhook signature for security"""
        try:
            # In production, implement proper signature verification
            # This is a simplified version
            return True
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {str(e)}")
            return False
    
    async def _process_message_change(self, value: Dict):
        """Process message change from webhook"""
        try:
            if "messages" in value:
                for message in value["messages"]:
                    await self._handle_incoming_message(message)
            
            if "statuses" in value:
                for status in value["statuses"]:
                    await self._handle_message_status(status)
                    
        except Exception as e:
            logger.error(f"Error processing message change: {str(e)}")
    
    async def _handle_incoming_message(self, message: Dict):
        """Handle incoming WhatsApp message"""
        try:
            # Extract message details
            message_id = message.get("id")
            from_number = message.get("from")
            timestamp = message.get("timestamp")
            
            # Extract message content
            message_content = None
            message_type = None
            
            if "text" in message:
                message_content = message["text"]["body"]
                message_type = "text"
            elif "image" in message:
                message_content = message["image"]["caption"] if "caption" in message["image"] else ""
                message_type = "image"
            elif "document" in message:
                message_content = message["document"]["caption"] if "caption" in message["document"] else ""
                message_type = "document"
            
            if not message_content:
                await self._send_message(from_number, "I can only process text messages at the moment. Please send your health question as text.")
                return
            
            # Process the message through chatbot
            from chatbot.core import HealthChatbot
            from chatbot.database import DatabaseManager
            from chatbot.translation import TranslationService
            from chatbot.health_data import HealthDataService
            
            # Initialize services (in production, these would be injected)
            db_manager = DatabaseManager()
            translation_service = TranslationService()
            health_data_service = HealthDataService()
            chatbot = HealthChatbot(db_manager, translation_service, health_data_service)
            
            # Detect language
            detected_language = await translation_service.detect_language(message_content)
            
            # Process message
            response = await chatbot.process_message(
                message=message_content,
                user_id=from_number,
                language=detected_language,
                platform="whatsapp"
            )
            
            # Send response
            await self._send_message(from_number, response["response"])
            
            # Log the conversation
            await db_manager.log_conversation(
                user_id=from_number,
                message=message_content,
                intent=response.get("intent", "unknown"),
                confidence=response.get("confidence", 0.0),
                entities=response.get("entities", []),
                platform="whatsapp",
                language=detected_language,
                response=response["response"]
            )
            
        except Exception as e:
            logger.error(f"Error handling incoming message: {str(e)}")
            await self._send_message(from_number, "I apologize, but I'm having trouble processing your message. Please try again.")
    
    async def _handle_message_status(self, status: Dict):
        """Handle message delivery status updates"""
        try:
            message_id = status.get("id")
            status_type = status.get("status")
            timestamp = status.get("timestamp")
            
            logger.info(f"Message {message_id} status: {status_type}")
            
            # Update message status in database if needed
            # This could be used for analytics and delivery confirmation
            
        except Exception as e:
            logger.error(f"Error handling message status: {str(e)}")
    
    async def _send_message(self, to_number: str, message_text: str, message_type: str = "text") -> bool:
        """Send message via WhatsApp Business API"""
        try:
            if not self.access_token or not self.phone_number_id:
                logger.error("WhatsApp credentials not configured")
                return False
            
            url = f"{self.api_url}/{self.phone_number_id}/messages"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": message_type,
                "text": {
                    "body": message_text
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Message sent successfully to {to_number}")
                return True
            else:
                logger.error(f"Failed to send message: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {str(e)}")
            return False
    
    async def send_template_message(self, to_number: str, template_name: str, parameters: List[str] = None) -> bool:
        """Send template message via WhatsApp Business API"""
        try:
            if not self.access_token or not self.phone_number_id:
                logger.error("WhatsApp credentials not configured")
                return False
            
            url = f"{self.api_url}/{self.phone_number_id}/messages"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": "template",
                "template": {
                    "name": template_name,
                    "language": {
                        "code": "en"
                    }
                }
            }
            
            if parameters:
                payload["template"]["components"] = [
                    {
                        "type": "body",
                        "parameters": [{"type": "text", "text": param} for param in parameters]
                    }
                ]
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Template message sent successfully to {to_number}")
                return True
            else:
                logger.error(f"Failed to send template message: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending WhatsApp template message: {str(e)}")
            return False
    
    async def send_interactive_message(self, to_number: str, header_text: str, body_text: str, 
                                     footer_text: str, buttons: List[Dict]) -> bool:
        """Send interactive message with buttons"""
        try:
            if not self.access_token or not self.phone_number_id:
                logger.error("WhatsApp credentials not configured")
                return False
            
            url = f"{self.api_url}/{self.phone_number_id}/messages"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "header": {
                        "type": "text",
                        "text": header_text
                    },
                    "body": {
                        "text": body_text
                    },
                    "footer": {
                        "text": footer_text
                    },
                    "action": {
                        "buttons": buttons
                    }
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Interactive message sent successfully to {to_number}")
                return True
            else:
                logger.error(f"Failed to send interactive message: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending WhatsApp interactive message: {str(e)}")
            return False
    
    async def send_list_message(self, to_number: str, header_text: str, body_text: str, 
                               footer_text: str, button_text: str, sections: List[Dict]) -> bool:
        """Send list message with options"""
        try:
            if not self.access_token or not self.phone_number_id:
                logger.error("WhatsApp credentials not configured")
                return False
            
            url = f"{self.api_url}/{self.phone_number_id}/messages"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": "interactive",
                "interactive": {
                    "type": "list",
                    "header": {
                        "type": "text",
                        "text": header_text
                    },
                    "body": {
                        "text": body_text
                    },
                    "footer": {
                        "text": footer_text
                    },
                    "action": {
                        "button": button_text,
                        "sections": sections
                    }
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"List message sent successfully to {to_number}")
                return True
            else:
                logger.error(f"Failed to send list message: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending WhatsApp list message: {str(e)}")
            return False
    
    async def send_health_tips_broadcast(self, phone_numbers: List[str], tips: str) -> Dict:
        """Send health tips to multiple users"""
        results = {
            "successful": [],
            "failed": []
        }
        
        for phone_number in phone_numbers:
            try:
                success = await self._send_message(phone_number, tips)
                if success:
                    results["successful"].append(phone_number)
                else:
                    results["failed"].append(phone_number)
            except Exception as e:
                logger.error(f"Error sending broadcast to {phone_number}: {str(e)}")
                results["failed"].append(phone_number)
        
        return results
    
    async def send_outbreak_alert(self, phone_numbers: List[str], alert_data: Dict) -> Dict:
        """Send outbreak alert to multiple users"""
        alert_message = f"🚨 OUTBREAK ALERT 🚨\n\n"
        alert_message += f"Disease: {alert_data.get('disease_name', 'Unknown')}\n"
        alert_message += f"Location: {alert_data.get('location', 'Unknown')}\n"
        alert_message += f"Severity: {alert_data.get('severity', 'Unknown')}\n"
        alert_message += f"Cases: {alert_data.get('cases_count', 'Unknown')}\n\n"
        alert_message += f"Prevention: {alert_data.get('prevention_measures', 'Follow health guidelines')}\n\n"
        alert_message += "Stay safe and follow preventive measures!"
        
        return await self.send_health_tips_broadcast(phone_numbers, alert_message)
    
    async def send_vaccination_reminder(self, phone_number: str, vaccine_name: str, 
                                     due_date: str, location: str) -> bool:
        """Send vaccination reminder to user"""
        reminder_message = f"💉 VACCINATION REMINDER 💉\n\n"
        reminder_message += f"Vaccine: {vaccine_name}\n"
        reminder_message += f"Due Date: {due_date}\n"
        reminder_message += f"Location: {location}\n\n"
        reminder_message += "Please schedule your vaccination appointment.\n"
        reminder_message += "Contact your nearest health center for more information."
        
        return await self._send_message(phone_number, reminder_message)
