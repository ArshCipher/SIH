import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioException

logger = logging.getLogger(__name__)

class SMSService:
    """Service for SMS integration using Twilio"""
    
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.phone_number = os.getenv("TWILIO_PHONE_NUMBER", "")
        
        # Initialize Twilio client
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
        else:
            self.client = None
            logger.warning("Twilio credentials not configured")
    
    async def process_webhook(self, data: Dict) -> Dict:
        """Process incoming SMS webhook data"""
        try:
            # Extract message data from Twilio webhook
            message_body = data.get("Body", "")
            from_number = data.get("From", "")
            message_sid = data.get("MessageSid", "")
            
            if not message_body or not from_number:
                return {"status": "error", "message": "Invalid webhook data"}
            
            # Process the message through chatbot
            from chatbot.core import HealthChatbot
            from chatbot.database import DatabaseManager
            from chatbot.translation import TranslationService
            from chatbot.health_data import HealthDataService
            
            # Initialize services
            db_manager = DatabaseManager()
            translation_service = TranslationService()
            health_data_service = HealthDataService()
            chatbot = HealthChatbot(db_manager, translation_service, health_data_service)
            
            # Detect language
            detected_language = await translation_service.detect_language(message_body)
            
            # Process message
            response = await chatbot.process_message(
                message=message_body,
                user_id=from_number,
                language=detected_language,
                platform="sms"
            )
            
            # Send response via SMS
            await self.send_sms(from_number, response["response"])
            
            # Log the conversation
            await db_manager.log_conversation(
                user_id=from_number,
                message=message_body,
                intent=response.get("intent", "unknown"),
                confidence=response.get("confidence", 0.0),
                entities=response.get("entities", []),
                platform="sms",
                language=detected_language,
                response=response["response"]
            )
            
            return {"status": "success", "message": "SMS processed"}
            
        except Exception as e:
            logger.error(f"Error processing SMS webhook: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def send_sms(self, to_number: str, message_text: str) -> bool:
        """Send SMS via Twilio"""
        try:
            if not self.client:
                logger.error("Twilio client not initialized")
                return False
            
            # Truncate message if too long (SMS limit is 160 characters)
            if len(message_text) > 160:
                message_text = message_text[:157] + "..."
            
            message = self.client.messages.create(
                body=message_text,
                from_=self.phone_number,
                to=to_number
            )
            
            logger.info(f"SMS sent successfully to {to_number}, SID: {message.sid}")
            return True
            
        except TwilioException as e:
            logger.error(f"Twilio error sending SMS: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error sending SMS: {str(e)}")
            return False
    
    async def send_bulk_sms(self, phone_numbers: List[str], message_text: str) -> Dict:
        """Send SMS to multiple recipients"""
        results = {
            "successful": [],
            "failed": []
        }
        
        for phone_number in phone_numbers:
            try:
                success = await self.send_sms(phone_number, message_text)
                if success:
                    results["successful"].append(phone_number)
                else:
                    results["failed"].append(phone_number)
            except Exception as e:
                logger.error(f"Error sending bulk SMS to {phone_number}: {str(e)}")
                results["failed"].append(phone_number)
        
        return results
    
    async def send_health_tips_sms(self, phone_numbers: List[str], tips: str) -> Dict:
        """Send health tips via SMS to multiple users"""
        return await self.send_bulk_sms(phone_numbers, f"Health Tip: {tips}")
    
    async def send_outbreak_alert_sms(self, phone_numbers: List[str], alert_data: Dict) -> Dict:
        """Send outbreak alert via SMS"""
        alert_message = f"ALERT: {alert_data.get('disease_name', 'Unknown')} outbreak in {alert_data.get('location', 'Unknown')}. "
        alert_message += f"Severity: {alert_data.get('severity', 'Unknown')}. "
        alert_message += f"Prevention: {alert_data.get('prevention_measures', 'Follow health guidelines')}"
        
        return await self.send_bulk_sms(phone_numbers, alert_message)
    
    async def send_vaccination_reminder_sms(self, phone_number: str, vaccine_name: str, 
                                         due_date: str, location: str) -> bool:
        """Send vaccination reminder via SMS"""
        reminder_message = f"REMINDER: {vaccine_name} vaccination due on {due_date} at {location}. "
        reminder_message += "Contact your nearest health center to schedule."
        
        return await self.send_sms(phone_number, reminder_message)


class AlertService:
    """Service for managing outbreak alerts and notifications"""
    
    def __init__(self):
        self.outbreak_check_interval = int(os.getenv("OUTBREAK_CHECK_INTERVAL", "3600"))
        self.vaccination_reminder_interval = int(os.getenv("VACCINATION_REMINDER_INTERVAL", "86400"))
        
        # External health API endpoints (optional - will use fallback if not available)
        self.health_api_base_url = os.getenv("HEALTH_API_BASE_URL", "")
        self.health_api_key = os.getenv("HEALTH_API_KEY", "")
    
    async def get_outbreak_alerts(self, location: str = None) -> List[Dict]:
        """Get current outbreak alerts"""
        try:
            # Try to fetch from external API first
            api_alerts = await self._fetch_outbreak_alerts_from_api(location)
            if api_alerts:
                return api_alerts
            
            # Fallback to sample data
            sample_alerts = [
                {
                    "disease_name": "Dengue",
                    "location": "Mumbai",
                    "severity": "moderate",
                    "cases_count": 45,
                    "alert_level": "yellow",
                    "prevention_measures": "Eliminate mosquito breeding sites, use repellent, wear protective clothing",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "disease_name": "Malaria",
                    "location": "Delhi",
                    "severity": "high",
                    "cases_count": 23,
                    "alert_level": "orange",
                    "prevention_measures": "Use mosquito nets, insect repellent, avoid stagnant water",
                    "created_at": datetime.now().isoformat()
                }
            ]
            
            if location:
                sample_alerts = [alert for alert in sample_alerts if location.lower() in alert["location"].lower()]
            
            return sample_alerts
            
        except Exception as e:
            logger.error(f"Error getting outbreak alerts: {str(e)}")
            return []
    
    async def _fetch_outbreak_alerts_from_api(self, location: str = None) -> Optional[List[Dict]]:
        """Fetch outbreak alerts from external health API (optional)"""
        try:
            # Only try API if both URL and key are provided
            if not self.health_api_base_url or not self.health_api_key:
                logger.info("Health API credentials not configured, using fallback data")
                return None
            
            headers = {
                "Authorization": f"Bearer {self.health_api_key}",
                "Content-Type": "application/json"
            }
            
            params = {"format": "json"}
            if location:
                params["location"] = location
            
            response = requests.get(
                f"{self.health_api_base_url}/outbreak-alerts",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching outbreak alerts from API: {str(e)}")
            return None
    
    async def add_outbreak_alert(self, alert_data: Dict) -> bool:
        """Add new outbreak alert"""
        try:
            from chatbot.database import DatabaseManager
            db_manager = DatabaseManager()
            
            # Add to database
            await db_manager.add_outbreak_alert(alert_data)
            
            # Send notifications to users in affected area
            await self._notify_users_about_outbreak(alert_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding outbreak alert: {str(e)}")
            return False
    
    async def _notify_users_about_outbreak(self, alert_data: Dict):
        """Notify users about outbreak in their area"""
        try:
            # Get users in affected location
            from chatbot.database import DatabaseManager
            db_manager = DatabaseManager()
            
            # This would query users by location in a real implementation
            # For now, we'll use a sample list
            affected_users = await self._get_users_in_location(alert_data.get("location", ""))
            
            if affected_users:
                # Send WhatsApp notifications
                from chatbot.integrations import WhatsAppService
                whatsapp_service = WhatsAppService()
                
                await whatsapp_service.send_outbreak_alert(
                    [user["phone_number"] for user in affected_users if user.get("phone_number")],
                    alert_data
                )
                
                # Send SMS notifications
                sms_service = SMSService()
                await sms_service.send_outbreak_alert_sms(
                    [user["phone_number"] for user in affected_users if user.get("phone_number")],
                    alert_data
                )
            
        except Exception as e:
            logger.error(f"Error notifying users about outbreak: {str(e)}")
    
    async def _get_users_in_location(self, location: str) -> List[Dict]:
        """Get users in specific location (sample implementation)"""
        # In a real implementation, this would query the database
        # for users in the specified location
        return [
            {"phone_number": "+1234567890", "location": location},
            {"phone_number": "+0987654321", "location": location}
        ]
    
    async def check_for_new_outbreaks(self) -> List[Dict]:
        """Periodically check for new outbreaks"""
        try:
            # This would be called by a scheduled task
            current_alerts = await self.get_outbreak_alerts()
            
            # Check for new alerts by comparing with stored alerts
            from chatbot.database import DatabaseManager
            db_manager = DatabaseManager()
            stored_alerts = await db_manager.get_outbreak_alerts()
            
            new_alerts = []
            for alert in current_alerts:
                if not any(
                    stored["disease_name"] == alert["disease_name"] and 
                    stored["location"] == alert["location"]
                    for stored in stored_alerts
                ):
                    new_alerts.append(alert)
                    await self.add_outbreak_alert(alert)
            
            return new_alerts
            
        except Exception as e:
            logger.error(f"Error checking for new outbreaks: {str(e)}")
            return []
    
    async def send_vaccination_reminders(self) -> Dict:
        """Send vaccination reminders to users"""
        try:
            from chatbot.database import DatabaseManager
            db_manager = DatabaseManager()
            
            # Get users who need vaccination reminders
            users_needing_reminders = await self._get_users_needing_vaccination_reminders()
            
            results = {
                "successful": [],
                "failed": []
            }
            
            for user in users_needing_reminders:
                try:
                    # Send WhatsApp reminder
                    from chatbot.integrations import WhatsAppService
                    whatsapp_service = WhatsAppService()
                    
                    await whatsapp_service.send_vaccination_reminder(
                        user["phone_number"],
                        user["vaccine_name"],
                        user["due_date"],
                        user["location"]
                    )
                    
                    # Send SMS reminder
                    sms_service = SMSService()
                    await sms_service.send_vaccination_reminder_sms(
                        user["phone_number"],
                        user["vaccine_name"],
                        user["due_date"],
                        user["location"]
                    )
                    
                    results["successful"].append(user["user_id"])
                    
                except Exception as e:
                    logger.error(f"Error sending vaccination reminder to {user['user_id']}: {str(e)}")
                    results["failed"].append(user["user_id"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error sending vaccination reminders: {str(e)}")
            return {"successful": [], "failed": []}
    
    async def _get_users_needing_vaccination_reminders(self) -> List[Dict]:
        """Get users who need vaccination reminders (sample implementation)"""
        # In a real implementation, this would query the database
        # for users whose vaccination dates are approaching
        return [
            {
                "user_id": "user1",
                "phone_number": "+1234567890",
                "vaccine_name": "COVID-19 Booster",
                "due_date": "2024-01-15",
                "location": "Mumbai Health Center"
            },
            {
                "user_id": "user2",
                "phone_number": "+0987654321",
                "vaccine_name": "Influenza",
                "due_date": "2024-01-20",
                "location": "Delhi Health Center"
            }
        ]
    
    async def get_alert_statistics(self) -> Dict:
        """Get statistics about alerts and notifications"""
        try:
            from chatbot.database import DatabaseManager
            db_manager = DatabaseManager()
            
            # Get outbreak alerts
            outbreak_alerts = await db_manager.get_outbreak_alerts()
            
            # Calculate statistics
            total_alerts = len(outbreak_alerts)
            active_alerts = len([alert for alert in outbreak_alerts if alert.get("expires_at", "") > datetime.now().isoformat()])
            
            # Group by severity
            severity_counts = {}
            for alert in outbreak_alerts:
                severity = alert.get("severity", "unknown")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Group by disease
            disease_counts = {}
            for alert in outbreak_alerts:
                disease = alert.get("disease_name", "unknown")
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
            
            return {
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "severity_distribution": severity_counts,
                "disease_distribution": disease_counts,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting alert statistics: {str(e)}")
            return {}
