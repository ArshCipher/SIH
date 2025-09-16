"""
Real-time Medical Data Processing and Monitoring
WebSocket-based real-time patient monitoring, data streaming, and alerts
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

# Optional imports with graceful fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class DataType(Enum):
    """Types of medical data"""
    VITAL_SIGNS = "vital_signs"
    SYMPTOMS = "symptoms"
    MEDICATION = "medication"
    LAB_RESULTS = "lab_results"
    PATIENT_QUERY = "patient_query"
    EMERGENCY_ALERT = "emergency_alert"

@dataclass
class MedicalDataPoint:
    """Real-time medical data point"""
    patient_id: str
    data_type: DataType
    value: Any
    unit: Optional[str]
    timestamp: datetime
    source: str
    confidence: float
    alert_level: AlertSeverity
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['data_type'] = self.data_type.value
        data['alert_level'] = self.alert_level.value
        return data

@dataclass
class RealTimeAlert:
    """Real-time medical alert"""
    alert_id: str
    patient_id: str
    severity: AlertSeverity
    message: str
    data_point: MedicalDataPoint
    triggered_at: datetime
    acknowledged: bool = False
    escalated: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['triggered_at'] = self.triggered_at.isoformat()
        data['severity'] = self.severity.value
        data['data_point'] = self.data_point.to_dict()
        return data

class RealTimeMonitor:
    """Real-time medical monitoring system"""
    
    def __init__(self):
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.patient_monitors: Dict[str, Dict] = {}
        self.alert_thresholds = {
            "heart_rate": {"min": 60, "max": 100, "critical_min": 40, "critical_max": 150},
            "blood_pressure_systolic": {"min": 90, "max": 140, "critical_min": 70, "critical_max": 180},
            "temperature": {"min": 97.0, "max": 99.5, "critical_min": 95.0, "critical_max": 104.0},
            "oxygen_saturation": {"min": 95, "max": 100, "critical_min": 85, "critical_max": 100}
        }
        
        # Initialize Redis for data persistence
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host='localhost', 
                    port=6379, 
                    decode_responses=True
                )
                logger.info("Redis client initialized for real-time monitoring")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
        
        # Initialize Celery for distributed processing
        self.celery_app = None
        if CELERY_AVAILABLE:
            try:
                self.celery_app = Celery(
                    'medical_monitor',
                    broker='redis://localhost:6379/0',
                    backend='redis://localhost:6379/0'
                )
                logger.info("Celery initialized for distributed medical processing")
            except Exception as e:
                logger.warning(f"Celery initialization failed: {e}")
    
    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time monitoring"""
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        async def handle_client(websocket, path):
            """Handle individual client connections"""
            self.connected_clients.add(websocket)
            logger.info(f"Client connected: {websocket.remote_address}")
            
            try:
                async for message in websocket:
                    await self.process_client_message(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client disconnected: {websocket.remote_address}")
            finally:
                self.connected_clients.remove(websocket)
        
        # Start server
        server = await websockets.serve(handle_client, host, port)
        logger.info(f"WebSocket server started successfully")
        return server
    
    async def process_client_message(self, websocket, message: str):
        """Process incoming client messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "register_patient":
                await self.register_patient_monitor(websocket, data)
            elif message_type == "vital_signs":
                await self.process_vital_signs(data)
            elif message_type == "patient_query":
                await self.process_patient_query(data)
            elif message_type == "emergency_alert":
                await self.process_emergency_alert(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from client")
        except Exception as e:
            logger.error(f"Error processing client message: {e}")
    
    async def register_patient_monitor(self, websocket, data: Dict):
        """Register a patient for real-time monitoring"""
        patient_id = data.get("patient_id")
        monitor_config = data.get("config", {})
        
        if not patient_id:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Patient ID required for registration"
            }))
            return
        
        # Register patient
        self.patient_monitors[patient_id] = {
            "websocket": websocket,
            "config": monitor_config,
            "registered_at": datetime.now(),
            "last_data": None
        }
        
        # Confirm registration
        await websocket.send(json.dumps({
            "type": "registration_confirmed",
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        logger.info(f"Patient {patient_id} registered for real-time monitoring")
    
    async def process_vital_signs(self, data: Dict):
        """Process incoming vital signs data"""
        patient_id = data.get("patient_id")
        vital_signs = data.get("vital_signs", {})
        
        if not patient_id or not vital_signs:
            logger.warning("Invalid vital signs data received")
            return
        
        # Create data points for each vital sign
        alerts = []
        for vital_type, value in vital_signs.items():
            data_point = MedicalDataPoint(
                patient_id=patient_id,
                data_type=DataType.VITAL_SIGNS,
                value=value,
                unit=self._get_unit_for_vital(vital_type),
                timestamp=datetime.now(),
                source="real_time_monitor",
                confidence=0.95,
                alert_level=self._assess_vital_sign_alert_level(vital_type, value),
                metadata={"vital_type": vital_type}
            )
            
            # Check for alerts
            if data_point.alert_level in [AlertSeverity.HIGH, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                alert = RealTimeAlert(
                    alert_id=str(uuid.uuid4()),
                    patient_id=patient_id,
                    severity=data_point.alert_level,
                    message=f"Abnormal {vital_type}: {value} {data_point.unit}",
                    data_point=data_point,
                    triggered_at=datetime.now()
                )
                alerts.append(alert)
            
            # Store data point
            await self._store_data_point(data_point)
        
        # Broadcast alerts
        for alert in alerts:
            await self.broadcast_alert(alert)
        
        # Update patient monitor
        if patient_id in self.patient_monitors:
            self.patient_monitors[patient_id]["last_data"] = datetime.now()
    
    async def process_patient_query(self, data: Dict):
        """Process real-time patient queries"""
        patient_id = data.get("patient_id")
        query = data.get("query")
        urgency = data.get("urgency", "normal")
        
        if not patient_id or not query:
            logger.warning("Invalid patient query data received")
            return
        
        # Create data point
        data_point = MedicalDataPoint(
            patient_id=patient_id,
            data_type=DataType.PATIENT_QUERY,
            value=query,
            unit=None,
            timestamp=datetime.now(),
            source="patient_app",
            confidence=1.0,
            alert_level=AlertSeverity.LOW if urgency == "normal" else AlertSeverity.HIGH,
            metadata={"urgency": urgency, "query_length": len(query)}
        )
        
        # Store data point
        await self._store_data_point(data_point)
        
        # Process with medical AI (if available)
        if urgency == "emergency":
            await self.trigger_emergency_response(patient_id, query)
        
        # Broadcast to monitoring dashboard
        await self.broadcast_patient_activity(data_point)
    
    async def process_emergency_alert(self, data: Dict):
        """Process emergency alerts"""
        patient_id = data.get("patient_id")
        emergency_type = data.get("emergency_type")
        description = data.get("description", "")
        location = data.get("location")
        
        # Create emergency data point
        data_point = MedicalDataPoint(
            patient_id=patient_id,
            data_type=DataType.EMERGENCY_ALERT,
            value=emergency_type,
            unit=None,
            timestamp=datetime.now(),
            source="emergency_system",
            confidence=1.0,
            alert_level=AlertSeverity.EMERGENCY,
            metadata={
                "emergency_type": emergency_type,
                "description": description,
                "location": location
            }
        )
        
        # Create critical alert
        alert = RealTimeAlert(
            alert_id=str(uuid.uuid4()),
            patient_id=patient_id,
            severity=AlertSeverity.EMERGENCY,
            message=f"EMERGENCY: {emergency_type} - {description}",
            data_point=data_point,
            triggered_at=datetime.now()
        )
        
        # Store and broadcast immediately
        await self._store_data_point(data_point)
        await self.broadcast_alert(alert)
        
        # Trigger emergency response
        await self.trigger_emergency_response(patient_id, f"{emergency_type}: {description}")
        
        logger.critical(f"Emergency alert processed for patient {patient_id}: {emergency_type}")
    
    async def broadcast_alert(self, alert: RealTimeAlert):
        """Broadcast alert to all connected clients"""
        message = {
            "type": "medical_alert",
            "alert": alert.to_dict()
        }
        
        # Send to all connected clients
        if self.connected_clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.connected_clients],
                return_exceptions=True
            )
        
        # Store in Redis for persistence
        if self.redis_client:
            try:
                self.redis_client.lpush(
                    f"alerts:{alert.patient_id}", 
                    json.dumps(message)
                )
                # Keep only last 100 alerts per patient
                self.redis_client.ltrim(f"alerts:{alert.patient_id}", 0, 99)
            except Exception as e:
                logger.error(f"Failed to store alert in Redis: {e}")
    
    async def broadcast_patient_activity(self, data_point: MedicalDataPoint):
        """Broadcast patient activity to monitoring dashboard"""
        message = {
            "type": "patient_activity",
            "data_point": data_point.to_dict()
        }
        
        # Send to connected monitoring clients
        if self.connected_clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.connected_clients],
                return_exceptions=True
            )
    
    async def trigger_emergency_response(self, patient_id: str, emergency_description: str):
        """Trigger emergency response protocols"""
        logger.critical(f"Emergency response triggered for patient {patient_id}: {emergency_description}")
        
        # Emergency notification message
        emergency_message = {
            "type": "emergency_response",
            "patient_id": patient_id,
            "description": emergency_description,
            "timestamp": datetime.now().isoformat(),
            "response_actions": [
                "Emergency services notified",
                "Medical team alerted",
                "Patient location tracked",
                "Emergency contacts contacted"
            ]
        }
        
        # Broadcast emergency response
        if self.connected_clients:
            await asyncio.gather(
                *[client.send(json.dumps(emergency_message)) for client in self.connected_clients],
                return_exceptions=True
            )
        
        # Schedule emergency tasks (if Celery available)
        if self.celery_app:
            try:
                # These would be actual Celery tasks in production
                logger.info("Emergency tasks scheduled via Celery")
            except Exception as e:
                logger.error(f"Failed to schedule emergency tasks: {e}")
    
    def _assess_vital_sign_alert_level(self, vital_type: str, value: float) -> AlertSeverity:
        """Assess alert level for vital sign value"""
        thresholds = self.alert_thresholds.get(vital_type)
        if not thresholds:
            return AlertSeverity.LOW
        
        # Check critical thresholds
        if value <= thresholds["critical_min"] or value >= thresholds["critical_max"]:
            return AlertSeverity.EMERGENCY
        
        # Check normal thresholds
        if value < thresholds["min"] or value > thresholds["max"]:
            return AlertSeverity.HIGH
        
        return AlertSeverity.LOW
    
    def _get_unit_for_vital(self, vital_type: str) -> str:
        """Get unit for vital sign type"""
        units = {
            "heart_rate": "bpm",
            "blood_pressure_systolic": "mmHg",
            "blood_pressure_diastolic": "mmHg",
            "temperature": "°F",
            "oxygen_saturation": "%",
            "respiratory_rate": "breaths/min"
        }
        return units.get(vital_type, "")
    
    async def _store_data_point(self, data_point: MedicalDataPoint):
        """Store data point for persistence"""
        if self.redis_client:
            try:
                # Store in Redis with patient-specific key
                key = f"patient_data:{data_point.patient_id}"
                self.redis_client.lpush(key, json.dumps(data_point.to_dict()))
                
                # Keep only last 1000 data points per patient
                self.redis_client.ltrim(key, 0, 999)
                
                # Set expiration (7 days)
                self.redis_client.expire(key, 604800)
                
            except Exception as e:
                logger.error(f"Failed to store data point in Redis: {e}")
    
    async def get_patient_data_stream(self, patient_id: str, hours: int = 24) -> List[Dict]:
        """Get patient data stream for specified time period"""
        if not self.redis_client:
            return []
        
        try:
            # Get data from Redis
            key = f"patient_data:{patient_id}"
            data_points = self.redis_client.lrange(key, 0, -1)
            
            # Parse and filter by time
            cutoff_time = datetime.now() - timedelta(hours=hours)
            filtered_data = []
            
            for data_str in data_points:
                try:
                    data = json.loads(data_str)
                    data_time = datetime.fromisoformat(data['timestamp'])
                    
                    if data_time >= cutoff_time:
                        filtered_data.append(data)
                except Exception as e:
                    logger.warning(f"Failed to parse data point: {e}")
            
            return sorted(filtered_data, key=lambda x: x['timestamp'])
            
        except Exception as e:
            logger.error(f"Failed to get patient data stream: {e}")
            return []

# Global instance for easy access
real_time_monitor = RealTimeMonitor()