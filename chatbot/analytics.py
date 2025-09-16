from fastapi.responses import HTMLResponse
import json
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from sqlalchemy import func
from chatbot.database import Conversation

logger = logging.getLogger(__name__)

class AnalyticsDashboard:
    """Analytics dashboard for monitoring chatbot effectiveness"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    async def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data"""
        try:
            # Get basic analytics
            basic_analytics = await self.db_manager.get_analytics_data()
            
            # Get additional metrics
            conversation_metrics = await self._get_conversation_metrics()
            user_metrics = await self._get_user_metrics()
            health_metrics = await self._get_health_metrics()
            platform_metrics = await self._get_platform_metrics()
            
            return {
                "basic_analytics": basic_analytics,
                "conversation_metrics": conversation_metrics,
                "user_metrics": user_metrics,
                "health_metrics": health_metrics,
                "platform_metrics": platform_metrics,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            return {}
    
    async def _get_conversation_metrics(self) -> Dict:
        """Get conversation-specific metrics"""
        try:
            session = self.db_manager.get_session()
            
            # Total conversations
            total_conversations = session.query(Conversation).count()
            
            # Conversations in last 24 hours
            yesterday = datetime.now() - timedelta(days=1)
            recent_conversations = session.query(Conversation).filter(
                Conversation.timestamp >= yesterday
            ).count()
            
            # Average confidence score
            avg_confidence = session.query(
                func.avg(Conversation.confidence)
            ).scalar() or 0
            
            # Intent distribution
            intent_counts = {}
            intents = session.query(Conversation.intent).distinct().all()
            for intent in intents:
                count = session.query(Conversation).filter(
                    Conversation.intent == intent[0]
                ).count()
                intent_counts[intent[0]] = count
            
            # Language distribution
            language_counts = {}
            languages = session.query(Conversation.language).distinct().all()
            for language in languages:
                count = session.query(Conversation).filter(
                    Conversation.language == language[0]
                ).count()
                language_counts[language[0]] = count
            
            session.close()
            
            return {
                "total_conversations": total_conversations,
                "recent_conversations_24h": recent_conversations,
                "average_confidence": round(avg_confidence, 2),
                "intent_distribution": intent_counts,
                "language_distribution": language_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation metrics: {str(e)}")
            return {}
    
    async def _get_user_metrics(self) -> Dict:
        """Get user-specific metrics"""
        try:
            session = self.db_manager.get_session()
            
            # Total unique users
            total_users = session.query(
                func.count(func.distinct(Conversation.user_id))
            ).scalar()
            
            # Active users (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            active_users = session.query(
                func.count(func.distinct(Conversation.user_id))
            ).filter(Conversation.timestamp >= week_ago).scalar()
            
            # Users by platform
            platform_users = {}
            platforms = session.query(Conversation.platform).distinct().all()
            for platform in platforms:
                count = session.query(
                    func.count(func.distinct(Conversation.user_id))
                ).filter(Conversation.platform == platform[0]).scalar()
                platform_users[platform[0]] = count
            
            session.close()
            
            return {
                "total_users": total_users,
                "active_users_7d": active_users,
                "users_by_platform": platform_users
            }
            
        except Exception as e:
            logger.error(f"Error getting user metrics: {str(e)}")
            return {}
    
    async def _get_health_metrics(self) -> Dict:
        """Get health-specific metrics"""
        try:
            session = self.db_manager.get_session()
            
            # Most asked about diseases
            disease_queries = session.query(Conversation).filter(
                Conversation.intent == "disease_symptoms"
            ).all()
            
            # Extract disease names from messages (simplified)
            disease_counts = {}
            for query in disease_queries:
                message = query.message.lower()
                for disease in ["covid", "malaria", "dengue", "diabetes", "hypertension"]:
                    if disease in message:
                        disease_counts[disease] = disease_counts.get(disease, 0) + 1
            
            # Emergency queries
            emergency_queries = session.query(Conversation).filter(
                Conversation.intent == "emergency_help"
            ).count()
            
            # Vaccination queries
            vaccination_queries = session.query(Conversation).filter(
                Conversation.intent == "vaccination_schedule"
            ).count()
            
            session.close()
            
            return {
                "most_asked_diseases": disease_counts,
                "emergency_queries": emergency_queries,
                "vaccination_queries": vaccination_queries
            }
            
        except Exception as e:
            logger.error(f"Error getting health metrics: {str(e)}")
            return {}
    
    async def _get_platform_metrics(self) -> Dict:
        """Get platform-specific metrics"""
        try:
            session = self.db_manager.get_session()
            
            # Messages by platform
            platform_counts = {}
            platforms = session.query(Conversation.platform).distinct().all()
            for platform in platforms:
                count = session.query(Conversation).filter(
                    Conversation.platform == platform[0]
                ).count()
                platform_counts[platform[0]] = count
            
            # Response times by platform (simplified)
            platform_response_times = {}
            for platform in platforms:
                # This would calculate actual response times in a real implementation
                platform_response_times[platform[0]] = 1.5  # Sample value
            
            session.close()
            
            return {
                "messages_by_platform": platform_counts,
                "response_times_by_platform": platform_response_times
            }
            
        except Exception as e:
            logger.error(f"Error getting platform metrics: {str(e)}")
            return {}
    
    async def get_effectiveness_metrics(self) -> Dict:
        """Calculate effectiveness metrics for the chatbot"""
        try:
            session = self.db_manager.get_session()
            
            # Accuracy calculation (based on confidence scores)
            high_confidence_responses = session.query(Conversation).filter(
                Conversation.confidence >= 0.8
            ).count()
            
            total_responses = session.query(Conversation).count()
            accuracy = (high_confidence_responses / total_responses * 100) if total_responses > 0 else 0
            
            # User satisfaction (simplified - based on follow-up questions)
            follow_up_questions = session.query(Conversation).filter(
                Conversation.intent.in_(["general_health", "fallback"])
            ).count()
            
            satisfaction_rate = ((total_responses - follow_up_questions) / total_responses * 100) if total_responses > 0 else 0
            
            # Coverage metrics
            unique_users = session.query(
                func.count(func.distinct(Conversation.user_id))
            ).scalar()
            
            # Engagement metrics
            avg_messages_per_user = total_responses / unique_users if unique_users > 0 else 0
            
            session.close()
            
            return {
                "accuracy_percentage": round(accuracy, 2),
                "satisfaction_rate": round(satisfaction_rate, 2),
                "user_coverage": unique_users,
                "avg_messages_per_user": round(avg_messages_per_user, 2),
                "total_interactions": total_responses
            }
            
        except Exception as e:
            logger.error(f"Error getting effectiveness metrics: {str(e)}")
            return {}
    
    def generate_dashboard_html(self, data: Dict) -> str:
        """Generate HTML dashboard"""
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Public Health Chatbot Analytics</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                    margin-bottom: 10px;
                }}
                .metric-label {{
                    color: #666;
                    font-size: 0.9em;
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .chart-title {{
                    font-size: 1.2em;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #333;
                }}
                .status-indicator {{
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 5px;
                }}
                .status-healthy {{ background-color: #4CAF50; }}
                .status-warning {{ background-color: #FF9800; }}
                .status-error {{ background-color: #F44336; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 Public Health Chatbot Analytics</h1>
                    <p>Real-time monitoring and effectiveness metrics</p>
                    <p>Last updated: {data.get('generated_at', 'Unknown')}</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{data.get('basic_analytics', {}).get('total_conversations', 0)}</div>
                        <div class="metric-label">Total Conversations</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{data.get('conversation_metrics', {}).get('average_confidence', 0)}</div>
                        <div class="metric-label">Average Confidence</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{data.get('user_metrics', {}).get('total_users', 0)}</div>
                        <div class="metric-label">Total Users</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{data.get('user_metrics', {}).get('active_users_7d', 0)}</div>
                        <div class="metric-label">Active Users (7d)</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Platform Distribution</div>
                    <canvas id="platformChart" width="400" height="200"></canvas>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Language Distribution</div>
                    <canvas id="languageChart" width="400" height="200"></canvas>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Intent Distribution</div>
                    <canvas id="intentChart" width="400" height="200"></canvas>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Most Asked About Diseases</div>
                    <canvas id="diseaseChart" width="400" height="200"></canvas>
                </div>
            </div>
            
            <script>
                // Platform Chart
                const platformCtx = document.getElementById('platformChart').getContext('2d');
                new Chart(platformCtx, {{
                    type: 'doughnut',
                    data: {{
                        labels: {list(data.get('platform_metrics', {}).get('messages_by_platform', {}).keys())},
                        datasets: [{{
                            data: {list(data.get('platform_metrics', {}).get('messages_by_platform', {}).values())},
                            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false
                    }}
                }});
                
                // Language Chart
                const languageCtx = document.getElementById('languageChart').getContext('2d');
                new Chart(languageCtx, {{
                    type: 'bar',
                    data: {{
                        labels: {list(data.get('conversation_metrics', {}).get('language_distribution', {}).keys())},
                        datasets: [{{
                            label: 'Messages',
                            data: {list(data.get('conversation_metrics', {}).get('language_distribution', {}).values())},
                            backgroundColor: '#36A2EB'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false
                    }}
                }});
                
                // Intent Chart
                const intentCtx = document.getElementById('intentChart').getContext('2d');
                new Chart(intentCtx, {{
                    type: 'pie',
                    data: {{
                        labels: {list(data.get('conversation_metrics', {}).get('intent_distribution', {}).keys())},
                        datasets: [{{
                            data: {list(data.get('conversation_metrics', {}).get('intent_distribution', {}).values())},
                            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false
                    }}
                }});
                
                // Disease Chart
                const diseaseCtx = document.getElementById('diseaseChart').getContext('2d');
                new Chart(diseaseCtx, {{
                    type: 'horizontalBar',
                    data: {{
                        labels: {list(data.get('health_metrics', {}).get('most_asked_diseases', {}).keys())},
                        datasets: [{{
                            label: 'Queries',
                            data: {list(data.get('health_metrics', {}).get('most_asked_diseases', {}).values())},
                            backgroundColor: '#4BC0C0'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false
                    }}
                }});
            </script>
        </body>
        </html>
        """
        return html
