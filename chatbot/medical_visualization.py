"""
Advanced Medical Data Visualization Dashboard
Competition-grade medical analytics with Plotly, Seaborn, and interactive dashboards
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

# Core data processing
import numpy as np
import pandas as pd

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = sns = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = px = make_subplots = pyo = None

# FastAPI for web dashboard
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.templating import Jinja2Templates
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)

class MedicalVisualizationDashboard:
    """Competition-grade medical data visualization dashboard"""
    
    def __init__(self):
        self.app = None
        self.templates = None
        
        # Color schemes for medical data
        self.medical_colors = {
            "normal": "#2E8B57",      # Sea Green
            "warning": "#FF8C00",     # Dark Orange  
            "critical": "#DC143C",    # Crimson
            "emergency": "#8B0000",   # Dark Red
            "info": "#4682B4"         # Steel Blue
        }
        
        # Chart templates
        self.chart_configs = {
            "vital_signs": {
                "height": 400,
                "title_font_size": 16,
                "axis_font_size": 12
            },
            "trends": {
                "height": 500,
                "title_font_size": 18,
                "axis_font_size": 14
            },
            "alerts": {
                "height": 300,
                "title_font_size": 14,
                "axis_font_size": 10
            }
        }
        
        if FASTAPI_AVAILABLE:
            self._initialize_web_app()
    
    def _initialize_web_app(self):
        """Initialize FastAPI web application for dashboard"""
        self.app = FastAPI(title="Medical AI Dashboard", version="1.0.0")
        
        # Add routes
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return await self.render_main_dashboard()
        
        @self.app.get("/api/patient/{patient_id}/vitals")
        async def get_patient_vitals(patient_id: str):
            return await self.get_patient_vitals_data(patient_id)
        
        @self.app.get("/api/analytics/summary")
        async def get_analytics_summary():
            return await self.get_system_analytics()
        
        @self.app.get("/api/alerts/active")
        async def get_active_alerts():
            return await self.get_active_alerts()
    
    async def render_main_dashboard(self) -> str:
        """Render main medical dashboard"""
        
        # Generate sample data for demonstration
        dashboard_data = await self.generate_dashboard_data()
        
        # Create main dashboard HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Medical AI Dashboard - Competition Grade</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .dashboard-card {{
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin: 10px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                }}
                .alert-badge {{
                    font-size: 12px;
                    padding: 5px 10px;
                }}
            </style>
        </head>
        <body class="bg-light">
            <nav class="navbar navbar-dark bg-primary">
                <div class="container-fluid">
                    <span class="navbar-brand mb-0 h1">🏥 Medical AI Dashboard - Competition Grade</span>
                    <span class="badge bg-success">Live Monitoring Active</span>
                </div>
            </nav>
            
            <div class="container-fluid mt-3">
                <!-- Key Metrics Row -->
                <div class="row">
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3>{dashboard_data['total_patients']}</h3>
                            <p class="mb-0">Active Patients</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3>{dashboard_data['active_alerts']}</h3>
                            <p class="mb-0">Active Alerts</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3>{dashboard_data['ai_accuracy']:.1f}%</h3>
                            <p class="mb-0">AI Accuracy</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3>{dashboard_data['response_time']:.1f}s</h3>
                            <p class="mb-0">Avg Response Time</p>
                        </div>
                    </div>
                </div>
                
                <!-- Charts Row -->
                <div class="row">
                    <div class="col-md-8">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5>🫀 Real-time Vital Signs Monitoring</h5>
                            </div>
                            <div class="card-body">
                                <div id="vital-signs-chart" style="height: 400px;"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5>🚨 Recent Medical Alerts</h5>
                            </div>
                            <div class="card-body">
                                <div id="alerts-list">
                                    {self._generate_alerts_html(dashboard_data['recent_alerts'])}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Advanced Analytics Row -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5>📊 AI Model Performance</h5>
                            </div>
                            <div class="card-body">
                                <div id="model-performance-chart" style="height: 300px;"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card dashboard-card">
                            <div class="card-header">
                                <h5>🎯 Diagnostic Accuracy Trends</h5>
                            </div>
                            <div class="card-body">
                                <div id="accuracy-trends-chart" style="height: 300px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Vital Signs Chart
                const vitalSignsData = {json.dumps(dashboard_data['vital_signs_chart'])};
                Plotly.newPlot('vital-signs-chart', vitalSignsData.data, vitalSignsData.layout);
                
                // Model Performance Chart
                const modelPerfData = {json.dumps(dashboard_data['model_performance_chart'])};
                Plotly.newPlot('model-performance-chart', modelPerfData.data, modelPerfData.layout);
                
                // Accuracy Trends Chart
                const accuracyData = {json.dumps(dashboard_data['accuracy_trends_chart'])};
                Plotly.newPlot('accuracy-trends-chart', accuracyData.data, accuracyData.layout);
                
                // Auto-refresh every 30 seconds
                setInterval(() => {{
                    location.reload();
                }}, 30000);
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        
        # Generate sample data (in production, this would come from real medical data)
        current_time = datetime.now()
        time_points = [current_time - timedelta(minutes=i*5) for i in range(24)]
        
        # Vital signs data
        vital_signs_data = {
            "heart_rate": np.random.normal(75, 10, 24),
            "blood_pressure_sys": np.random.normal(120, 15, 24),
            "temperature": np.random.normal(98.6, 1, 24),
            "oxygen_sat": np.random.normal(98, 2, 24)
        }
        
        return {
            "total_patients": 1247,
            "active_alerts": 23,
            "ai_accuracy": 94.7,
            "response_time": 2.3,
            "recent_alerts": [
                {"patient": "Patient-001", "type": "High Blood Pressure", "severity": "warning", "time": "2 min ago"},
                {"patient": "Patient-045", "type": "Irregular Heart Rate", "severity": "critical", "time": "5 min ago"},
                {"patient": "Patient-123", "type": "Low Oxygen Saturation", "severity": "warning", "time": "8 min ago"},
                {"patient": "Patient-089", "type": "Emergency Alert", "severity": "emergency", "time": "12 min ago"}
            ],
            "vital_signs_chart": self._create_vital_signs_chart(time_points, vital_signs_data),
            "model_performance_chart": self._create_model_performance_chart(),
            "accuracy_trends_chart": self._create_accuracy_trends_chart()
        }
    
    def _create_vital_signs_chart(self, time_points: List[datetime], vital_data: Dict) -> Dict:
        """Create real-time vital signs chart"""
        
        if not PLOTLY_AVAILABLE:
            return {"data": [], "layout": {}}
        
        traces = []
        
        # Heart Rate
        traces.append(go.Scatter(
            x=[t.strftime("%H:%M") for t in time_points],
            y=vital_data["heart_rate"],
            mode='lines+markers',
            name='Heart Rate (bpm)',
            line=dict(color=self.medical_colors["info"], width=2),
            yaxis='y1'
        ))
        
        # Blood Pressure
        traces.append(go.Scatter(
            x=[t.strftime("%H:%M") for t in time_points],
            y=vital_data["blood_pressure_sys"],
            mode='lines+markers',
            name='Blood Pressure (mmHg)',
            line=dict(color=self.medical_colors["warning"], width=2),
            yaxis='y2'
        ))
        
        # Temperature
        traces.append(go.Scatter(
            x=[t.strftime("%H:%M") for t in time_points],
            y=vital_data["temperature"],
            mode='lines+markers',
            name='Temperature (°F)',
            line=dict(color=self.medical_colors["critical"], width=2),
            yaxis='y3'
        ))
        
        layout = go.Layout(
            title="Real-time Patient Vital Signs",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Heart Rate (bpm)", side="left", color=self.medical_colors["info"]),
            yaxis2=dict(title="Blood Pressure (mmHg)", side="right", overlaying="y", color=self.medical_colors["warning"]),
            yaxis3=dict(title="Temperature (°F)", side="right", overlaying="y", position=0.95, color=self.medical_colors["critical"]),
            legend=dict(x=0, y=1),
            hovermode='x unified',
            height=400
        )
        
        return {"data": traces, "layout": layout}
    
    def _create_model_performance_chart(self) -> Dict:
        """Create AI model performance chart"""
        
        if not PLOTLY_AVAILABLE:
            return {"data": [], "layout": {}}
        
        models = ["BioBERT", "ClinicalBERT", "PubMedBERT", "Medical GPT", "Safety Model", "Ensemble"]
        accuracy = [92.5, 94.2, 93.8, 96.1, 89.7, 97.3]
        
        trace = go.Bar(
            x=models,
            y=accuracy,
            marker=dict(
                color=[self.medical_colors["normal"] if acc >= 95 else 
                      self.medical_colors["warning"] if acc >= 90 else 
                      self.medical_colors["critical"] for acc in accuracy]
            ),
            text=[f"{acc}%" for acc in accuracy],
            textposition='auto'
        )
        
        layout = go.Layout(
            title="AI Model Accuracy Comparison",
            xaxis=dict(title="Model"),
            yaxis=dict(title="Accuracy (%)", range=[85, 100]),
            height=300
        )
        
        return {"data": [trace], "layout": layout}
    
    def _create_accuracy_trends_chart(self) -> Dict:
        """Create diagnostic accuracy trends chart"""
        
        if not PLOTLY_AVAILABLE:
            return {"data": [], "layout": {}}
        
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
        accuracy_trend = np.random.normal(94.5, 2, 30)
        accuracy_trend = np.clip(accuracy_trend, 88, 98)  # Keep realistic range
        
        trace = go.Scatter(
            x=dates,
            y=accuracy_trend,
            mode='lines+markers',
            name='Diagnostic Accuracy',
            line=dict(color=self.medical_colors["normal"], width=3),
            fill='tonexty',
            fillcolor='rgba(46, 139, 87, 0.1)'
        )
        
        # Add target line
        target_line = go.Scatter(
            x=dates,
            y=[95] * 30,
            mode='lines',
            name='Target (95%)',
            line=dict(color=self.medical_colors["info"], width=2, dash='dash')
        )
        
        layout = go.Layout(
            title="30-Day Diagnostic Accuracy Trend",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Accuracy (%)", range=[88, 98]),
            height=300,
            showlegend=True
        )
        
        return {"data": [trace, target_line], "layout": layout}
    
    def _generate_alerts_html(self, alerts: List[Dict]) -> str:
        """Generate HTML for recent alerts"""
        
        severity_badges = {
            "emergency": "danger",
            "critical": "danger", 
            "warning": "warning",
            "info": "info"
        }
        
        html = ""
        for alert in alerts:
            badge_class = severity_badges.get(alert["severity"], "secondary")
            html += f"""
            <div class="alert alert-{badge_class} alert-dismissible fade show" role="alert">
                <strong>{alert["patient"]}</strong><br>
                {alert["type"]}
                <span class="badge bg-{badge_class} alert-badge">{alert["severity"].upper()}</span>
                <br><small class="text-muted">{alert["time"]}</small>
            </div>
            """
        
        return html
    
    async def get_patient_vitals_data(self, patient_id: str) -> JSONResponse:
        """Get patient vital signs data for API"""
        
        # Generate sample patient data
        current_time = datetime.now()
        vitals = {
            "patient_id": patient_id,
            "timestamp": current_time.isoformat(),
            "heart_rate": np.random.normal(75, 10),
            "blood_pressure_systolic": np.random.normal(120, 15),
            "blood_pressure_diastolic": np.random.normal(80, 10),
            "temperature": np.random.normal(98.6, 1),
            "oxygen_saturation": np.random.normal(98, 2),
            "respiratory_rate": np.random.normal(16, 3)
        }
        
        return JSONResponse(content=vitals)
    
    async def get_system_analytics(self) -> JSONResponse:
        """Get system-wide analytics"""
        
        analytics = {
            "total_patients": 1247,
            "active_sessions": 89,
            "ai_queries_today": 3452,
            "average_response_time": 2.3,
            "accuracy_rate": 94.7,
            "emergency_alerts_today": 12,
            "system_uptime": "99.8%",
            "models_active": 6,
            "last_updated": datetime.now().isoformat()
        }
        
        return JSONResponse(content=analytics)
    
    async def get_active_alerts(self) -> JSONResponse:
        """Get current active medical alerts"""
        
        alerts = [
            {
                "alert_id": "ALT-001",
                "patient_id": "PAT-001", 
                "severity": "critical",
                "type": "Irregular Heart Rate",
                "description": "Heart rate variability detected outside normal parameters",
                "triggered_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "status": "active"
            },
            {
                "alert_id": "ALT-002",
                "patient_id": "PAT-045",
                "severity": "warning", 
                "type": "High Blood Pressure",
                "description": "Systolic pressure elevated to 150 mmHg",
                "triggered_at": (datetime.now() - timedelta(minutes=12)).isoformat(),
                "status": "active"
            }
        ]
        
        return JSONResponse(content={"alerts": alerts, "total_count": len(alerts)})
    
    def create_medical_report_visualizations(self, patient_data: Dict) -> Dict[str, Any]:
        """Create comprehensive medical report visualizations"""
        
        if not PLOTLY_AVAILABLE:
            return {}
        
        # Create multiple charts for medical reporting
        charts = {}
        
        # 1. Vital Signs Trend Chart
        if "vital_signs_history" in patient_data:
            charts["vital_trends"] = self._create_vital_trends_chart(patient_data["vital_signs_history"])
        
        # 2. Medication Adherence Chart
        if "medication_data" in patient_data:
            charts["medication_adherence"] = self._create_medication_chart(patient_data["medication_data"])
        
        # 3. Risk Assessment Chart
        if "risk_factors" in patient_data:
            charts["risk_assessment"] = self._create_risk_assessment_chart(patient_data["risk_factors"])
        
        return charts
    
    def _create_risk_assessment_chart(self, risk_data: Dict) -> Dict:
        """Create patient risk assessment visualization"""
        
        risk_categories = list(risk_data.keys())
        risk_scores = list(risk_data.values())
        
        # Color code based on risk level
        colors = [
            self.medical_colors["normal"] if score < 30 else
            self.medical_colors["warning"] if score < 60 else
            self.medical_colors["critical"]
            for score in risk_scores
        ]
        
        trace = go.Bar(
            x=risk_categories,
            y=risk_scores,
            marker=dict(color=colors),
            text=[f"{score}%" for score in risk_scores],
            textposition='auto'
        )
        
        layout = go.Layout(
            title="Patient Risk Assessment",
            xaxis=dict(title="Risk Factors"),
            yaxis=dict(title="Risk Score (%)", range=[0, 100]),
            height=400
        )
        
        return {"data": [trace], "layout": layout}

# Global instance for easy access
medical_dashboard = MedicalVisualizationDashboard()