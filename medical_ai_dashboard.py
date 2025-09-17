"""
Medical AI Performance Dashboard and Automation Hub

This script provides a comprehensive dashboard for testing, monitoring, 
and improving the medical AI system with 500+ diseases coverage.
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
import logging

# Import our testing and improvement modules
from medical_ai_tester import MedicalAITester
from medical_ai_improver import MedicalAIImprover

# Configure rich console
console = Console()
logging.basicConfig(level=logging.INFO)

class MedicalAIDashboard:
    def __init__(self):
        self.tester = MedicalAITester()
        self.improver = MedicalAIImprover()
        self.performance_data = []
        self.is_monitoring = False
        self.console = Console()
        
        # Dashboard metrics
        self.metrics = {
            "total_tests": 0,
            "avg_completeness": 0.0,
            "avg_relevance": 0.0,
            "avg_response_time": 0.0,
            "emergency_detection_rate": 0.0,
            "categories_tested": 0,
            "improvements_implemented": 0,
            "last_test_time": None,
            "system_status": "Idle"
        }

    def create_dashboard_layout(self) -> Layout:
        """Create the main dashboard layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="metrics", size=15),
            Layout(name="categories")
        )
        
        layout["right"].split_column(
            Layout(name="recent_tests", size=15),
            Layout(name="improvements")
        )
        
        return layout

    def update_header(self, layout: Layout):
        """Update dashboard header"""
        header_text = Text("🏥 MEDICAL AI PERFORMANCE DASHBOARD", style="bold blue")
        subtitle = Text(f"Monitoring 500+ diseases • Last updated: {datetime.now().strftime('%H:%M:%S')}", style="dim")
        
        layout["header"].update(Panel(
            f"{header_text}\n{subtitle}",
            style="blue"
        ))

    def update_metrics_panel(self, layout: Layout):
        """Update the metrics panel"""
        table = Table(title="📊 System Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        # Add metrics
        table.add_row("Total Tests", str(self.metrics["total_tests"]), "✅" if self.metrics["total_tests"] > 0 else "⏳")
        table.add_row("Avg Completeness", f"{self.metrics['avg_completeness']:.2%}", "✅" if self.metrics["avg_completeness"] > 0.75 else "⚠️")
        table.add_row("Avg Relevance", f"{self.metrics['avg_relevance']:.2%}", "✅" if self.metrics["avg_relevance"] > 0.80 else "⚠️")
        table.add_row("Response Time", f"{self.metrics['avg_response_time']:.2f}s", "✅" if self.metrics["avg_response_time"] < 3.0 else "⚠️")
        table.add_row("Emergency Detection", f"{self.metrics['emergency_detection_rate']:.2%}", "✅" if self.metrics["emergency_detection_rate"] > 0.90 else "🚨")
        table.add_row("Categories Tested", str(self.metrics["categories_tested"]), "✅" if self.metrics["categories_tested"] > 10 else "⏳")
        table.add_row("System Status", self.metrics["system_status"], "🟢" if self.metrics["system_status"] == "Active" else "🟡")
        
        layout["metrics"].update(Panel(table, style="green"))

    def update_categories_panel(self, layout: Layout):
        """Update the categories performance panel"""
        table = Table(title="🏥 Category Performance", show_header=True, header_style="bold cyan")
        table.add_column("Category", style="cyan")
        table.add_column("Completeness", style="green")
        table.add_column("Relevance", style="blue")
        table.add_column("Tests", style="yellow")
        
        # Sample category data (would be populated from actual test results)
        categories = [
            ("Cardiovascular", "85%", "92%", "25"),
            ("Respiratory", "78%", "88%", "22"),
            ("Neurological", "82%", "90%", "20"),
            ("Emergency", "95%", "96%", "18"),
            ("Infectious", "80%", "85%", "24"),
            ("Endocrine", "77%", "89%", "19")
        ]
        
        for cat, comp, rel, tests in categories:
            table.add_row(cat, comp, rel, tests)
        
        layout["categories"].update(Panel(table, style="blue"))

    def update_recent_tests_panel(self, layout: Layout):
        """Update recent tests panel"""
        table = Table(title="🔬 Recent Test Results", show_header=True, header_style="bold yellow")
        table.add_column("Time", style="dim")
        table.add_column("Query", style="cyan", max_width=30)
        table.add_column("Category", style="green")
        table.add_column("Score", style="yellow")
        
        # Sample recent tests (would be populated from actual results)
        recent_tests = [
            ("14:23", "chest pain symptoms", "cardiovascular", "92%"),
            ("14:22", "difficulty breathing", "respiratory", "88%"),
            ("14:21", "severe headache", "neurological", "85%"),
            ("14:20", "मुझे बुखार है", "hindi_queries", "90%"),
            ("14:19", "diabetes management", "endocrine", "93%")
        ]
        
        for time_str, query, category, score in recent_tests:
            table.add_row(time_str, query, category, score)
        
        layout["recent_tests"].update(Panel(table, style="yellow"))

    def update_improvements_panel(self, layout: Layout):
        """Update improvements panel"""
        table = Table(title="🎯 Active Improvements", show_header=True, header_style="bold red")
        table.add_column("Priority", style="red")
        table.add_column("Category", style="cyan")
        table.add_column("Action", style="green")
        table.add_column("Progress", style="yellow")
        
        # Sample improvements (would be populated from actual improvement actions)
        improvements = [
            ("HIGH", "Emergency", "Enhance detection", "75%"),
            ("MED", "Respiratory", "Expand knowledge", "60%"),
            ("MED", "Neurological", "Optimize responses", "40%"),
            ("LOW", "Dermatology", "Add conditions", "20%")
        ]
        
        for priority, category, action, progress in improvements:
            table.add_row(priority, category, action, progress)
        
        layout["improvements"].update(Panel(table, style="red"))

    def update_footer(self, layout: Layout):
        """Update dashboard footer"""
        footer_text = f"🤖 AI Models: BioBERT, ClinicalBERT, PubMedBERT | 📊 Database: 500+ diseases | 🌐 Languages: English, Hindi"
        layout["footer"].update(Panel(footer_text, style="dim"))

    async def run_live_dashboard(self):
        """Run the live updating dashboard"""
        layout = self.create_dashboard_layout()
        
        with Live(layout, refresh_per_second=1, screen=True):
            while True:
                # Update all panels
                self.update_header(layout)
                self.update_metrics_panel(layout)
                self.update_categories_panel(layout)
                self.update_recent_tests_panel(layout)
                self.update_improvements_panel(layout)
                self.update_footer(layout)
                
                await asyncio.sleep(2)

    async def run_comprehensive_testing_cycle(self):
        """Run a comprehensive testing cycle"""
        self.console.print("🚀 Starting comprehensive testing cycle...", style="bold green")
        
        self.metrics["system_status"] = "Testing"
        
        with Progress() as progress:
            # Create progress tasks
            test_task = progress.add_task("Running tests...", total=100)
            analysis_task = progress.add_task("Analyzing results...", total=100)
            improvement_task = progress.add_task("Implementing improvements...", total=100)
            
            # Run comprehensive test
            progress.update(test_task, advance=30)
            test_report = await self.tester.run_comprehensive_test()
            progress.update(test_task, advance=70)
            
            # Update metrics
            self.metrics.update({
                "total_tests": test_report['test_summary']['total_queries_tested'],
                "avg_completeness": test_report['test_summary']['overall_completeness_score'],
                "avg_relevance": test_report['test_summary']['overall_relevance_score'],
                "avg_response_time": test_report['test_summary']['average_response_time'],
                "categories_tested": test_report['test_summary']['categories_tested'],
                "last_test_time": datetime.now(),
                "system_status": "Analyzing"
            })
            
            progress.update(analysis_task, advance=50)
            
            # Run improvement analysis
            improvement_result = await self.improver.run_continuous_improvement()
            progress.update(analysis_task, advance=50)
            progress.update(improvement_task, advance=100)
            
            self.metrics["improvements_implemented"] = len(improvement_result['implementations']['successful_implementations'])
            self.metrics["system_status"] = "Active"
        
        self.console.print("✅ Testing cycle completed!", style="bold green")
        return test_report, improvement_result

    def generate_performance_charts(self):
        """Generate performance visualization charts"""
        
        # Create sample data for visualization
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        performance_data = {
            'date': dates,
            'completeness': [0.75 + 0.1 * (i % 10) / 10 for i in range(30)],
            'relevance': [0.80 + 0.08 * (i % 8) / 8 for i in range(30)],
            'response_time': [2.5 + 0.5 * (i % 5) / 5 for i in range(30)]
        }
        
        df = pd.DataFrame(performance_data)
        
        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Medical AI Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Completeness over time
        axes[0, 0].plot(df['date'], df['completeness'], marker='o', color='green')
        axes[0, 0].set_title('Completeness Score Trend')
        axes[0, 0].set_ylabel('Completeness Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Relevance over time
        axes[0, 1].plot(df['date'], df['relevance'], marker='o', color='blue')
        axes[0, 1].set_title('Relevance Score Trend')
        axes[0, 1].set_ylabel('Relevance Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Response time over time
        axes[1, 0].plot(df['date'], df['response_time'], marker='o', color='orange')
        axes[1, 0].set_title('Response Time Trend')
        axes[1, 0].set_ylabel('Response Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Category performance heatmap
        categories = ['Cardiovascular', 'Respiratory', 'Neurological', 'Emergency', 'Infectious', 'Endocrine']
        metrics = ['Completeness', 'Relevance', 'Speed']
        data = [[0.85, 0.92, 0.88], [0.78, 0.88, 0.85], [0.82, 0.90, 0.90], 
                [0.95, 0.96, 0.95], [0.80, 0.85, 0.82], [0.77, 0.89, 0.87]]
        
        im = axes[1, 1].imshow(data, cmap='RdYlGn', aspect='auto')
        axes[1, 1].set_xticks(range(len(metrics)))
        axes[1, 1].set_yticks(range(len(categories)))
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].set_yticklabels(categories)
        axes[1, 1].set_title('Category Performance Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f'medical_ai_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

    async def continuous_monitoring_mode(self):
        """Run continuous monitoring with periodic testing and improvement"""
        self.console.print("🔄 Starting continuous monitoring mode...", style="bold blue")
        self.is_monitoring = True
        
        # Start dashboard in background
        dashboard_task = asyncio.create_task(self.run_live_dashboard())
        
        test_interval = 3600  # 1 hour
        
        while self.is_monitoring:
            try:
                self.console.print(f"⏰ Next test cycle in {test_interval} seconds...", style="dim")
                await asyncio.sleep(test_interval)
                
                if self.is_monitoring:
                    await self.run_comprehensive_testing_cycle()
                    
            except KeyboardInterrupt:
                self.console.print("🛑 Stopping continuous monitoring...", style="bold red")
                self.is_monitoring = False
                break
            except Exception as e:
                self.console.print(f"❌ Error in monitoring: {e}", style="bold red")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
        
        dashboard_task.cancel()

    def show_main_menu(self):
        """Show the main menu"""
        self.console.print("\n" + "="*60, style="bold blue")
        self.console.print("🏥 MEDICAL AI TESTING & IMPROVEMENT DASHBOARD", style="bold blue", justify="center")
        self.console.print("500+ Diseases • Multi-Model AI • Continuous Learning", style="dim", justify="center")
        self.console.print("="*60 + "\n", style="bold blue")
        
        menu_table = Table(show_header=False, box=None)
        menu_table.add_column("Option", style="bold cyan", width=3)
        menu_table.add_column("Description", style="white")
        menu_table.add_column("Details", style="dim")
        
        menu_table.add_row("1", "🔬 Run Comprehensive Test", "Test all medical categories")
        menu_table.add_row("2", "🎯 Performance Analysis", "Analyze and improve system")
        menu_table.add_row("3", "📊 Live Dashboard", "Real-time monitoring dashboard")
        menu_table.add_row("4", "🔄 Continuous Monitoring", "24/7 testing and improvement")
        menu_table.add_row("5", "📈 Generate Charts", "Create performance visualizations")
        menu_table.add_row("6", "⚡ Quick Test Suite", "Fast sample testing")
        menu_table.add_row("7", "🌐 Hindi Language Test", "Test multilingual support")
        menu_table.add_row("8", "🚨 Emergency Detection Test", "Validate emergency responses")
        menu_table.add_row("9", "💾 Export Results", "Save detailed reports")
        menu_table.add_row("0", "🚪 Exit", "Quit application")
        
        self.console.print(menu_table)
        
        return input("\n🎯 Select option (0-9): ").strip()

async def main():
    """Main application entry point"""
    dashboard = MedicalAIDashboard()
    
    while True:
        try:
            choice = dashboard.show_main_menu()
            
            if choice == "0":
                dashboard.console.print("👋 Goodbye! Keep improving healthcare AI!", style="bold green")
                break
                
            elif choice == "1":
                dashboard.console.print("\n🔬 Running comprehensive test...", style="bold yellow")
                test_report, improvement_result = await dashboard.run_comprehensive_testing_cycle()
                dashboard.console.print(f"✅ Test completed! {test_report['test_summary']['total_queries_tested']} queries tested.", style="bold green")
                
            elif choice == "2":
                dashboard.console.print("\n🎯 Running performance analysis...", style="bold yellow")
                improvement_result = await dashboard.improver.run_continuous_improvement()
                dashboard.console.print("✅ Analysis completed! Check improvement report.", style="bold green")
                
            elif choice == "3":
                dashboard.console.print("\n📊 Starting live dashboard...", style="bold yellow")
                dashboard.console.print("Press Ctrl+C to exit dashboard", style="dim")
                await dashboard.run_live_dashboard()
                
            elif choice == "4":
                dashboard.console.print("\n🔄 Starting continuous monitoring...", style="bold yellow")
                dashboard.console.print("Press Ctrl+C to stop monitoring", style="dim")
                await dashboard.continuous_monitoring_mode()
                
            elif choice == "5":
                dashboard.console.print("\n📈 Generating performance charts...", style="bold yellow")
                dashboard.generate_performance_charts()
                dashboard.console.print("✅ Charts generated and saved!", style="bold green")
                
            elif choice == "6":
                dashboard.console.print("\n⚡ Running quick test suite...", style="bold yellow")
                quick_queries = [
                    ("chest pain", "cardiovascular"),
                    ("difficulty breathing", "respiratory"),
                    ("severe headache", "neurological"),
                    ("high fever", "infectious"),
                    ("मुझे बुखार है", "hindi_queries")
                ]
                
                for query, category in quick_queries:
                    result = await dashboard.tester.test_medical_ai(query, category)
                    if result:
                        dashboard.console.print(f"  ✓ {query}: {result.relevance_score:.2%} relevance, {result.response_time:.2f}s", style="green")
                
            elif choice == "7":
                dashboard.console.print("\n🌐 Testing Hindi language support...", style="bold yellow")
                hindi_queries = ["मुझे बुखार है", "सिर दर्द", "पेट में दर्द", "सांस लेने में तकलीफ"]
                for query in hindi_queries:
                    result = await dashboard.tester.test_medical_ai(query, "hindi_queries")
                    if result:
                        dashboard.console.print(f"  ✓ {query}: {result.language_detected} detected", style="green")
                
            elif choice == "8":
                dashboard.console.print("\n🚨 Testing emergency detection...", style="bold yellow")
                emergency_queries = ["cardiac arrest", "severe chest pain", "can't breathe", "unconscious"]
                for query in emergency_queries:
                    result = await dashboard.tester.test_medical_ai(query, "emergency")
                    if result:
                        status = "✅ DETECTED" if result.emergency_detected else "❌ MISSED"
                        dashboard.console.print(f"  {status}: {query}", style="green" if result.emergency_detected else "red")
                
            elif choice == "9":
                dashboard.console.print("\n💾 Exporting results...", style="bold yellow")
                dashboard.tester.save_results()
                dashboard.console.print("✅ Results exported successfully!", style="bold green")
                
            else:
                dashboard.console.print("❌ Invalid option. Please try again.", style="bold red")
                
            if choice != "0":
                input("\nPress Enter to continue...")
                
        except KeyboardInterrupt:
            dashboard.console.print("\n🛑 Operation cancelled by user.", style="bold yellow")
            dashboard.is_monitoring = False
        except Exception as e:
            dashboard.console.print(f"❌ Error: {e}", style="bold red")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Application terminated by user.")
    except Exception as e:
        print(f"❌ Application error: {e}")