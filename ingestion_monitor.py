#!/usr/bin/env python3
"""
Real-time monitoring dashboard for massive Chromium ingestion.
Simplified with clear separation of session vs overall data.
"""

import os
import sys
import json
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_system.vector.database import VectorDatabase


class IngestionMonitor:
    """Monitor for massive ingestion pipeline."""
    
    def __init__(self, cache_dir: Path = Path("data/massive_cache")):
        """Initialize the monitor."""
        self.cache_dir = cache_dir
        self.progress_file = cache_dir / "progress.json"
        self.error_log = cache_dir / "errors.log"
        self.vector_db = VectorDatabase(collection_name="chromium_complete")
    
    def load_progress(self):
        """Load current session progress."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_overall_stats(self):
        """Get overall accumulated statistics from database."""
        try:
            db_stats = self.vector_db.get_collection_stats()
            total_docs = db_stats.get('total_documents', 0)
            return {
                'total_documents': total_docs,
                'collection_name': db_stats.get('collection_name', 'chromium_complete'),
                'is_healthy': total_docs > 0
            }
        except Exception as e:
            return {'error': str(e), 'total_documents': 0}
    
    def get_system_resources(self):
        """Get current system resource usage."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3)
        }
    
    def get_recent_errors(self, count=5):
        """Get recent errors from log."""
        if not self.error_log.exists():
            return []
        try:
            with open(self.error_log, 'r') as f:
                errors = f.readlines()
            return errors[-count:]
        except Exception:
            return []


def create_gauge(value, title, color="darkblue"):
    """Create a gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=200)
    return fig


def create_dashboard():
    """Create Streamlit dashboard."""
    st.set_page_config(
        page_title="Chromium RAG Ingestion Monitor",
        page_icon="🚀",
        layout="wide"
    )
    
    # Hide sidebar completely with CSS
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
            [data-testid="collapsedControl"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and refresh control in header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚀 Chromium RAG Ingestion Monitor")
    with col2:
        refresh_interval = st.selectbox(
            "Refresh (sec)",
            [5, 10, 30, 60],
            index=1,
            label_visibility="visible"
        )
    
    # Initialize monitor
    monitor = IngestionMonitor()
    
    # Define 5-phase plan
    phases = [
        {"name": "Phase 1", "desc": "6 months", "target": 20000},
        {"name": "Phase 2", "desc": "1 year", "target": 40000},
        {"name": "Phase 3", "desc": "2 years", "target": 100000},
        {"name": "Phase 4", "desc": "5 years", "target": 300000},
        {"name": "Phase 5", "desc": "Complete", "target": 800000},
    ]
    
    # Load data
    session = monitor.load_progress()
    overall = monitor.get_overall_stats()
    resources = monitor.get_system_resources()
    errors = monitor.get_recent_errors()
    
    # Calculate metrics
    total_accumulated = overall.get('total_documents', 0)
    session_commits = session.get('commits_processed', 0)
    session_docs = session.get('documents_created', 0)
    session_start = session.get('session_start', 0)
    
    # Calculate rates
    if session_start > 0:
        elapsed = time.time() - session_start
        rate = session_commits / elapsed if elapsed > 0 else 0
    else:
        rate = 0
    
    # Main Dashboard - Overall Plan Status
    st.markdown("---")
    st.subheader("📋 Overall Plan Progress")
    
    # Overall progress bar at top
    total_percent = (total_accumulated / 1200000 * 100)
    st.progress(min(total_accumulated / 1200000, 1.0))
    st.caption(f"**Total:** {total_accumulated:,} / 1,200,000 commits ({total_percent:.1f}% complete)")
    
    # Phase progress in columns
    cols = st.columns(5)
    cumulative_target = 0
    for idx, phase in enumerate(phases):
        cumulative_target += phase["target"]
        phase_completed = max(0, min(total_accumulated - (cumulative_target - phase["target"]), phase["target"]))
        phase_percent = (phase_completed / phase["target"] * 100)
        
        if total_accumulated >= cumulative_target:
            status = "✅"
        elif phase_completed > 0:
            status = "⏳"
        else:
            status = "⭕"
        
        with cols[idx]:
            st.markdown(f"**{status} {phase['name']}**")
            st.caption(phase['desc'])
            st.progress(min(phase_percent / 100, 1.0))
            st.caption(f"{phase_completed:,}/{phase['target']:,}")
    
    # Main Dashboard
    st.markdown("---")
    
    # Overall Statistics (Accumulated)
    st.subheader("📊 Overall Statistics (Accumulated)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", f"{total_accumulated:,}")
    
    with col2:
        st.metric("Status", "🟢 Healthy" if overall.get('is_healthy') else "🔴 Empty")
    
    with col3:
        st.metric("Completion", f"{total_percent:.1f}%")
    
    st.markdown("---")
    
    # Current Session Statistics
    st.subheader("⚡ Current Session")
    
    if session_commits > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Commits Processed", f"{session_commits:,}")
        
        with col2:
            st.metric("Documents Created", f"{session_docs:,}")
        
        with col3:
            st.metric("Batches Completed", session.get('batches_completed', 0))
        
        with col4:
            st.metric("Processing Rate", f"{rate:.1f} c/s")
        
        # Session timeline
        if session_start > 0:
            elapsed_hours = (time.time() - session_start) / 3600
            avg_batch_time = elapsed_hours * 60 / session.get('batches_completed', 1)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Session Duration:** {elapsed_hours:.2f} hours")
                st.write(f"**Avg Batch Time:** {avg_batch_time:.2f} minutes")
            
            with col2:
                if rate > 0:
                    remaining = 1200000 - total_accumulated
                    eta_hours = remaining / (rate * 3600)
                    eta_time = datetime.now() + timedelta(hours=eta_hours)
                    st.write(f"**Estimated Completion:** {eta_time.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Remaining Time:** {eta_hours:.1f} hours")
    else:
        st.info("⏸️ No active ingestion session")
    
    st.markdown("---")
    
    # Phase Progress Table
    st.subheader("📋 Phase Breakdown")
    
    phase_data = []
    cumulative_target = 0
    for phase in phases:
        cumulative_target += phase["target"]
        phase_completed = max(0, min(total_accumulated - (cumulative_target - phase["target"]), phase["target"]))
        
        if total_accumulated >= cumulative_target:
            status = "✅ Complete"
        elif phase_completed > 0:
            status = "⏳ In Progress"
        else:
            status = "⭕ Pending"
        
        phase_data.append({
            "Phase": f"{phase['name']} ({phase['desc']})",
            "Target": f"{phase['target']:,}",
            "Completed": f"{phase_completed:,}",
            "Progress": f"{(phase_completed / phase['target'] * 100):.1f}%",
            "Status": status
        })
    
    df = pd.DataFrame(phase_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Commit Range (from session)
    if session.get('first_commit_sha') or session.get('last_commit_sha'):
        st.subheader("📍 Current Session Commit Range")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**First Commit:**")
            if session.get('first_commit_sha'):
                st.code(session['first_commit_sha'][:8])
                if session.get('first_commit_date'):
                    st.caption(session['first_commit_date'])
        
        with col2:
            st.markdown("**Last Commit:**")
            if session.get('last_commit_sha'):
                st.code(session['last_commit_sha'][:8])
                if session.get('last_commit_date'):
                    st.caption(session['last_commit_date'])
        
        st.markdown("---")
    
    # System Resources
    st.subheader("💻 System Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.plotly_chart(create_gauge(resources["cpu_percent"], "CPU Usage (%)", "darkblue"), 
                       use_container_width=True)
    
    with col2:
        st.plotly_chart(create_gauge(resources["memory_percent"], "Memory Usage (%)", "darkgreen"), 
                       use_container_width=True)
    
    with col3:
        st.plotly_chart(create_gauge(resources["disk_usage_percent"], "Disk Usage (%)", "darkorange"), 
                       use_container_width=True)
    
    st.write(f"**Memory:** {resources['memory_used_gb']:.1f} / {resources['memory_total_gb']:.1f} GB")
    st.write(f"**Disk Free:** {resources['disk_free_gb']:.1f} GB")
    
    st.markdown("---")
    
    # Error Monitoring
    st.subheader("⚠️ Errors")
    
    if errors:
        st.warning(f"Recent errors detected ({len(errors)})")
        for error in errors:
            st.code(error.strip(), language="text")
    else:
        st.success("No recent errors")
    
    st.markdown("---")
    
    # Control Panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    with col2:
        if st.button("📊 Export Stats"):
            export_data = {
                "overall": overall,
                "session": session,
                "resources": resources,
                "timestamp": datetime.now().isoformat()
            }
            st.download_button(
                "Download JSON",
                json.dumps(export_data, indent=2),
                file_name=f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Auto-refresh
    time.sleep(refresh_interval)
    st.rerun()


if __name__ == "__main__":
    create_dashboard()
