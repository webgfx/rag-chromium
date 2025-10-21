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

from rag_system.vector import VectorDatabase


class IngestionMonitor:
    """Monitor for massive ingestion pipeline."""
    
    def __init__(self, cache_dir: Path = Path("data/massive_cache")):
        """Initialize the monitor."""
        self.cache_dir = cache_dir
        self.status_file = Path('data/status.json')
        self.progress_file = cache_dir / "progress.json"
        self.error_log = cache_dir / "errors.log"
    
    def load_status(self):
        """Load comprehensive status from ingestion process."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"Failed to load status file: {e}")
                return self._get_fallback_status()
        return self._get_fallback_status()
    
    def _get_fallback_status(self):
        """Get fallback status from progress file if status file unavailable."""
        progress = {}
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
            except Exception:
                pass
        
        return {
            'timestamp': datetime.now().isoformat(),
            'progress': progress,
            'database': {
                'total_documents': progress.get('documents_created', 0),
                'collection_name': 'chromium_complete',
                'is_healthy': progress.get('documents_created', 0) > 0
            },
            'system': self._get_current_system_resources(),
            'stats': progress
        }
    
    def _get_current_system_resources(self):
        """Get current system resource usage."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "disk_free_gb": psutil.disk_usage('/').free / (1024**3)
            }
        except Exception as e:
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "memory_used_gb": 0,
                "memory_total_gb": 0,
                "disk_usage_percent": 0,
                "disk_free_gb": 0
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
    try:
        st.set_page_config(
            page_title="Chromium RAG Ingestion Monitor",
            page_icon="🚀",
            layout="wide"
        )
    except Exception as e:
        st.error(f"Error setting page config: {e}")
        return
    
    # Hide sidebar, running status icon, and header buttons with CSS
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
            [data-testid="collapsedControl"] {
                display: none;
            }
            [data-testid="stStatusWidgetRunningIcon"] {
                display: none;
            }
            [data-testid="stBaseButton-header"] {
                display: none !important;
            }
            button[kind="header"] {
                display: none !important;
            }
            .st-emotion-cache-1gk7ll6 {
                display: none !important;
            }
            /* Prevent content from fading during rerun */
            .stApp {
                opacity: 1 !important;
            }
            [data-testid="stAppViewContainer"] {
                opacity: 1 !important;
            }
            .element-container {
                opacity: 1 !important;
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
            [2, 5, 10, 30, 60],
            index=1,
            label_visibility="visible"
        )
        st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Auto-refresh using st.rerun after a delay
    # This allows the page to render first, then schedule the next refresh
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
        st.session_state.last_refresh_time = time.time()
    
    current_time = time.time()
    time_since_refresh = current_time - st.session_state.last_refresh_time
    
    # Only rerun if enough time has passed
    if time_since_refresh >= refresh_interval:
        st.session_state.refresh_counter += 1
        st.session_state.last_refresh_time = current_time
        st.rerun()
    
    # Initialize monitor
    monitor = IngestionMonitor()
    
    # Load comprehensive status from ingestion process
    status = monitor.load_status()
    session = status.get('progress', {})
    overall = status.get('database', {})
    resources = status.get('system', monitor._get_current_system_resources())
    errors = monitor.get_recent_errors()
    
    # Load processed ranges from status file
    processed_ranges = status.get('processed_ranges', [])
    
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
    
    # Main Dashboard
    st.markdown("---")
    
    # Processed Commit Ranges Timeline
    unique_ranges = []
    
    # Use ranges from status file (handle both old and new format)
    if processed_ranges:
        try:
            for r in processed_ranges:
                # Current simplified format with 'start' and 'end'
                if 'start' in r and 'end' in r:
                    # Calculate commits_count from index range
                    start_idx = r['start'].get('index', 0)
                    end_idx = r['end'].get('index', 0)
                    commits_count = abs(end_idx - start_idx) + 1
                    
                    unique_ranges.append({
                        'actual_first_commit_sha': r['end']['sha'],
                        'actual_first_commit_date': r['end'].get('date'),
                        'actual_last_commit_sha': r['start']['sha'],
                        'actual_last_commit_date': r['start'].get('date'),
                        'start_index': start_idx,
                        'end_index': end_idx,
                        'commits_processed': commits_count,
                        'commits': commits_count,
                        'status': r.get('status', 'completed'),
                        'range_id': r.get('id', r.get('range_id', 0))
                    })
                # Legacy format with 'start_commit' and 'end_commit'
                elif 'start_commit' in r and 'end_commit' in r:
                    commits_count = r.get('commits_count', 0)
                    unique_ranges.append({
                        'actual_first_commit_sha': r['end_commit']['sha'],
                        'actual_first_commit_date': r['end_commit']['date'],
                        'actual_last_commit_sha': r['start_commit']['sha'],
                        'actual_last_commit_date': r['start_commit']['date'],
                        'commits_processed': commits_count,
                        'commits': commits_count,
                        'status': r.get('status', 'completed'),
                        'range_id': r.get('range_id', 0)
                    })
                # Old format (fallback)
                else:
                    commits_count = r.get('commits', 0)
                    unique_ranges.append({
                        'actual_first_commit_sha': r.get('end_sha', 'N/A'),
                        'actual_first_commit_date': r.get('end_date', 'N/A'),
                        'actual_last_commit_sha': r.get('start_sha', 'N/A'),
                        'actual_last_commit_date': r.get('start_date', 'N/A'),
                        'commits_processed': commits_count,
                        'commits': commits_count,
                        'status': r.get('status', 'completed'),
                        'range_id': r.get('range_id', 0)
                    })
        except Exception as e:
            st.error(f"Error processing ranges: {e}")
            unique_ranges = []
    
    # Repository info - first commit date is the repo's first commit, not first processed
    repo_name = "Chromium"  # Default, could be made configurable
    first_commit_date = "2008-09-02"  # Chromium's first commit date
    
    st.subheader(f"📋 {repo_name} Timeline ({first_commit_date} - present)")
    
    # Show progress summary - either from completed ranges or current session
    total_docs = overall.get('total_documents', 0)
    total_target = 1755640  # Total commits in Chromium repository
    completion_percent = (total_docs / total_target * 100)
    
    # Get current range info from status
    current_range = status.get('current_range')
    
    # Check if there's an in-progress range - check both processed_ranges and unique_ranges
    in_progress_range = None
    if processed_ranges:
        for r in processed_ranges:
            if r.get('status') == 'processing':
                in_progress_range = r
                break
    
    if unique_ranges:
        total_ranges = len(unique_ranges)
        # Count only completed ranges for the summary
        completed_ranges = [r for r in unique_ranges if r.get('status') == 'completed']
        total_commits = sum(r.get('commits', 0) for r in completed_ranges)
        
        # Show in-progress range prominently
        if in_progress_range:
            in_progress_commits = in_progress_range.get('commits_count', 0)
            st.markdown(f"**{len(completed_ranges)}** completed ranges • **{total_commits:,}** commits | "
                       f"🔄 **Processing Range {in_progress_range.get('range_id')}** • **{in_progress_commits:,}** commits in progress")
            st.markdown(f"📊 **{total_docs:,}** total documents ({completion_percent:.1f}% complete)")
            
            # Show in-progress range details
            start_info = in_progress_range.get('start_commit', {})
            end_info = in_progress_range.get('end_commit', {})
            st.markdown(f"📅 **Current Range**: `{start_info.get('date', 'N/A')[:10]}` → `{end_info.get('date', 'N/A')[:10]}` • "
                       f"Start: `{start_info.get('sha', 'N/A')[:8]}` • Latest: `{end_info.get('sha', 'N/A')[:8]}`")
        else:
            st.markdown(f"**{total_ranges}** ranges • **{total_commits:,}** commits • **{total_docs:,}** documents ({completion_percent:.1f}% complete)")
    elif total_docs > 0:
        # Show current session progress
        if current_range:
            start_info = current_range.get('start_commit', {})
            end_info = current_range.get('end_commit', {})
            st.markdown(f"**In Progress** • **{current_range.get('commits_count', 0):,}** commits • **{total_docs:,}** documents ({completion_percent:.1f}% complete)")
            st.markdown(f"📅 Range: `{start_info.get('date', 'N/A')[:10]}` to `{end_info.get('date', 'N/A')[:10]}`")
            st.markdown(f"🔹 Start: `{start_info.get('sha', 'N/A')[:8]}` • End: `{end_info.get('sha', 'N/A')[:8]}`")
        else:
            st.markdown(f"**In Progress** • **{session_commits:,}** commits processed • **{total_docs:,}** documents ({completion_percent:.1f}% complete)")
            if session.get('first_commit_date') and session.get('last_commit_date'):
                st.markdown(f"📅 Processing: `{session.get('first_commit_date', 'N/A')[:10]}` to `{session.get('last_commit_date', 'N/A')[:10]}`")
    else:
        st.info("No documents processed yet")
    
    if unique_ranges:
        
        # Parse dates and create timeline data
        from datetime import datetime as dt
        import plotly.graph_objects as go
        
        timeline_data = []
        for i, cr in enumerate(unique_ranges):
            try:
                # Skip if dates are missing or invalid
                start_date_str = cr.get('actual_last_commit_date', '')
                end_date_str = cr.get('actual_first_commit_date', '')
                if not start_date_str or not end_date_str or start_date_str == 'N/A' or end_date_str == 'N/A':
                    continue
                
                start_date = dt.fromisoformat(start_date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                end_date = dt.fromisoformat(end_date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                timeline_data.append({
                    'idx': i,
                    'start_sha': cr.get('actual_last_commit_sha', '')[:8],
                    'end_sha': cr.get('actual_first_commit_sha', '')[:8],
                    'start_date': start_date,
                    'end_date': end_date,
                    'start_index': cr.get('start_index', 0),
                    'end_index': cr.get('end_index', 0),
                    'commits': cr.get('commits_processed', 0),
                    'documents': cr.get('documents_created', 0)
                })
            except (ValueError, KeyError) as e:
                continue
        
        # Sort by start date
        timeline_data.sort(key=lambda x: x['start_date'])
        
        # Find most recently updated range (latest end_date)
        most_recent_idx = -1
        if timeline_data:
            latest_range = max(timeline_data, key=lambda x: x['end_date'])
            most_recent_idx = latest_range['idx']
        
        if timeline_data:
            # Timeline uses commit index on x-axis
            # Focus starting from the first processed range
            total_commits = 1755640  # Total Chromium commits
            
            # Find the range of processed data
            if timeline_data:
                min_index = min(t['start_index'] for t in timeline_data)
                max_index = max(t['end_index'] for t in timeline_data)
                # View starts from first processed commit, ends at total commits
                view_start = min_index
                view_end = total_commits
            else:
                view_start = 0
                view_end = total_commits
            
            # Create single timeline visualization
            fig = go.Figure()
            
            # Add baseline timeline (grey line from first processed to end)
            fig.add_trace(go.Scatter(
                x=[view_start, view_end],
                y=[0, 0],
                mode='lines',
                line=dict(width=3, color='lightgrey'),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Add each processed range as a bar on single row
            for t in timeline_data:
                # Use different color for most recently updated range
                is_recent = (t['idx'] == most_recent_idx)
                line_color = '#FF6B00' if is_recent else '#00B050'  # Bright orange vs bright green
                marker_color = '#CC5500' if is_recent else '#008040'  # Darker shades for markers
                
                fig.add_trace(go.Scatter(
                    x=[t['start_index'], t['end_index']],
                    y=[0, 0],
                    mode='lines+markers',
                    line=dict(width=22, color=line_color),  # Slightly thicker for better visibility
                    marker=dict(size=12, color=marker_color, symbol='square'),  # Square markers, larger
                    hovertemplate=f"<b>Range {t['idx']+1}</b><br>" +
                                  f"Index: {t['start_index']:,} → {t['end_index']:,}<br>" +
                                  f"Start: {t['start_sha']} ({t['start_date'].strftime('%Y-%m-%d')})<br>" +
                                  f"End: {t['end_sha']} ({t['end_date'].strftime('%Y-%m-%d')})<br>" +
                                  f"Commits: {t['commits']:,}<br>" +
                                  "<extra></extra>",
                    showlegend=False
                ))
            
            # Update layout for single timeline
            fig.update_layout(
                showlegend=False,
                yaxis=dict(visible=False),
                height=250,
                xaxis=dict(
                    range=[view_start, view_end], 
                    title="Commit Index",
                    tickformat=',d'
                ),
                hovermode='closest',
                margin=dict(t=20, b=40, l=40, r=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    elif total_docs > 0 and (current_range or (session.get('first_commit_date') and session.get('last_commit_date'))):
        # Show simple timeline for in-progress session
        from datetime import datetime as dt
        import plotly.graph_objects as go
        
        try:
            # Use current_range if available, otherwise fall back to session
            if current_range:
                start_info = current_range.get('start_commit', {})
                end_info = current_range.get('end_commit', {})
                start_date_str = start_info.get('date')
                end_date_str = end_info.get('date')
                commits_count = current_range.get('commits_count', 0)
                docs_count = current_range.get('documents_created', 0)
            else:
                start_date_str = session.get('first_commit_date')
                end_date_str = session.get('last_commit_date')
                commits_count = session_commits
                docs_count = total_docs
            
            # Skip visualization if dates are invalid
            if not start_date_str or not end_date_str or start_date_str == 'N/A' or end_date_str == 'N/A':
                st.info("Current processing range dates not available")
            else:
                start_date = dt.fromisoformat(start_date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                end_date = dt.fromisoformat(end_date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                chromium_first_commit = dt(2008, 9, 1)
                chromium_latest = dt.now()
                
                fig = go.Figure()
                
                # Add baseline timeline
                fig.add_trace(go.Scatter(
                    x=[chromium_first_commit, chromium_latest],
                    y=[0, 0],
                    mode='lines',
                    line=dict(width=3, color='lightgrey'),
                    hoverinfo='skip',
                    showlegend=False
                ))
                
                # Add current processing range
                fig.add_trace(go.Scatter(
                    x=[start_date, end_date],
                    y=[0, 0],
                    mode='lines+markers',
                    line=dict(width=8, color='#FF6B00'),
                    marker=dict(size=12, color='#CC5500'),
                    name='In Progress',
                    hovertemplate='<b>In Progress</b><br>Start: %{x|%Y-%m-%d}<br>Commits: ' + f'{commits_count:,}<br>Documents: {docs_count:,}<extra></extra>'
                ))
                
                fig.update_layout(
                    height=150,
                    showlegend=False,
                    yaxis=dict(visible=False, range=[-0.5, 0.5]),
                    xaxis=dict(range=[chromium_first_commit, chromium_latest], title=None),
                    hovermode='closest',
                    margin=dict(t=20, b=40, l=40, r=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display timeline: {e}")
    elif total_docs == 0:
        st.info("No commit ranges have been processed yet. Ingestion will create timeline data as it progresses.")
    
    st.markdown("---")
    
    # Current Status
    st.subheader("⚡ Current Status")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing Rate", f"{rate:.1f} c/s")
    with col2:
        # Get the last commit processed in current session (from progress file)
        recent_date = "N/A"
        recent_time = ""
        if session.get('last_commit_date'):
            try:
                commit_date = datetime.fromisoformat(session['last_commit_date'].replace('Z', '+00:00')).replace(tzinfo=None)
                recent_date = commit_date.strftime('%Y-%m-%d')
                recent_time = commit_date.strftime('%H:%M:%S')
            except Exception:
                pass
        st.metric("Recent Processed Commit", recent_date)
        if recent_time:
            st.caption(f"Time: {recent_time}")
    with col3:
        # Show current batch progress
        current_batch = session.get('batches_completed', 0)
        total_session_commits = session.get('commits_processed', 0) - session.get('commits_processed_before_session', 0)
        st.metric("Session Progress", f"{current_batch} batches")
        st.caption(f"{total_session_commits:,} commits")
    
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
    
    # Trigger next refresh after page renders
    time.sleep(0.1)  # Small delay to ensure page renders
    if time.time() - st.session_state.last_refresh_time >= refresh_interval:
        st.rerun()


if __name__ == "__main__":
    try:
        create_dashboard()
    except Exception as e:
        print(f"Fatal error in dashboard: {e}")
        import traceback
        traceback.print_exc()
        st.error(f"Error: {e}")
        st.error(traceback.format_exc())
