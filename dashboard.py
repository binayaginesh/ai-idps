import streamlit as st
import pandas as pd
import time
import os
import re
from collections import deque
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="AI-IDPS Real-Time Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 14px;
        color: #B0BEC5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .attack-alert {
        color: #F44336 !important;
    }
    .status-safe { color: #4CAF50; font-weight: bold; font-size: 24px; }
    .status-warning { color: #FF9800; font-weight: bold; font-size: 24px; }
    .status-danger { color: #F44336; font-weight: bold; font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
IDPS_LOG = LOG_DIR / "idps.log"
ATTACK_LOG = LOG_DIR / "attacks.log"

# Parsing Regex
# Example: 2026-02-25 06:44:55 [WARNING] realtime_monitor — ATTACK DETECTED 192.168.1.100:54321 → 10.0.0.5:80 (BENIGN 81.00%)
_RE_ATTACK = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*?ATTACK DETECTED\s+"
    r"(?P<src>[\d\.]+:\d+)\s+[→|->]+\s+(?P<dst>[\d\.]+:\d+)\s+"
    r"\((?P<class>[\w\s-]+)\s+(?P<prob>[\d\.]+%)\)"
)

# Example: 2026-02-25 06:43:55 [INFO] realtime_monitor — BENIGN flow 192.168.1.100 → 10.0.0.5
_RE_BENIGN = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*?BENIGN flow\s+"
    r"(?P<src>[\d\.]+)\s+[→|->]+\s+(?P<dst>[\d\.]+)"
)

# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1)  # Cache for 1 second to prevent excessive disk reads
def load_log_data():
    """Reads logs and parses them into pandas DataFrames."""
    benign_count = 0
    total_flows = 0
    attacks = []
    
    # Parse Attacks Log
    if ATTACK_LOG.exists():
        with open(ATTACK_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Tail the last 1000 lines for performance
            for line in lines[-1000:]:
                m = _RE_ATTACK.search(line)
                if m:
                    attacks.append({
                        "Timestamp": m.group("ts"),
                        "Source": m.group("src"),
                        "Destination": m.group("dst"),
                        "Classification": m.group("class").strip(),
                        "Confidence": m.group("prob"),
                    })
                    total_flows += 1

    # Parse IDPS Log for Benign traffic counts (rough estimate from tail)
    if IDPS_LOG.exists():
        with open(IDPS_LOG, "r", encoding="utf-8") as f:
            # For massive logs, we just read the end or do a wc -l equivalent. 
            # In python, we can read chunks. But for simplicity here, we'll read lines.
            lines = f.readlines()[-5000:] 
            for line in lines:
                if "BENIGN flow" in line:
                    benign_count += 1
                if "ATTACK DETECTED" in line or "BENIGN flow" in line:
                    # just incrementing a rough total for the dashboard timeline
                    pass
    
    # Try to make sense of total vs benign
    # If using grep/wc on linux it's faster, but this is pure python
    # We will compute total_flows as attacks + benign
    total_flows = benign_count + len(attacks)
    
    df_attacks = pd.DataFrame(attacks)
    if not df_attacks.empty:
        df_attacks["Timestamp"] = pd.to_datetime(df_attacks["Timestamp"])
        # Sort descending
        df_attacks = df_attacks.sort_values(by="Timestamp", ascending=False)
        
    return df_attacks, benign_count, total_flows


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

def main():
    st.title("🛡️ AI-IDPS Real-Time Monitor")
    st.markdown("Monitoring interface for Phase 2 AI-Based Intrusion Detection and Prevention System.")
    
    # Load Data
    df_attacks, benign_count, total_flows = load_log_data()
    attack_count = len(df_attacks) if not df_attacks.empty else 0
    total_flows_estimate = benign_count + attack_count
    
    # System Status logic
    status_color = "status-safe"
    status_text = "SECURE"
    if attack_count > 0:
        # Check if attacks are recent (last 60 seconds)
        if not df_attacks.empty:
            latest_attack = df_attacks.iloc[0]["Timestamp"]
            now = pd.Timestamp.now()
            if (now - latest_attack).total_seconds() < 60:
                 status_color = "status-danger"
                 status_text = "UNDER ATTACK"
            else:
                 status_color = "status-warning"
                 status_text = "ELEVATED (Recent Attacks)"

    # Top KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">System Status</div><div class="{status_color}">{status_text}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Flows (Recent)</div><div class="metric-value">{total_flows_estimate}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Benign Flows</div><div class="metric-value">{benign_count}</div></div>', unsafe_allow_html=True)
    with col4:
         st.markdown(f'<div class="metric-card"><div class="metric-label">Attacks Detected</div><div class="metric-value attack-alert">{attack_count}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Charts & Tables Row
    col_table, col_chart = st.columns([2, 1])
    
    with col_table:
        st.subheader("🚨 Recent Attack Alerts")
        if df_attacks.empty:
            st.info("No attacks detected in the current log window.")
        else:
            # Display recent 50 attacks
            st.dataframe(
                df_attacks.head(50), 
                use_container_width=True,
                column_config={
                    "Timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                    "Confidence": st.column_config.TextColumn()
                },
                hide_index=True
            )
            
    with col_chart:
        st.subheader("📊 Traffic Distribution")
        if total_flows_estimate > 0:
            pie_data = pd.DataFrame({
                "Count": [benign_count, attack_count],
                "Type": ["Benign", "Attack"]
            })
            # Use Streamlit's native charting via index
            pie_data.set_index("Type", inplace=True)
            st.bar_chart(pie_data, color=["#4CAF50", "#F44336"])
        else:
            st.write("Awaiting traffic data...")
            
        if not df_attacks.empty:
            st.subheader("Attack Classifications")
            class_counts = df_attacks["Classification"].value_counts()
            st.bar_chart(class_counts, color="#F44336")

    # Auto-refresh mechanism
    st.empty()
    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    main()
