"""Esnaf Dashboard — Streamlit-based monitoring for the trading agent.

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="Esnaf — Trading Agent",
    page_icon="📊",
    layout="wide",
)

DB_PATH = Path("esnaf.db")


@st.cache_resource
def get_db():
    """Connect to the SQLite database (read-only)."""
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def load_trades(conn) -> pd.DataFrame:
    """Load trade history."""
    try:
        return pd.read_sql_query(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 200", conn
        )
    except Exception:
        return pd.DataFrame()


def load_decisions(conn) -> pd.DataFrame:
    """Load recent decision log."""
    try:
        return pd.read_sql_query(
            "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 100", conn
        )
    except Exception:
        return pd.DataFrame()


def load_regime_history(conn) -> pd.DataFrame:
    """Load regime classification history."""
    try:
        return pd.read_sql_query(
            "SELECT * FROM regime_history ORDER BY timestamp DESC LIMIT 100", conn
        )
    except Exception:
        return pd.DataFrame()


def load_portfolio_snapshots(conn) -> pd.DataFrame:
    """Load portfolio value over time."""
    try:
        return pd.read_sql_query(
            "SELECT * FROM portfolio_snapshots ORDER BY timestamp", conn
        )
    except Exception:
        return pd.DataFrame()


def load_memories(conn) -> pd.DataFrame:
    """Load agent memory entries."""
    try:
        return pd.read_sql_query(
            "SELECT * FROM memory ORDER BY timestamp DESC LIMIT 50", conn
        )
    except Exception:
        return pd.DataFrame()


# === Main App ===

st.title("📊 Esnaf — Crypto Trading Agent")
st.caption("Regime-Aware • LLM-Powered • Paper Trading")

conn = get_db()

if conn is None:
    st.warning("No database found at `esnaf.db`. Start the agent first to create it.")
    st.stop()

# Auto-refresh
st.sidebar.header("Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    import time
    time.sleep(0.1)  # Let Streamlit render first
    st.rerun()  # Will re-run after the sleep in the browser

# === Top Metrics ===

trades_df = load_trades(conn)
decisions_df = load_decisions(conn)
regimes_df = load_regime_history(conn)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Trades", len(trades_df))

with col2:
    if not trades_df.empty and "value" in trades_df.columns:
        total_volume = trades_df["value"].sum()
        st.metric("Total Volume", f"${total_volume:,.2f}")
    else:
        st.metric("Total Volume", "$0")

with col3:
    if not trades_df.empty and "fee" in trades_df.columns:
        total_fees = trades_df["fee"].sum()
        st.metric("Total Fees", f"${total_fees:,.2f}")
    else:
        st.metric("Total Fees", "$0")

with col4:
    st.metric("Decisions Logged", len(decisions_df))

with col5:
    if not regimes_df.empty:
        latest_regime = regimes_df.iloc[0]
        st.metric("Current Regime", latest_regime.get("regime", "N/A"))
    else:
        st.metric("Current Regime", "N/A")

st.divider()

# === Tabs ===

tab_trades, tab_decisions, tab_regimes, tab_memory, tab_performance = st.tabs(
    ["📈 Trades", "🧠 Decisions", "🌐 Regimes", "💭 Memory", "📊 Performance"]
)

# --- Trades Tab ---
with tab_trades:
    st.subheader("Trade History")
    if trades_df.empty:
        st.info("No trades yet. Start the agent in paper trading mode.")
    else:
        # Color-code actions
        st.dataframe(
            trades_df[["timestamp", "symbol", "action", "side", "price", "quantity", "value", "fee", "regime", "confidence"]],
            use_container_width=True,
            hide_index=True,
        )

# --- Decisions Tab ---
with tab_decisions:
    st.subheader("Agent Decision Log")
    if decisions_df.empty:
        st.info("No decisions recorded yet.")
    else:
        for _, row in decisions_df.head(20).iterrows():
            with st.expander(
                f"{'✅' if row.get('validation_passed') else '❌'} "
                f"{row.get('action', 'N/A')} — {row.get('timestamp', '')[:19]} "
                f"({row.get('regime', 'unknown')} @ {row.get('regime_confidence', 0):.0%})"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Regime:**", row.get("regime", "N/A"))
                    st.write("**Confidence:**", f"{row.get('regime_confidence', 0):.0%}")
                    st.write("**Regime Reasoning:**", row.get("regime_reasoning", "N/A"))
                with col_b:
                    st.write("**Action:**", row.get("action", "N/A"))
                    st.write("**Model:**", row.get("model_used", "N/A"))
                    st.write("**Validation:**", "Passed" if row.get("validation_passed") else f"Rejected: {row.get('validation_reason', '')}")

                if row.get("reasoning"):
                    st.write("**Reasoning:**", row["reasoning"])

# --- Regimes Tab ---
with tab_regimes:
    st.subheader("Regime Classification History")
    if regimes_df.empty:
        st.info("No regime data yet.")
    else:
        # Regime distribution chart
        regime_counts = regimes_df["regime"].value_counts()
        st.bar_chart(regime_counts)

        # Confidence over time
        regimes_df["timestamp"] = pd.to_datetime(regimes_df["timestamp"])
        chart_df = regimes_df[["timestamp", "confidence", "regime"]].copy()
        st.line_chart(chart_df.set_index("timestamp")["confidence"])

        st.dataframe(
            regimes_df[["timestamp", "regime", "confidence", "strategy", "reasoning"]].head(30),
            use_container_width=True,
            hide_index=True,
        )

# --- Memory Tab ---
with tab_memory:
    st.subheader("Agent Memory (Lessons & Patterns)")
    memories_df = load_memories(conn)
    if memories_df.empty:
        st.info("No memories stored yet.")
    else:
        for _, row in memories_df.iterrows():
            icon = "💡" if row.get("entry_type") == "lesson" else "🔍"
            st.write(f"{icon} **[{row.get('entry_type', 'unknown')}]** {row.get('content', '')}")
            if row.get("metadata"):
                try:
                    meta = json.loads(row["metadata"])
                    st.caption(f"Cycle: {meta.get('cycle_id', 'N/A')} | Regime: {meta.get('regime', 'N/A')}")
                except (json.JSONDecodeError, TypeError):
                    pass

# --- Performance Tab ---
with tab_performance:
    st.subheader("Performance Analytics")

    if trades_df.empty:
        st.info("Need trade data for performance analysis.")
    else:
        # Win/Loss analysis
        if "value" in trades_df.columns:
            buy_trades = trades_df[trades_df["action"] == "BUY"]
            sell_trades = trades_df[trades_df["action"] == "SELL"]

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Buy Orders", len(buy_trades))
            with col_b:
                st.metric("Sell Orders", len(sell_trades))
            with col_c:
                st.metric("Hold Decisions",
                    len(decisions_df[decisions_df["action"] == "HOLD"]) if not decisions_df.empty else 0
                )

        # Regime accuracy (if we have was_accurate data)
        if not regimes_df.empty and "was_accurate" in regimes_df.columns:
            accurate = regimes_df[regimes_df["was_accurate"].notna()]
            if not accurate.empty:
                accuracy = accurate["was_accurate"].mean()
                st.metric("Regime Accuracy", f"{accuracy:.0%}")

        # Validation pass rate
        if not decisions_df.empty and "validation_passed" in decisions_df.columns:
            pass_rate = decisions_df["validation_passed"].mean()
            st.metric("Validation Pass Rate", f"{pass_rate:.0%}")

# Footer
st.divider()
st.caption("Esnaf v0.1 — Paper Trading Mode | Data refreshes on page reload")
