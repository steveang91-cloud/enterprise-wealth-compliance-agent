
import streamlit as st
import sqlite3
import os
import re
import json
import sqlparse
from google import genai
from google.genai import types

# Set Page Config
st.set_page_config(page_title="Enterprise Wealth Analyst App", layout="wide")

# --- 1. CONFIGURATION & SECRETS RESOLUTION ---
try:
    # Fallback to streamlit secrets first, then colab userdata
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        from google.colab import userdata
        api_key = userdata.get('GOOGLE_API_KEY')
except Exception:
    api_key = os.getenv("GOOGLE_API_KEY", "")

# Initialise API Client
if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = genai.Client()

# --- 2. MODEL SECURITY GUARDRAILS ---
def is_sql_mutation(sql_query: str) -> bool:
    try:
        parsed = sqlparse.parse(sql_query)
        for statement in parsed:
            if statement.get_type() in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE"]:
                return True
            for token in statement.flatten():
                if token.ttype in sqlparse.tokens.Keyword and token.value.upper() in [
                    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "GRANT"
                ]:
                    return True
        return False
    except Exception:
        return True

def before_model_guardrail(prompt: str) -> str:
    malicious_patterns = [
        r"ignore\s+.*compliance",
        r"override\s+.*policy",
        r"bypass\s+.*limit",
        r"approve\s+.*immediately\s+without"
    ]
    for pattern in malicious_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            raise ValueError("🚨 Security Block: Prompt contains override/injection keywords.")
    
    # Scrub PII
    scrubbed = re.sub(r"\b[A-Z]\d{7}[A-Z]\b", "[REDACTED_ID]", prompt, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[REDACTED_CARD]", scrubbed)
    scrubbed = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]", scrubbed)
    return scrubbed

# --- 3. DYNAMIC METADATA DISCOVERY & MCP DATABASE TOOLS ---
def mcp_list_tables() -> str:
    try:
        conn = sqlite3.connect("file:enterprise.db?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return f"Tables: {tables}"
    except Exception as e:
        return str(e)

def mcp_describe_table(table_name: str) -> str:
    try:
        conn = sqlite3.connect("file:enterprise.db?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]
        conn.close()
        return f"Columns: {', '.join(columns)}"
    except Exception as e:
        return str(e)

def mcp_read_query(sql_query: str) -> str:
    if is_sql_mutation(sql_query):
        return "🚨 Security Block: Mutation commands are prohibited."
    try:
        conn = sqlite3.connect("file:enterprise.db?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        results = [dict(zip(columns, row)) for row in rows]
        return str(results)
    except Exception as e:
        return str(e)

def mcp_read_policy(filename: str) -> str:
    base_dir = os.path.abspath("policies")
    target_path = os.path.abspath(os.path.join(base_dir, os.path.basename(filename)))
    if not target_path.startswith(base_dir):
        return "🚨 Security Violation: Path traversal blocked."
    try:
        if not os.path.exists(target_path):
            return f"File '{filename}' not found."
        with open(target_path, 'r') as f:
            return f.read()
    except Exception as e:
        return str(e)

# --- 4. APP LAYOUT & INTERACTIVE DASHBOARD ---
st.title("🏛️ Enterprise Client Relations & Compliance Dashboard")
st.markdown("Production-grade High-Value Fee Waiver & Credit Limit Request Processor.")

# Sidebar Metrics & Evolution Simulator
with st.sidebar:
    st.header("🔒 Guardrail & Token Telemetry")
    if 'token_cost' not in st.session_state:
        st.session_state.token_cost = 0.0
        st.session_state.tokens_used = 0

    st.metric(label="Accumulated Session Cost", value=f"${st.session_state.token_cost:.6f} USD")
    st.metric(label="Tokens Consumed", value=f"{st.session_state.tokens_used:,}")
    
    st.markdown("--- ")
    st.header("⚙️ Database Schema Evolution")
    if st.button("Simulate Zero-Code Migration (Add 'premium_travel_tier')"):
        conn = sqlite3.connect('enterprise.db')
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE clients ADD COLUMN premium_travel_tier TEXT DEFAULT 'Silver';")
            cursor.execute("UPDATE clients SET premium_travel_tier = 'Gold' WHERE client_id = '9941';")
            conn.commit()
            st.success("Dynamic Migration Complete!")
        except sqlite3.OperationalError:
            st.warning("Schema already updated (premium_travel_tier exists).")
        finally:
            conn.close()

# Main Prompt Input Panel
prompt_input = st.text_area(
    "Enter Client request note (RM or client request details):",
    value="Client Sarah Lim (Account #9941) is traveling to London tomorrow and wants a temporary credit limit increase to RM 50,000 plus a foreign transaction fee waiver.",
    height=120
)

if st.button("Run Compliance & Credit Evaluation"):
    if not prompt_input.strip():
        st.error("Please enter a prompt.")
    else:
        st.markdown("### Processing Compliance Pipeline...")
        
        # Pre-model guardrail check
        try:
            safe_prompt = before_model_guardrail(prompt_input)
            st.info("🛡️ Guardrail Check: Input sanitized successfully.")
        except ValueError as err:
            st.error(str(err))
            st.stop()
            
        # Invoke the compliance expert agent
        try:
            sys_instruction = (
                "You are an expert high-value Wealth Management & Compliance Agent.\n"
                "Your goal is to evaluate temporary travel limits and fee waivers.\n"
                "When evaluating requests, you must always perform these sequential checks:\n"
                "1. Query client's core banking details via SQLite tools to determine tier and balance.\n"
                "2. Query regional hazard and travel rules from the branch risk data.\n"
                "3. Consult the compliance policies handbook using the policy reading tools.\n"
                "4. Draft a formal, professional reply ensuring absolute compliance.\n\n"
                "Regulatory strict rules:\n"
                "- POLICY HANDBOOK PRECEDENCE: Global policy rules and caps defined in the policy handbook "
                "(such as the RM 10,000 cap for Standard Retail clients) take absolute precedence over regional "
                "branch risk limits or auto-approval thresholds. You MUST reject or limit any requests exceeding these handbook caps.\n"
                "- You MUST explicitly include the mandatory 'FX Conversion Risk Disclosure Statement' if fee waivers are processed.\n"
                "- You MUST enforce a 30-day expiry limit on any approved temporary credit increase.\n"
                "- If the approved limit is strictly above RM 30,000, you MUST set the execution_mode value "
                "to 'REQUIRE_BRANCH_MANAGER_SIGN_OFF' in your final operational status JSON block.\n"
            )
            
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=safe_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    tools=[mcp_list_tables, mcp_describe_table, mcp_read_query, mcp_read_policy]
                )
            )
            
            # Update metrics
            if response.usage_metadata:
                in_tok = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                out_tok = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                total_tok = in_tok + out_tok
                st.session_state.tokens_used += total_tok
                cost = (in_tok * (0.075 / 1_000_000)) + (out_tok * (0.30 / 1_000_000))
                st.session_state.token_cost += cost
                st.experimental_rerun()
                
            st.success("Evaluation Concluded.")
            st.markdown(response.text)
            
        except Exception as e:
            st.error(f"Agent Encountered an Issue: {str(e)}")
