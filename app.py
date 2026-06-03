import streamlit as st
import pandas as pd
from utils import CodeGenerator
import uuid 

from collections import Counter
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from rule_ordering import visualize_graph

# -------------------------
# Wrapper for button
# -------------------------
def on_generate_code():
    df = st.session_state["df"]
    rules = st.session_state.get("rules", []).values()
    mode = st.session_state.get("mode", "Detect")
    model = st.session_state["model"]
    endpoint = st.session_state["endpoint"]
    token = st.session_state["token"]
    no_tries = st.session_state["no_tries"]
    rules_order = st.session_state["rules_order"]
    code_generator = CodeGenerator()
    results = code_generator.generate_code(df, rules, mode, model, endpoint, 
                                           token, rules_order, no_tries)
    st.session_state["result_df"] = results['df']
    st.session_state["functions"] = results['functions']
    st.session_state["indices"] = results['indices']
    st.session_state["values"] = results['values']
    
    if 'optimal_path' in results:
        st.session_state["optimal_path"] = results['optimal_path']
        st.session_state["edges"] = results['edges']
        
        print('Edges:')
        print(results['optimal_path'])
        print(results['edges'])
    # print(st.session_state["values"])
    
def on_generate_rules():
    df = st.session_state["df"]
    mode = st.session_state.get("mode", "Detect")
    model = st.session_state["model"]
    endpoint = st.session_state["endpoint"]
    token = st.session_state["token"]
    code_generator = CodeGenerator()
    results = code_generator.generate_rules(df, mode,model, endpoint, token)
    st.session_state["rules"] = {str(uuid.uuid4()): rule for rule in results}
    # st.session_state["result_df"] = result_df
    # st.session_state["indices"] = indices    
    

st.set_page_config(layout="wide")

st.title("ReaRClean")

# =========================
# SIDEBAR / NAV BAR
# =========================
st.sidebar.header("Configuration")

# Model
model_input = st.sidebar.text_input(
    "Model",
    value=st.session_state.get("model", "codestral:22b")
)

# Endpoint
endpoint_input = st.sidebar.text_input(
    "Endpoint",
    value=st.session_state.get("endpoint", "http://test5.magellan2.imsi.athenarc.gr/v1/")
)

# Token (password style)
token_input = st.sidebar.text_input(
    "Token",
    type="password",
    value=st.session_state.get("token", "f3a1c4b5e60d2f9a73b8c9e4d5f1a2b6c7d8e9f0a1b2c3d4e5f6g7h8i9j0k1l2")
)

# No tries (integer)
no_tries_input = st.sidebar.number_input(
    "No Tries",
    min_value=1,
    value=st.session_state.get("no_tries", 10),
    step=1
)

# Update button
if st.sidebar.button("Update"):
    st.session_state["model"] = model_input
    st.session_state["endpoint"] = endpoint_input
    st.session_state["token"] = token_input
    st.session_state["no_tries"] = no_tries_input
    st.sidebar.success("Configuration updated!")


# =========================
# -------- UPPER HALF -----
# =========================

st.header("Input")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.session_state["uploaded_file_name"] = uploaded_file.name
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.dataframe(df, width='stretch')

with col2:
    mode = st.radio(
        "Mode",
        ["Detect", "Correct"],
        horizontal=True
    )
    st.session_state["mode"] = mode
    
    st.subheader("Rules")

    # Initialize session state
    if "rules" not in st.session_state:
        # st.session_state["rules"] = [""]
        st.session_state["rules"] = {uuid.uuid4(): ""}

    def add_rule():
        # st.session_state["rules"].append("")
        st.session_state["rules"][uuid.uuid4()] = ""

    # def remove_rule(index):
    #     st.session_state["rules"].pop(index)
    
    #     for k in list(st.session_state.keys()):
    #         if k.startswith("rule_"):
    #             del st.session_state[k]
    
    #     st.rerun()
    
    def remove_rule(key):
        st.session_state["rules"].pop(key)
    
        # for k in list(st.session_state.keys()):
        #     if k.startswith("rule_"):
        #         del st.session_state[k]
    
        # st.rerun()    
        
        
    st.button("Generate rules", on_click=on_generate_rules)

    for i, (key, rule) in enumerate(st.session_state["rules"].items()):
        cols = st.columns([5, 1])
        with cols[0]:
            #st.session_state["rules"][key] = st.text_input(
            #    f"Rule {i+1}",
            #    value=rule,
            #    key=key
            #)
            st.session_state["rules"][key] = st.text_area(
                f"Rule {i+1}",
                value=rule,
                key=key,
                height=100
            )
        with cols[1]:
            if len(st.session_state["rules"]) > 1:
                if st.button("➖", key=f"remove_{key}"):
                    remove_rule(key)
                    st.rerun()

    # for i, rule in enumerate(st.session_state["rules"]):
    #     cols = st.columns([5, 1])
    #     with cols[0]:
    #         st.session_state["rules"][i] = st.text_input(
    #             f"Rule {i+1}",
    #             value=rule,
    #             key=f"rule_{i}"
    #         )
    #     with cols[1]:
    #         if len(st.session_state["rules"]) > 1:
    #             if st.button("➖", key=f"remove_{i}"):
    #                 remove_rule(i)
    #                 st.rerun()

    st.button("➕ Add Rule", on_click=add_rule)
    
    if mode == 'Correct':
        rules_order = st.radio(
            "Rules Ordering",
            ["Linear", "Advanced"],
            horizontal=True
        )
        st.session_state["rules_order"] = rules_order
    else:
        st.session_state["rules_order"] = "Linear"

    st.button("Generate code", on_click=on_generate_code)

    # Backend list of rules (cleaned)
    # rules_list = [
    #     r for r in st.session_state["rules"] if r.strip() != ""
    # ]    


st.divider()


# =========================
# -------- LOWER HALF -----
# =========================

st.header("Analysis")

col_code, col_rules, col_results = st.columns([1, 1, 1])



# -------------------------
# 2️⃣ CODE + RUN COLUMN
# -------------------------
with col_code:

    st.subheader("Generated Code")

    # Example auto-generated code stub
    if "functions" in st.session_state:
        functions = st.session_state["functions"]
        st.info(f"Generated {len(functions)} function(s)")
    else:
        functions = []
    generated_code = "\n".join(functions)
    st.code(generated_code, language="python")

# -------------------------
# 1️⃣ RULE INPUT COLUMN
# -------------------------
with col_rules:
    # st.subheader("Rules Coverage")
    st.markdown("### Rules Analysis")
    
    st.markdown("#### Rules Coverage")
    
    if 'values' in st.session_state:
        print(st.session_state["values"])
        df = pd.DataFrame([dict(Counter([res['column'] for res in rule])) for rule in st.session_state["values"]])
        df = df.fillna(0).astype(int)
        df.index = [f'Rule-{no}' for no in range(df.shape[0])]
        
        fig = px.imshow(
            df,
            text_auto=True,
            color_continuous_scale="YlGnBu",
            labels=dict(x="Columns", y="Rules", color="Errors"),
        )
        
        fig.update_layout(
            title="Rules Coverage Heatmap",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    if 'optimal_path' in st.session_state:
        
        # st.subheader("Rules Ordering")
        st.markdown("#### Rules Ordering")
        selected_edges = st.session_state["optimal_path"]
        total_edges = st.session_state["edges"]
        
        # html = visualize_graph(total_edges, selected_edges)
        
        # st.plotly_chart(fig, use_container_width=True)
        import streamlit as st
        from st_cytoscape import cytoscape
        
        elements, stylesheet = visualize_graph(total_edges, selected_edges)
        
        cytoscape(
            elements,
            stylesheet,
            layout={"name": "cose"},  # physics layout
            key="graph"
        )
            
    
    
# -------------------------
# 3️⃣ RESULTS COLUMN
# -------------------------
with col_results:

    st.subheader("Results")
    # st.write("Results will appear here.")
    if "result_df" in st.session_state:
        result_df = st.session_state["result_df"]
        
        pd.set_option("styler.render.max_elements", result_df.shape[0]*result_df.shape[1])
        mask = pd.DataFrame("", index=result_df.index, columns=result_df.columns)

        indices = st.session_state['indices']
        for idx, col in indices:
            if idx in mask.index and col in mask.columns:
                mask.loc[idx, col] = "background-color: #262730"
        
        styled_df = result_df.style.apply(lambda _: mask, axis=None)
        
        st.dataframe(styled_df, width='stretch')
        
        mode = st.session_state["mode"]
        if mode == 'Detect':
            st.info('Detected {} errors'.format(len(indices)))
        elif mode == 'Impute':
            st.info('Corrected {} errors'.format(len(indices)))
        
