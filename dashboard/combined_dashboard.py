import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import joblib
import os

DATA_DIR = 'data'
MODEL_DIR = 'models'
FEATURE_COLS = ['mean_login_hour', 'mean_logout_hour', 'files_per_day', 'usb_per_day', 'emails_per_day', 'out_of_session_access', 'degree_centrality', 'betweenness_centrality', 'keyword_flag', 'subject_len', 'sentiment']

st.set_page_config(layout="wide")
st.title('AI-Powered Insider Threat Detection: Combined Dashboard')

# Load data
def load_all_data():
    features = pd.read_csv(os.path.join(DATA_DIR, 'merged_features.csv'))
    scores = pd.read_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'))
    file_access = pd.read_csv(os.path.join(DATA_DIR, 'file_access.csv'), parse_dates=['access_time'])
    usb_usage = pd.read_csv(os.path.join(DATA_DIR, 'usb_usage.csv'), parse_dates=['plug_time', 'unplug_time'])
    return features, scores, file_access, usb_usage

features, scores, file_access, usb_usage = load_all_data()

# Keep datasets in session state so new records can be appended and re-scored
if 'features' not in st.session_state:
    st.session_state['features'] = features
if 'scores' not in st.session_state:
    st.session_state['scores'] = scores
if 'custom_users' not in st.session_state:
    st.session_state['custom_users'] = pd.DataFrame()

features = st.session_state['features']
scores = st.session_state['scores']
df = pd.merge(features, scores, on='user')

# Score new custom records using trained models + scaler
def score_custom_users():
    if st.session_state['custom_users'].empty:
        return
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    iso = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    svm = joblib.load(os.path.join(MODEL_DIR, 'oneclass_svm.pkl'))
    auto = joblib.load(os.path.join(MODEL_DIR, 'autoencoder.pkl'))

    all_features = pd.concat([
        st.session_state['features'][FEATURE_COLS],
        st.session_state['custom_users'][FEATURE_COLS]
    ], ignore_index=True)
    X_scaled = scaler.transform(all_features)

    iso_scores = -iso.score_samples(X_scaled)
    svm_scores = -svm.decision_function(X_scaled)
    auto_scores = np.mean((X_scaled - auto.predict(X_scaled))**2, axis=1)

    n_orig = len(st.session_state['features'])
    iso_s = iso_scores[n_orig:]
    svm_s = svm_scores[n_orig:]
    auto_s = auto_scores[n_orig:]
    custom_anom = np.maximum.reduce([iso_s, svm_s, auto_s])
    is_rt = (custom_anom >= RED_TEAM_BASELINE).astype(int)
    st.session_state['custom_users'] = st.session_state['custom_users'].assign(is_red_team=is_rt)
    new_scores = pd.DataFrame({
        'user': st.session_state['custom_users']['user'].values,
        'is_red_team': is_rt,
        'isolation_forest': iso_s,
        'oneclass_svm': svm_s,
        'autoencoder': auto_s
    })
    st.session_state['scores'] = pd.concat([st.session_state['scores'], new_scores], ignore_index=True)
    st.session_state['features'] = pd.concat([st.session_state['features'], st.session_state['custom_users']], ignore_index=True)

def reset_custom():
    st.session_state['features'] = features_orig
    st.session_state['scores'] = scores_orig
    st.session_state['custom_users'] = pd.DataFrame()

features_orig, scores_orig, _, _ = load_all_data()

# Auto-flag baseline: a custom user is marked red team if its anomaly score
# is at least as high as the least anomalous known red-team user.
def red_team_baseline():
    rt = scores_orig[scores_orig['is_red_team'] == 1]
    anom = rt[['isolation_forest', 'oneclass_svm', 'autoencoder']].max(axis=1)
    return float(anom.min())
RED_TEAM_BASELINE = red_team_baseline()

# Prepare node attributes for graph
def get_node_attrs():
    attrs = {}
    for _, row in scores.iterrows():
        anomaly = max(row['isolation_forest'], row['oneclass_svm'], row['autoencoder'])
        red_team = row['is_red_team']
        attrs[row['user']] = {
            'anomaly': anomaly,
            'red_team': red_team,
            'high_risk': (anomaly > 1.0) or (red_team == 1)
        }
    return attrs
attrs = get_node_attrs()

# Build full graph
def build_graph(custom_df=pd.DataFrame()):
    G = nx.Graph()
    for _, row in file_access.iterrows():
        G.add_edge(row['user'], row['file'], type='access')
    for _, row in usb_usage.iterrows():
        G.add_edge(row['user'], row['device'], type='usb')
    # Connect custom users to the SAME real files/devices real users access,
    # so they join the main connected component instead of isolated islands.
    if not custom_df.empty:
        real_files = np.asarray(file_access['file'].unique())
        real_devices = np.asarray(usb_usage['device'].unique())
        rng = np.random.default_rng(42)
        for _, row in custom_df.iterrows():
            user = row['user']
            n_f = int(round(row.get('files_per_day', 0)))
            n_u = int(round(row.get('usb_per_day', 0)))
            if len(real_files):
                for f in rng.choice(real_files, size=min(n_f, len(real_files)), replace=False):
                    G.add_edge(user, f, type='access')
            if len(real_devices):
                for d in rng.choice(real_devices, size=min(n_u, len(real_devices)), replace=False):
                    G.add_edge(user, d, type='usb')
    return G
G = build_graph(st.session_state.get('custom_users', pd.DataFrame()))

# At-risk subgraph
def get_at_risk_subgraph(G, attrs):
    high_risk_nodes = {n for n, v in attrs.items() if v['high_risk']}
    connected_nodes = set()
    for node in high_risk_nodes:
        connected_nodes.add(node)
        connected_nodes.update(G.neighbors(node))
    return G.subgraph(connected_nodes).copy()

# Tabs
anomaly_tab, user_tab, graph_tab, data_tab, how_tab = st.tabs(["Anomaly Table", "User Detail", "At-Risk Graph", "Data Input", "How Does It Work?"])

with anomaly_tab:
    st.header('User Anomaly Scores')
    score_method = st.selectbox('Select Model', ['isolation_forest', 'oneclass_svm', 'autoencoder'], key='score_method')
    df['Red Team'] = df['is_red_team_x'].apply(lambda x: '🚩' if x == 1 else '') if 'is_red_team_x' in df.columns else df['is_red_team'].apply(lambda x: '🚩' if x == 1 else '')
    df['rank'] = df[score_method].rank(ascending=False)
    df_sorted = df.sort_values(score_method, ascending=False)
    cols = ['user', 'Red Team', score_method, 'rank'] + [c for c in df.columns if c not in ['user', score_method, 'rank', 'Red Team']]
    st.dataframe(df_sorted[cols], height=500)
    st.subheader('Top 5 Anomalous Users')
    top5 = df_sorted.head(5)
    st.bar_chart(top5.set_index('user')[score_method])

with user_tab:
    st.header('User Detail')
    selected_user = st.selectbox('Select User', df_sorted['user'], key='user_detail')
    user_row = df_sorted[df_sorted['user'] == selected_user].iloc[0]
    st.write('**Red Team:**', '🚩' if user_row['Red Team'] else 'No')
    st.write('**Features:**')
    st.json({k: user_row[k] for k in ['mean_login_hour', 'mean_logout_hour', 'files_per_day', 'usb_per_day', 'emails_per_day', 'out_of_session_access', 'degree_centrality', 'betweenness_centrality', 'keyword_flag', 'subject_len', 'sentiment'] if k in user_row})
    st.write('**Anomaly Scores:**')
    st.json({k: user_row[k] for k in ['isolation_forest', 'oneclass_svm', 'autoencoder']})

with graph_tab:
    st.header('At-Risk Nodes and Their Connections')
    subG = get_at_risk_subgraph(G, attrs)
    net = Network(height='900px', width='100%', notebook=False, bgcolor='#222222', font_color='white')
    net.barnes_hut(gravity=-2000, central_gravity=0.1, spring_length=200, spring_strength=0.01, damping=0.85, overlap=1)
    net.set_options('''
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "fit": true, "iterations": 2500, "updateInterval": 50},
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.01,
          "damping": 0.85,
          "avoidOverlap": 1
        }
      }
    }
    ''')
    for node in subG.nodes():
        if node in attrs:
            score = attrs[node]['anomaly']
            red = attrs[node]['red_team']
            color = 'red' if red else ('orange' if score > 1.5 else 'yellow' if score > 1.0 else 'lightblue')
            size = 30 if red else (20 if score > 1.5 else 15 if score > 1.0 else 10)
            title = f"User: {node}<br>Anomaly Score: {score:.2f}<br>Red Team: {'Yes' if red else 'No'}"
        elif str(node).startswith('file'):
            color = 'green'
            size = 8
            title = f"File: {node}"
        elif str(node).startswith('usb'):
            color = 'purple'
            size = 8
            title = f"Device: {node}"
        else:
            color = 'gray'
            size = 8
            title = str(node)
        net.add_node(node, label=str(node), color=color, size=size, title=title)
    for edge in subG.edges(data=True):
        net.add_edge(edge[0], edge[1], color='gray' if edge[2]['type']=='access' else 'purple')
    net.save_graph('dashboard/graph.html')
    st.components.v1.html(open('dashboard/graph.html', 'r', encoding='utf-8').read(), height=900, scrolling=False)

with data_tab:
    st.header('Input Records & Re-Score')
    st.markdown('Enter up to 5 user records with behavioral features. Click **Submit & Score** to append them to the dataset and re-run the anomaly detection models. New records appear in the Anomaly Table, Top 5 chart, and At-Risk Graph.')

    feature_labels = {
        'mean_login_hour': ('Mean Login Hour (0-24)', 8.0, 0.0, 24.0),
        'mean_logout_hour': ('Mean Logout Hour (0-24)', 16.0, 0.0, 24.0),
        'files_per_day': ('Files Accessed / Day', 3.5, 0.0, 100.0),
        'usb_per_day': ('USB Plugs / Day', 1.0, 0.0, 50.0),
        'emails_per_day': ('Emails Sent / Day', 2.5, 0.0, 50.0),
        'out_of_session_access': ('Out-of-Session Access Count', 70, 0, 500),
        'degree_centrality': ('Degree Centrality (0-1)', 0.65, 0.0, 1.0),
        'betweenness_centrality': ('Betweenness Centrality (0-1)', 0.03, 0.0, 1.0),
        'keyword_flag': ('Email Keyword Flag Rate (0-1)', 0.4, 0.0, 1.0),
        'subject_len': ('Avg Email Subject Length', 9.5, 0.0, 100.0),
        'sentiment': ('Email Sentiment', 0.0, -1.0, 1.0),
    }

    with st.form('record_form'):
        cols = st.columns(5)
        records = []
        for i in range(5):
            with cols[i]:
                st.subheader(f'Record {i+1}')
                user_name = st.text_input('User Name', value=f'custom_user{i+1}', key=f'name_{i}')
                rec = {'user': user_name}
                for feat, (label, default, lo, hi) in feature_labels.items():
                    step = 0.01 if isinstance(default, float) and hi <= 1.0 else (0.1 if isinstance(default, float) else 1)
                    rec[feat] = st.number_input(label, value=default, min_value=lo, max_value=hi, step=step, key=f'{feat}_{i}')
                records.append(rec)
        submitted = st.form_submit_button('Submit & Score')

    if submitted:
        new_df = pd.DataFrame(records)
        # Drop records whose user name is empty
        new_df = new_df[new_df['user'].str.strip() != '']
        if new_df.empty:
            st.warning('Please enter at least one record with a user name.')
        else:
            dup = new_df['user'].isin(st.session_state['features']['user'].values)
            if dup.any():
                st.warning('One or more user names already exist in the dataset. Custom users skipped.')
                new_df = new_df[~dup]
            if not new_df.empty:
                st.session_state['custom_users'] = new_df.reset_index(drop=True)
                score_custom_users()
                st.success(f'Added {len(new_df)} record(s). Re-scored with all models. See the other tabs.')
                st.rerun()

    if not st.session_state['custom_users'].empty:
        st.subheader('Custom Records Added')
        disp = st.session_state['custom_users'].copy()
        merged = pd.merge(disp, st.session_state['scores'], on='user', how='left')
        st.dataframe(merged[['user', 'isolation_forest', 'oneclass_svm', 'autoencoder']], height=200)
        if st.button('Reset to Original Dataset'):
            reset_custom()
            st.success('Reverted to original dataset.')
            st.rerun()

with how_tab:
    st.header('How Does It Work?')
    st.markdown('''
## System Overview
This system detects insider threats by analyzing user behavior, system access, and relationships using advanced machine learning and graph analysis techniques.

---

### 1. **Data Simulation & Feature Engineering**
- **Simulated Logs:** The system generates synthetic logs for user logins, file access, USB usage, and emails, mimicking real organizational activity.
- **Feature Engineering:** Extracts features such as:
    - Login/logout patterns (mean hours, frequency)
    - File/USB/email activity rates
    - Out-of-session file access
    - Graph centrality (degree, betweenness)
    - NLP features from email subjects (keyword flags, length)

---

### 2. **Anomaly Detection Algorithms**
- **Isolation Forest**
    - *Mathematics:* Randomly partitions data to isolate points. Anomalies are isolated faster (shorter average path length in trees).
    - *Computer Science:* Ensemble of binary trees; each tree splits on random features/values. The anomaly score is based on the average path length to isolate a sample.
- **One-Class SVM**
    - *Mathematics:* Finds a boundary in feature space that encloses most data (support vectors). Points outside are anomalies.
    - *Computer Science:* Uses kernel methods (e.g., RBF) to map data to high-dimensional space and find a maximal margin hyperplane.
- **Autoencoder**
    - *Mathematics:* Neural network learns to compress and reconstruct input. High reconstruction error indicates anomaly.
    - *Computer Science:* Trains a feedforward neural network (MLP) to minimize reconstruction loss (MSE) between input and output.

---

### 3. **Graph Analysis**
- **Entity Graph:** Users, files, and devices are nodes; edges represent access or usage.
- **Centrality Measures:**
    - *Degree Centrality:* Number of connections (activity level).
    - *Betweenness Centrality:* Frequency a node lies on shortest paths (potential for information flow/control).
- **At-Risk Subgraph:** Focuses on high-risk users and their direct connections for visualization and investigation.

---

### 4. **Explainability**
- **SHAP (SHapley Additive exPlanations):**
    - *Mathematics:* Based on cooperative game theory; attributes model output to each feature by averaging over all possible feature orderings.
    - *Computer Science:* Computes feature importances for each prediction, helping analysts understand why a user is flagged.
- **LIME (Local Interpretable Model-agnostic Explanations):**
    - *Mathematics:* Fits a simple, interpretable model locally around a prediction to approximate the complex model.
    - *Computer Science:* Perturbs input data and observes output changes to estimate feature influence (not supported for Isolation Forest, but available for other models).

---

### 5. **Dashboard & Visualization**
- **Streamlit:** Interactive web app for data exploration, anomaly review, and graph visualization.
- **PyVis/NetworkX:** Renders interactive network graphs for at-risk nodes and their relationships.

---

### 6. **Red Team Simulation**
- Injects malicious behaviors (after-hours access, mass downloads, suspicious USB usage) to test detection capability.

---

## Summary
This system combines unsupervised machine learning, graph theory, and explainable AI to provide a robust, interpretable approach to insider threat detection.
''') 