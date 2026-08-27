from streamlit.testing.v1 import AppTest

SUSPICIOUS = {
    'mean_login_hour_0': 23.0, 'mean_logout_hour_0': 3.0, 'files_per_day_0': 40.0,
    'usb_per_day_0': 15.0, 'emails_per_day_0': 30.0, 'out_of_session_access_0': 450,
    'degree_centrality_0': 0.95, 'betweenness_centrality_0': 0.9,
    'keyword_flag_0': 0.9, 'subject_len_0': 20.0, 'sentiment_0': -0.5,
}

# 1) Boot test
at = AppTest.from_file("dashboard/combined_dashboard.py", default_timeout=30)
at.run()
assert not at.exception, f"Boot raised: {at.exception}"
print("[BOOT] OK - no exceptions on load")

# 2) Fill the form for record 1 and submit (real flow)
at.text_input(key='name_0').set_value('custom_user1')
for k, v in SUSPICIOUS.items():
    at.number_input(key=k).set_value(v)
at.button(key='FormSubmitter:record_form-Submit & Score').click().run()
assert not at.exception, f"Submit flow raised: {at.exception}"

# 3) Verify anomaly scores updated in session state
scores = at.session_state['scores']
assert 'custom_user1' in scores['user'].values, "custom_user1 missing from scores"
cu = scores[scores['user'] == 'custom_user1'].iloc[0]
print(f"[SCORE] custom_user1 -> iso={cu.isolation_forest:.3f} svm={cu.oneclass_svm:.3f} auto={cu.autoencoder:.2f}")
assert cu.autoencoder > 100, "autoencoder score not elevated"
assert cu.isolation_forest > scores[scores['user'] == 'user13']['isolation_forest'].iloc[0], \
    "custom_user1 not more anomalous than red-team user13"

# 4) Verify graph wiring (mirrors build_graph) for the custom user
import networkx as nx, pandas as pd
G = nx.Graph()
fa = pd.read_csv('data/file_access.csv', parse_dates=['access_time'])
uu = pd.read_csv('data/usb_usage.csv', parse_dates=['plug_time', 'unplug_time'])
for _, r in fa.iterrows():
    G.add_edge(r['user'], r['file'], type='access')
for _, r in uu.iterrows():
    G.add_edge(r['user'], r['device'], type='usb')
cus = at.session_state['custom_users']
for _, r in cus.iterrows():
    for i in range(int(round(r['files_per_day']))):
        G.add_edge(r['user'], f"cf_{r['user']}_{i}", type='access')
    for i in range(int(round(r['usb_per_day']))):
        G.add_edge(r['user'], f"cusb_{r['user']}_{i}", type='usb')
assert 'custom_user1' in G.nodes(), "custom_user1 not in graph"
assert G.degree['custom_user1'] == 55, f"expected 55 edges, got {G.degree['custom_user1']}"
assert any(str(n).startswith('cf_') for n in G.neighbors('custom_user1'))
assert any(str(n).startswith('cusb_') for n in G.neighbors('custom_user1'))
print(f"[GRAPH] custom_user1 present with {G.degree['custom_user1']} synthetic edges")

# 5) Reset flow
reset_clicked = False
for b in at.button:
    if b.label == 'Reset to Original Dataset':
        b.click().run()
        reset_clicked = True
        break
assert reset_clicked, "Reset button not found"
assert not at.exception, f"Reset raised: {at.exception}"
assert 'custom_user1' not in at.session_state['features']['user'].values, "reset did not remove custom_user1"
print("[RESET] custom_user1 removed, reverted to original dataset")

print("\nALL TESTS PASSED")
