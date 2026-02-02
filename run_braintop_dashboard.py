
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import time
import networkx as nx


# Import Braintop Renderer
import sys
import os
sys.path.append('igbundle-llm')
from generate_braintop_viz import BraintopRenderer

# --- 1. CONFIGURATION ---
print("Initializing Braintop Renderer (Real Topology)...")
CKPT_PATH = "output/igbundle_v2_cp300_merged"
RENDERER = BraintopRenderer(CKPT_PATH, lite_mode=True)

print("Loading Embedding Model (all-MiniLM-L6-v2)...")
MAX_RETRIES = 5
MODEL = None
for i in range(MAX_RETRIES):
    try:
        from sentence_transformers import SentenceTransformer
        MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully.")
        break
    except Exception as e:
        print(f"Failed to load model (Attempt {i+1}/{MAX_RETRIES}): {e}")
        time.sleep(2)
        
if MODEL is None:
    print("Warning: Semantic Model failed. Visualization will be random.")

# --- 2. DASH APP ---
app = dash.Dash(__name__, title="Braintop Real-Time")

app.layout = html.Div([
    # --- EXTERNAL FONTS ---
    html.Link(rel='stylesheet', href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;700&display=swap'),
    
    # --- HEADER ---
    html.Div([
        html.Div([
            html.H1("IGBUNDLE // BRAINTOP", style={'fontSize': '24px', 'fontWeight': '700', 'letterSpacing': '2px', 'margin': '0', 'color': '#ffffff'}),
            html.Span("Real-Time Riemannian Manifold Projection", style={'fontSize': '12px', 'color': '#888', 'marginLeft': '15px', 'textTransform': 'uppercase'})
        ], style={'display': 'flex', 'alignItems': 'baseline'}),
        html.Div([
            html.Span("LIVE CONNECTION", style={'fontSize': '10px', 'color': '#00ff99', 'fontWeight': 'bold', 'padding': '5px 10px', 'border': '1px solid #00ff99', 'borderRadius': '20px'})
        ])
    ], style={
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
        'padding': '20px 30px', 'borderBottom': '1px solid #222', 'backgroundColor': '#050505'
    }),

    html.Div([
        # --- LEFT COLUMN: CONTROLS & MONITOR ---
        html.Div([
            # TERMINAL CARD
            html.Div([
                html.Div([
                    html.Span("●", style={'color': '#ff5f56', 'marginRight': '6px'}),
                    html.Span("●", style={'color': '#ffbd2e', 'marginRight': '6px'}),
                    html.Span("●", style={'color': '#27c93f', 'marginRight': '10px'}),
                    html.Span("probe_stream.log", style={'color': '#666', 'fontSize': '12px', 'fontFamily': 'JetBrains Mono'})
                ], style={'marginBottom': '10px', 'borderBottom': '1px solid #222', 'paddingBottom': '8px'}),
                
                html.Div(id='sequence-display', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '13px', 'color': '#00ff99', 
                    'minHeight': '120px', 'lineHeight': '1.5'
                })
            ], style={
                'backgroundColor': '#0a0a0a', 'border': '1px solid #333', 'borderRadius': '8px', 
                'padding': '15px', 'marginBottom': '20px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.5)'
            }),
            
            # TELEMETRY DECK (Live Science)
            html.Div([
                html.H4("LIVE TELEMETRY (Riemmanian)", style={'color': '#888', 'fontSize': '10px', 'letterSpacing': '1px', 'marginBottom': '15px'}),
                
                # Curvature Metric
                html.Div([
                    html.Span("Sectional Curvature (K)", style={'fontSize': '11px', 'color': '#ccc'}),
                    html.Span(id='telem-curvature', children="0.00", style={'float': 'right', 'color': '#00ff99', 'fontFamily': 'JetBrains Mono'})
                ], style={'marginBottom': '8px', 'borderBottom': '1px solid #222', 'paddingBottom': '4px'}),
                
                # Entropy Metric
                html.Div([
                    html.Span("Manifold Entropy (S)", style={'fontSize': '11px', 'color': '#ccc'}),
                    html.Span(id='telem-entropy', children="0.00", style={'float': 'right', 'color': '#ffbd2e', 'fontFamily': 'JetBrains Mono'})
                ], style={'marginBottom': '8px', 'borderBottom': '1px solid #222', 'paddingBottom': '4px'}),
                
                # Fiber Status
                html.Div([
                    html.Span("Active Bundle Section", style={'fontSize': '11px', 'color': '#ccc'}),
                    html.Span(id='telem-fiber', children="N/A", style={'float': 'right', 'color': '#27c93f', 'fontFamily': 'JetBrains Mono'})
                ], style={'marginBottom': '8px', 'borderBottom': '1px solid #222', 'paddingBottom': '4px'}),
                
                # PHYSICS TELEMETRY (PHASE 4)
                html.Div([
                    html.Span("Hamiltonian E", style={'fontSize': '11px', 'color': '#ccc'}),
                    html.Span(id='telem-energy', children="0.00", style={'float': 'right', 'color': '#ff5f56', 'fontFamily': 'JetBrains Mono'})
                ], style={'marginBottom': '8px', 'borderBottom': '1px solid #222', 'paddingBottom': '4px'}),

                html.Div([
                    html.Span("Retro Loss", style={'fontSize': '11px', 'color': '#ccc'}),
                    html.Span(id='telem-retro', children="0.00", style={'float': 'right', 'color': '#00bbff', 'fontFamily': 'JetBrains Mono'})
                ], style={'marginBottom': '8px', 'borderBottom': '1px solid #222', 'paddingBottom': '4px'})
                
            ], style={
                'backgroundColor': '#0a0a0a', 'border': '1px solid #222', 'borderRadius': '8px', 'padding': '20px'
            }),
            
            # MANUAL INPUT
            html.Div([
                dcc.Input(id='input-thought', type='text', placeholder='Inject Manual Thought...', 
                        style={'width': '100%', 'padding': '12px', 'backgroundColor': '#111', 'border': '1px solid #333', 'color': 'white', 'borderRadius': '4px', 'fontFamily': 'Inter'}),
                html.Button('PROJECT', id='btn-project', n_clicks=0, 
                           style={'width': '100%', 'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#222', 'border': '1px solid #333', 'color': 'white', 'cursor': 'pointer', 'fontWeight': '600'})
            ], style={'marginTop': '20px'})
            
        ], style={'width': '30%', 'minWidth': '300px', 'marginRight': '20px'}),
        
        # --- RIGHT COLUMN: VISUALIZATION ---
        html.Div([
            dcc.Loading(
                id="loading-1",
                type="default",
                children=dcc.Graph(id='braintop-graph', style={'height': '75vh', 'borderRadius': '8px', 'overflow': 'hidden'})
            )
        ], style={'flex': '1', 'backgroundColor': '#000', 'border': '1px solid #222', 'borderRadius': '8px', 'position': 'relative'})
        
    ], style={'display': 'flex', 'padding': '30px', 'maxWidth': '1600px', 'margin': '0 auto'}),

    # Store and Interval (Enabled by default, 500ms poll)
    dcc.Store(id='anim-store', data={'steps': [], 'current_idx': 0, 'active': False}),
    dcc.Interval(id='anim-interval', interval=500, n_intervals=0, disabled=False),

], style={'backgroundColor': '#000', 'color': 'white', 'minHeight': '100vh', 'fontFamily': 'Inter'})


# CALLBACK 2: Update Graph (Auto-Probing or Manual)
@app.callback(
    [Output('braintop-graph', 'figure'),
     Output('sequence-display', 'children'),
     Output('telem-curvature', 'children'),
     Output('telem-entropy', 'children'),
     Output('telem-fiber', 'children'),
     Output('telem-energy', 'children'),
     Output('telem-retro', 'children')],
    [Input('anim-interval', 'n_intervals'),
     Input('btn-project', 'n_clicks')],
    [State('input-thought', 'value'),
     State('anim-store', 'data')]
)
def update_graph(n_intervals, n_clicks, manual_thought, store):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'none'
    
    thought = "Waiting for Probe Data..."
    embedding = None
    msg = "SYSTEM IDLE. Waiting for Eval Script..."
    
    # Telemetry Defaults
    k_val = "0.00"
    s_val = "0.00"
    fiber_val = "Idle"
    h_val = "0.00"
    retro_val = "0.00"
    
    # 1. READ LIVE STATE
    try:
        import json
        state = {'status': 'idle', 'prompt': 'Waiting for LM Studio...', 'embedding': None}
        try:
            if os.path.exists("dashboard_state_lmstudio.json"):
                with open("dashboard_state_lmstudio.json", "r") as f:
                    state = json.load(f)
            elif os.path.exists("dashboard_state.json"): # Fallback
                 with open("dashboard_state.json", "r") as f:
                    state = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        thought = state.get("prompt", "Unknown")
        emb_list = state.get("embedding")
        if emb_list and isinstance(emb_list, list):
            embedding = np.array(emb_list)
        elif MODEL and thought:
             try:
                 embedding = MODEL.encode([thought])[0]
             except:
                 pass
            
        status_color = "#00ff99" if state.get("status") == "complete" else "#ffff00"
        lat = state.get('latency', 0)
        dom = state.get('domain', 'N/A')
        
        # Parse or Simulate Telemetry based on Embedding Activity
        if embedding is not None:
             # Simulate Curvature from Embedding variance/norm
             # High variance ~ high curvature?
             # Just for visualization until 'curvature' key is in JSON
             raw_k = state.get('curvature', np.std(embedding) * 1000)
             k_val = f"{raw_k:.2f}"
             
             # Entropy
             raw_s = state.get('entropy', np.abs(np.mean(embedding)) * 100)
             s_val = f"{raw_s:.2f}"
             
             # Fiber
             # Hash of prompt or embedding dominant feature?
             feat_idx = np.argmax(np.abs(embedding))
             fiber_val = f"Fiber {feat_idx}"
             
             # Physics
             h_val = f"{state.get('hamiltonian_energy', 0.0):.2f}"
             retro_val = f"{state.get('retrospection_loss', 0.0):.3f}"
             active_fiber_name = state.get('active_fiber', "Idle")
             if active_fiber_name != "Idle":
                 fiber_val = active_fiber_name
        
        msg = html.Div([
            html.Strong(f"[{state.get('status', 'IDLE').upper()}] PROBE {state.get('id', '?')}", style={'color': status_color}),
            html.Br(),
            f"DOMAIN: {dom} | LATENCY: {lat:.2f}s",
            html.Br(),
            html.Span(f"INPUT: {thought[:80]}...", style={'color': '#ccc'})
        ])
    except Exception as e:
        print(f"Read Error: {e}")
        msg = f"Reader Error: {str(e)}"

    # 2. RENDER
    if embedding is None:
        if manual_thought:
             thought = manual_thought
             msg = f"MANUAL INPUT: {thought}"
             if MODEL: embedding = MODEL.encode([thought])[0]
    
    if embedding is None:
         pass
        
    fig = RENDERER.render_frame(prompt=thought, embedding=embedding)
    
    # 3. ENFORCE DARK MODE + SMOOTHING
    fig.update_layout(
        template="plotly_dark",
        uirevision='constant', 
        transition={
            'duration': 500, 
            'easing': 'cubic-in-out'
        },
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font=dict(color='#ffffff', family="Consolas, monospace"),
        title=dict(
            text=f"Topological Activation: {thought[:30]}...",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=14, color="#00ff99")
        ),
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            bgcolor='#000000',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig, msg, k_val, s_val, fiber_val, h_val, retro_val

if __name__ == '__main__':
    print("Starting Braintop Real-Time Dashboard on http://127.0.0.1:8055")
    try:
        app.run(debug=False, port=8055)
    except Exception as e:
        print(f"Server Crash: {e}")
