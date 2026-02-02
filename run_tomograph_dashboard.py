"""
Real-Time IGBundle Model Tomograph Dashboard

Connects to LM Studio and displays live geometric telemetry.
No demos - pure model state visualization.

Author: Jesus Vilela Jato (ManifoldGL Research)
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import json
import time
import threading
import queue
import requests
from typing import Optional, Dict, Any

# WebSocket client
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

import sys
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
braintop_src = os.path.join(base_dir, "braintop", "src")
if braintop_src not in sys.path:
    sys.path.append(braintop_src)


# --- LM STUDIO STATUS ---
def check_lmstudio(url: str = "http://192.168.56.1:1234") -> Dict[str, Any]:
    """Check LM Studio connection status."""
    try:
        resp = requests.get(f"{url}/v1/models", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                model = data["data"][0]
                return {
                    "connected": True,
                    "model": model.get("id", "Unknown"),
                    "url": url
                }
    except:
        pass
    return {"connected": False, "model": None, "url": url}


# --- HTTP CLIENT ---
class HTTPClient:
    """HTTP polling client for tomograph data."""
    
    def __init__(self, state_file: str = "tomograph_state.json"):
        self.state_file = state_file
        self.last_frame_id = -1
    
    def get_frame(self) -> Optional[Dict]:
        """Get latest frame from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                frame_id = state.get("frame_id", 0)
                if frame_id > self.last_frame_id:
                    self.last_frame_id = frame_id
                    return state
        except:
            pass
        return None


# --- VISUALIZATION FUNCTIONS ---
def create_manifold_figure(frame_data: Optional[Dict]) -> go.Figure:
    """Create 3D manifold visualization."""
    fig = go.Figure()
    
    data = frame_data.get("data") if frame_data else None
    
    if data is None or data.get("base_coords_3d") is None:
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=10, color='#333'),
            name="Waiting for telemetry..."
        ))
    else:
        coords = np.array(data["base_coords_3d"])
        
        if data.get("fiber_activations"):
            fibers = np.array(data["fiber_activations"])
            colors = np.argmax(fibers, axis=1)
        elif data.get("curvature_values"):
            colors = np.array(data["curvature_values"])
        else:
            colors = np.arange(len(coords))
        
        if data.get("entropy_values"):
            sizes = 5 + 10 * np.array(data["entropy_values"])
        else:
            sizes = 8
        
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Viridis',
                opacity=0.85,
                showscale=True,
                colorbar=dict(title="Fiber", x=1.02)
            ),
            hovertemplate=(
                "<b>Point %{pointNumber}</b><br>"
                "x: %{x:.3f}<br>"
                "y: %{y:.3f}<br>"
                "z: %{z:.3f}<br>"
                "<extra></extra>"
            ),
            name="Geometric State"
        ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#000',
        plot_bgcolor='#000',
        uirevision='manifold',
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=True, gridcolor='#222', title=''),
            yaxis=dict(showbackground=False, showgrid=True, gridcolor='#222', title=''),
            zaxis=dict(showbackground=False, showgrid=True, gridcolor='#222', title=''),
            bgcolor='#000',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text="Geometric Manifold State",
            y=0.98, x=0.5, xanchor='center',
            font=dict(size=14, color='#00ff99')
        )
    )
    
    return fig


def create_curvature_heatmap(frame_data: Optional[Dict]) -> go.Figure:
    """Create curvature heatmap."""
    fig = go.Figure()
    
    data = frame_data.get("data") if frame_data else None
    
    if data and data.get("curvature_values"):
        curvature = np.array(data["curvature_values"])
        n = len(curvature)
        side = int(np.sqrt(n)) + 1
        
        curvature_padded = np.zeros(side * side)
        curvature_padded[:n] = curvature
        curvature_2d = curvature_padded.reshape(side, side)
        
        fig.add_trace(go.Heatmap(
            z=curvature_2d,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="κ"),
        ))
    else:
        fig.add_trace(go.Heatmap(z=[[0]], colorscale='RdBu'))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        title=dict(text="Curvature Map", font=dict(size=12, color='#888')),
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=250
    )
    
    return fig


def create_energy_chart(frame_data: Optional[Dict]) -> go.Figure:
    """Create energy chart."""
    fig = go.Figure()
    
    data = frame_data.get("data") if frame_data else None
    
    if data and data.get("energy_values"):
        energy = np.array(data["energy_values"])
        
        fig.add_trace(go.Scatter(
            x=list(range(len(energy))),
            y=energy,
            mode='lines+markers',
            line=dict(color='#ff6b35', width=2),
            marker=dict(size=4),
        ))
        
        fig.add_hline(y=energy.mean(), line_dash="dash", line_color="#555")
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        title=dict(text="Hamiltonian Energy", font=dict(size=12, color='#888')),
        margin=dict(l=50, r=20, t=40, b=30),
        xaxis=dict(title="", gridcolor='#222'),
        yaxis=dict(title="H", gridcolor='#222'),
        height=200
    )
    
    return fig


def create_op_distribution(frame_data: Optional[Dict]) -> go.Figure:
    """Create operation distribution chart."""
    fig = go.Figure()
    
    data = frame_data.get("data") if frame_data else None
    
    if data and data.get("op_distribution"):
        op_dist = data["op_distribution"]
        ops = list(op_dist.keys())
        probs = list(op_dist.values())
        
        colors = ['#00ff99' if p > 0.15 else '#555' for p in probs]
        
        fig.add_trace(go.Bar(
            x=ops, y=probs,
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition='outside',
            textfont=dict(size=9, color='#888')
        ))
    else:
        fig.add_trace(go.Bar(x=["N/A"], y=[0]))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        title=dict(text="Compiler Operations", font=dict(size=12, color='#888')),
        margin=dict(l=40, r=20, t=40, b=60),
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(title="P", range=[0, 1]),
        height=200
    )
    
    return fig


# --- DASH APP ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="IGBundle Tomograph"
)

HTTP_CLIENT = HTTPClient()
LMSTUDIO_URL = "http://192.168.56.1:1234"

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("IGBUNDLE // TOMOGRAPH", 
                       style={'fontSize': '28px', 'fontWeight': '700', 'letterSpacing': '3px', 
                              'margin': '0', 'color': '#ffffff', 'fontFamily': 'Consolas, monospace'}),
                html.Span("Real-Time Model Telemetry", 
                         style={'fontSize': '11px', 'color': '#666', 'marginLeft': '15px', 
                                'textTransform': 'uppercase', 'letterSpacing': '1px'})
            ])
        ], width=6),
        dbc.Col([
            html.Div([
                html.Span(id='lmstudio-status', children="CHECKING LM STUDIO...", 
                         style={'fontSize': '10px', 'fontWeight': 'bold', 'padding': '5px 12px',
                                'border': '1px solid #ffff00', 'borderRadius': '20px', 'color': '#ffff00',
                                'marginRight': '10px'}),
                html.Span(id='connection-status', children="CONNECTING...", 
                         style={'fontSize': '10px', 'fontWeight': 'bold', 'padding': '5px 12px',
                                'border': '1px solid #ffff00', 'borderRadius': '20px', 'color': '#ffff00'}),
            ], style={'textAlign': 'right'})
        ], width=6)
    ], style={'padding': '20px 30px', 'borderBottom': '1px solid #222', 'backgroundColor': '#050505'}),
    
    # Model Info Bar
    dbc.Row([
        dbc.Col([
            html.Div(id='model-info', children="Waiting for model info...",
                    style={'fontSize': '11px', 'color': '#555', 'padding': '8px 30px', 
                           'backgroundColor': '#0a0a0a', 'borderBottom': '1px solid #1a1a1a'})
        ])
    ]),
    
    # Main Content
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='manifold-3d', style={'height': '70vh'}, config={'displayModeBar': True})
        ], width=8),
        
        dbc.Col([
            html.Div([
                html.H5("GEOMETRIC METRICS", style={'color': '#555', 'fontSize': '10px', 
                                                     'letterSpacing': '1px', 'marginBottom': '15px'}),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("Curvature κ", style={'fontSize': '9px', 'color': '#666'}),
                            html.Div(id='metric-curvature', children="—", 
                                    style={'fontSize': '24px', 'fontWeight': '700', 'color': '#00d4ff'})
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.Span("Energy H", style={'fontSize': '9px', 'color': '#666'}),
                            html.Div(id='metric-energy', children="—", 
                                    style={'fontSize': '24px', 'fontWeight': '700', 'color': '#ff6b35'})
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.Span("Entropy S", style={'fontSize': '9px', 'color': '#666'}),
                            html.Div(id='metric-entropy', children="—", 
                                    style={'fontSize': '24px', 'fontWeight': '700', 'color': '#a855f7'})
                        ])
                    ], width=4),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("Consensus", style={'fontSize': '9px', 'color': '#666'}),
                            html.Div(id='metric-consensus', children="—", 
                                    style={'fontSize': '24px', 'fontWeight': '700', 'color': '#00ff99'})
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.Span("Latency", style={'fontSize': '9px', 'color': '#666'}),
                            html.Div(id='metric-latency', children="—", 
                                    style={'fontSize': '24px', 'fontWeight': '700', 'color': '#888'})
                        ])
                    ], width=6),
                ], style={'marginTop': '15px'})
            ], style={'backgroundColor': '#0a0a0a', 'padding': '15px', 'borderRadius': '8px',
                      'border': '1px solid #222', 'marginBottom': '15px'}),
            
            dcc.Graph(id='curvature-heatmap', style={'height': '25vh'}),
            dcc.Graph(id='op-distribution', style={'height': '20vh'}),
        ], width=4)
    ], style={'padding': '20px'}),
    
    # Bottom
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='energy-chart', style={'height': '20vh'})
        ], width=12)
    ], style={'padding': '0 20px 20px 20px'}),
    
    # Intervals
    dcc.Interval(id='update-interval', interval=100, n_intervals=0),
    dcc.Interval(id='lmstudio-check', interval=5000, n_intervals=0),
    
    dcc.Store(id='frame-store', data=None),
    
], fluid=True, style={'backgroundColor': '#000', 'minHeight': '100vh'})


@app.callback(
    [Output('lmstudio-status', 'children'),
     Output('lmstudio-status', 'style'),
     Output('model-info', 'children')],
    [Input('lmstudio-check', 'n_intervals')]
)
def update_lmstudio_status(n):
    """Check LM Studio connection."""
    status = check_lmstudio(LMSTUDIO_URL)
    
    if status["connected"]:
        text = f"LM STUDIO: {status['model'][:30]}"
        style = {'fontSize': '10px', 'fontWeight': 'bold', 'padding': '5px 12px',
                'border': '1px solid #00ff99', 'borderRadius': '20px', 'color': '#00ff99',
                'marginRight': '10px'}
        info = f"Model: {status['model']} | Endpoint: {status['url']}"
    else:
        text = "LM STUDIO: OFFLINE"
        style = {'fontSize': '10px', 'fontWeight': 'bold', 'padding': '5px 12px',
                'border': '1px solid #ff5555', 'borderRadius': '20px', 'color': '#ff5555',
                'marginRight': '10px'}
        info = f"Cannot connect to LM Studio at {LMSTUDIO_URL}"
    
    return text, style, info


@app.callback(
    [Output('frame-store', 'data'),
     Output('connection-status', 'children'),
     Output('connection-status', 'style')],
    [Input('update-interval', 'n_intervals')]
)
def update_frame(n):
    """Fetch latest frame."""
    frame = HTTP_CLIENT.get_frame()
    
    if frame and frame.get("status") == "active":
        status_text = "TELEMETRY ACTIVE"
        status_style = {'fontSize': '10px', 'fontWeight': 'bold', 'padding': '5px 12px',
                       'border': '1px solid #00ff99', 'borderRadius': '20px', 'color': '#00ff99'}
    else:
        status_text = "WAITING..."
        status_style = {'fontSize': '10px', 'fontWeight': 'bold', 'padding': '5px 12px',
                       'border': '1px solid #ffff00', 'borderRadius': '20px', 'color': '#ffff00'}
    
    return frame, status_text, status_style


@app.callback(
    [Output('manifold-3d', 'figure'),
     Output('curvature-heatmap', 'figure'),
     Output('energy-chart', 'figure'),
     Output('op-distribution', 'figure'),
     Output('metric-curvature', 'children'),
     Output('metric-energy', 'children'),
     Output('metric-entropy', 'children'),
     Output('metric-consensus', 'children'),
     Output('metric-latency', 'children')],
    [Input('frame-store', 'data')]
)
def update_visualizations(frame_data):
    """Update all visualizations."""
    
    manifold_fig = create_manifold_figure(frame_data)
    curvature_fig = create_curvature_heatmap(frame_data)
    energy_fig = create_energy_chart(frame_data)
    op_fig = create_op_distribution(frame_data)
    
    data = frame_data.get("data") if frame_data else None
    
    if data:
        curvature = "—"
        if data.get("curvature_values"):
            c = np.array(data["curvature_values"])
            curvature = f"{c.mean():.3f}"
        
        energy = "—"
        if data.get("energy_values"):
            e = np.array(data["energy_values"])
            energy = f"{e.mean():.2f}"
        
        entropy = "—"
        if data.get("entropy_values"):
            s = np.array(data["entropy_values"])
            entropy = f"{s.mean():.3f}"
        
        consensus = f"{data.get('consensus_loss', 0):.4f}"
        latency = f"{frame_data.get('latency', 0)*1000:.0f}ms"
    else:
        curvature = energy = entropy = consensus = latency = "—"
    
    return (manifold_fig, curvature_fig, energy_fig, op_fig,
            curvature, energy, entropy, consensus, latency)


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8050,
    lmstudio_url: str = "http://192.168.56.1:1234",
    debug: bool = False
):
    """Run the dashboard."""
    global LMSTUDIO_URL
    LMSTUDIO_URL = lmstudio_url
    
    print(f"\n[Dashboard] IGBundle Tomograph Dashboard")
    print(f"[Dashboard] URL: http://{host}:{port}")
    print(f"[Dashboard] LM Studio: {lmstudio_url}")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--lmstudio", default="http://192.168.56.1:1234")
    parser.add_argument("--ws", default="ws://localhost:8765")  # Kept for compat
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    run_dashboard(args.host, args.port, args.lmstudio, args.debug)
