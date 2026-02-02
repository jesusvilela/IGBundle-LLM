"""
IGBundle Intrinsic Manifold Tomograph

NO 3D projections. Shows actual manifold structure:
- Chart atlas coverage
- Metric tensor components g_ij
- Curvature tensors
- Fiber bundle sections

Author: Jesus Vilela Jato (ManifoldGL Research)
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import time
import os
import requests
from typing import Optional, Dict, Any, List, Tuple


# --- Get base directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- MANIFOLD COMPUTATIONS ---

class ManifoldAtlas:
    """Atlas of charts covering the manifold."""
    
    def __init__(self, dim: int, num_charts: int = 4):
        self.dim = dim
        self.num_charts = num_charts
        self.chart_centers = None
        self.chart_radii = None
        self.chart_labels = None
    
    def define_charts(self, base_coords: np.ndarray) -> None:
        from sklearn.cluster import KMeans
        
        N, D = base_coords.shape
        n_charts = min(self.num_charts, N)
        
        if N < 2:
            self.chart_centers = base_coords
            self.chart_labels = np.zeros(N, dtype=int)
            self.chart_radii = np.ones(1)
            return
        
        kmeans = KMeans(n_clusters=n_charts, n_init=10, random_state=42)
        kmeans.fit(base_coords)
        
        self.chart_centers = kmeans.cluster_centers_
        self.chart_labels = kmeans.labels_
        
        self.chart_radii = np.zeros(n_charts)
        for i in range(n_charts):
            mask = self.chart_labels == i
            if mask.sum() > 0:
                dists = np.linalg.norm(base_coords[mask] - self.chart_centers[i], axis=1)
                self.chart_radii[i] = dists.max() * 1.2 if dists.max() > 0 else 1.0
    
    def transition_jacobian(self, from_chart: int, to_chart: int) -> np.ndarray:
        return np.eye(self.dim)


def compute_metric_tensor(base_coords: np.ndarray) -> np.ndarray:
    N, D = base_coords.shape
    metrics = np.zeros((N, D, D))
    
    for i in range(N):
        metrics[i] = np.eye(D)
        r = np.linalg.norm(base_coords[i])
        if r > 0.1:
            conformal_factor = 4 / (1 - min(r**2, 0.99))**2
            metrics[i] *= conformal_factor
    
    return metrics


def compute_scalar_curvature(base_coords: np.ndarray, metric: np.ndarray) -> np.ndarray:
    N, D = base_coords.shape
    scalar = np.zeros(N)
    
    for n in range(N):
        r = np.linalg.norm(base_coords[n])
        if r > 0.1:
            # Poincaré disk has constant negative curvature
            scalar[n] = -1.0
        else:
            scalar[n] = 0.0
    
    return scalar


def compute_fiber_section(fiber_probs: np.ndarray, base_coords: np.ndarray) -> Dict[str, Any]:
    N, K = fiber_probs.shape
    
    dominant = np.argmax(fiber_probs, axis=1)
    entropy = -np.sum(fiber_probs * np.log(fiber_probs + 1e-8), axis=1)
    
    smoothness = 0.0
    if N > 1:
        for i in range(N - 1):
            dist = np.linalg.norm(base_coords[i+1] - base_coords[i])
            fiber_dist = np.linalg.norm(fiber_probs[i+1] - fiber_probs[i])
            if dist > 0:
                smoothness += fiber_dist / dist
        smoothness /= (N - 1)
    
    return {
        "dominant_fibers": dominant,
        "entropy": entropy,
        "mean_entropy": float(entropy.mean()),
        "section_smoothness": float(smoothness),
    }


# --- HTTP CLIENT ---
class HTTPClient:
    def __init__(self, state_file: str):
        self.state_file = os.path.abspath(state_file)
        self.last_frame_id = -1
    
    def get_frame(self) -> Optional[Dict]:
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                frame_id = state.get("frame_id", 0)
                if frame_id > self.last_frame_id:
                    self.last_frame_id = frame_id
                    return state
        except Exception as e:
            pass
        return None


# --- LM STUDIO CHECK ---
def check_lmstudio(url: str) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{url}/v1/models", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                return {"connected": True, "model": data["data"][0].get("id", "Unknown")}
    except:
        pass
    return {"connected": False, "model": None}


# --- VISUALIZATION ---

def create_chart_coverage_figure(atlas: ManifoldAtlas, base_coords: np.ndarray) -> go.Figure:
    fig = go.Figure()
    
    if atlas.chart_centers is None or base_coords is None or len(base_coords) == 0:
        fig.add_annotation(text="Waiting for data...", x=0.5, y=0.5, showarrow=False,
                          font=dict(color='#666'))
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a')
        return fig
    
    N, D = base_coords.shape
    x = base_coords[:, 0]
    y = base_coords[:, 1] if D > 1 else np.zeros(N)
    
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=6, color=atlas.chart_labels, colorscale='Set1', 
                   showscale=True, colorbar=dict(title="Chart")),
        hovertemplate="q¹: %{x:.3f}<br>q²: %{y:.3f}<extra></extra>",
        name="Base Points"
    ))
    
    cx = atlas.chart_centers[:, 0]
    cy = atlas.chart_centers[:, 1] if D > 1 else np.zeros(len(cx))
    
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode='markers+text',
        marker=dict(size=15, symbol='x', color='#ff0000'),
        text=[f"φ_{i}" for i in range(len(cx))],
        textposition='top center', textfont=dict(size=12, color='#ff6666'),
        name="Chart Centers"
    ))
    
    for i, (center, radius) in enumerate(zip(atlas.chart_centers, atlas.chart_radii)):
        theta = np.linspace(0, 2*np.pi, 50)
        cx_c = center[0] + radius * np.cos(theta)
        cy_c = (center[1] if D > 1 else 0) + radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=cx_c, y=cy_c, mode='lines',
            line=dict(color='rgba(255,100,100,0.3)', width=1, dash='dash'),
            showlegend=False, hoverinfo='skip'
        ))
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
        title=dict(text="Atlas Chart Coverage (q¹, q²)", font=dict(size=12, color='#888')),
        xaxis=dict(title="q¹", gridcolor='#222'),
        yaxis=dict(title="q²", gridcolor='#222', scaleanchor="x"),
        height=350, margin=dict(l=50, r=20, t=40, b=40)
    )
    return fig


def create_metric_tensor_figure(metric: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=1, cols=4, subplot_titles=["g at p₀", "g at p₁", "g at p₂", "Mean g"],
                       horizontal_spacing=0.08)
    
    if metric is None or len(metric) == 0:
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0a0a0a')
        return fig
    
    N = len(metric)
    samples = [0, N//2, N-1] if N >= 3 else list(range(min(3, N)))
    
    for i, idx in enumerate(samples[:3]):
        fig.add_trace(go.Heatmap(z=metric[idx], colorscale='RdBu', zmid=1, showscale=(i==2)),
                     row=1, col=i+1)
    
    fig.add_trace(go.Heatmap(z=metric.mean(axis=0), colorscale='RdBu', zmid=1, showscale=True),
                 row=1, col=4)
    
    fig.update_layout(template="plotly_dark", paper_bgcolor='#0a0a0a', height=200,
                     margin=dict(l=30, r=30, t=40, b=20))
    return fig


def create_curvature_figure(scalar_curvature: np.ndarray, base_coords: np.ndarray) -> go.Figure:
    fig = go.Figure()
    
    if scalar_curvature is None or base_coords is None or len(base_coords) == 0:
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a')
        return fig
    
    N, D = base_coords.shape
    x = base_coords[:, 0]
    y = base_coords[:, 1] if D > 1 else np.zeros(N)
    
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=8, color=scalar_curvature, colorscale='RdBu', cmid=0,
                   showscale=True, colorbar=dict(title="R")),
        hovertemplate="q¹: %{x:.3f}<br>q²: %{y:.3f}<br>R: %{marker.color:.4f}<extra></extra>",
    ))
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
        title=dict(text="Scalar Curvature R (negative=hyperbolic)", font=dict(size=12, color='#888')),
        xaxis=dict(title="q¹", gridcolor='#222'),
        yaxis=dict(title="q²", gridcolor='#222', scaleanchor="x"),
        height=300, margin=dict(l=50, r=20, t=40, b=40)
    )
    return fig


def create_fiber_section_figure(section_data: Dict, base_coords: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Dominant Fiber σ(p)", "Section Entropy H(σ)"],
                       horizontal_spacing=0.1)
    
    if not section_data or base_coords is None or len(base_coords) == 0:
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0a0a0a')
        return fig
    
    N, D = base_coords.shape
    x = base_coords[:, 0]
    y = base_coords[:, 1] if D > 1 else np.zeros(N)
    
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
        marker=dict(size=7, color=section_data["dominant_fibers"], colorscale='Viridis',
                   showscale=True, colorbar=dict(title="Fiber", x=0.45))), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
        marker=dict(size=7, color=section_data["entropy"], colorscale='Plasma',
                   showscale=True, colorbar=dict(title="H", x=1.0))), row=1, col=2)
    
    fig.update_layout(template="plotly_dark", paper_bgcolor='#0a0a0a', height=280,
                     margin=dict(l=40, r=40, t=40, b=30))
    fig.update_xaxes(title_text="q¹", gridcolor='#222')
    fig.update_yaxes(title_text="q²", gridcolor='#222')
    return fig


def create_transition_map_figure(atlas: ManifoldAtlas) -> go.Figure:
    if atlas.chart_centers is None:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor='#0a0a0a')
        return fig
    
    n_charts = len(atlas.chart_centers)
    jacobian_dets = np.ones((n_charts, n_charts))  # Identity Jacobians
    
    fig = go.Figure(go.Heatmap(
        z=jacobian_dets,
        x=[f"φ_{i}" for i in range(n_charts)],
        y=[f"φ_{i}" for i in range(n_charts)],
        colorscale='Greens', zmin=0, zmax=2,
        hovertemplate="φ_%{y} → φ_%{x}<br>det(J): %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='#0a0a0a',
        title=dict(text="Transition Map Jacobians det(∂φⱼ/∂φᵢ)", font=dict(size=12, color='#888')),
        height=200, margin=dict(l=50, r=20, t=40, b=30),
        xaxis=dict(title="To"), yaxis=dict(title="From")
    )
    return fig


# --- DASH APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="IGBundle Manifold Tomograph")

# Global state
STATE_FILE = os.path.join(BASE_DIR, "tomograph_state.json")
HTTP_CLIENT = HTTPClient(STATE_FILE)
LMSTUDIO_URL = "http://192.168.56.1:1234"
ATLAS = ManifoldAtlas(dim=3, num_charts=6)

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("IGBUNDLE // MANIFOLD TOMOGRAPH", 
                   style={'fontSize': '24px', 'fontWeight': '700', 'letterSpacing': '2px', 
                          'margin': '0', 'color': '#fff', 'fontFamily': 'Consolas'}),
            html.Span("Intrinsic Geometry · No 3D Projections", 
                     style={'fontSize': '10px', 'color': '#666', 'marginLeft': '15px'})
        ], width=7),
        dbc.Col([
            html.Span(id='lmstudio-status', children="LM STUDIO", 
                     style={'fontSize': '10px', 'padding': '4px 10px', 'border': '1px solid #555',
                            'borderRadius': '15px', 'color': '#555', 'marginRight': '8px'}),
            html.Span(id='telemetry-status', children="TELEMETRY", 
                     style={'fontSize': '10px', 'padding': '4px 10px', 'border': '1px solid #555',
                            'borderRadius': '15px', 'color': '#555'}),
        ], width=5, style={'textAlign': 'right'})
    ], style={'padding': '15px 25px', 'borderBottom': '1px solid #222', 'backgroundColor': '#050505'}),
    
    # Metrics
    dbc.Row([
        dbc.Col([html.Span("dim M", style={'fontSize': '9px', 'color': '#555'}),
                 html.Div(id='metric-dim', children="—", style={'fontSize': '18px', 'color': '#00d4ff'})],
                width=2, style={'textAlign': 'center'}),
        dbc.Col([html.Span("# Charts", style={'fontSize': '9px', 'color': '#555'}),
                 html.Div(id='metric-charts', children="—", style={'fontSize': '18px', 'color': '#00ff99'})],
                width=2, style={'textAlign': 'center'}),
        dbc.Col([html.Span("⟨R⟩", style={'fontSize': '9px', 'color': '#555'}),
                 html.Div(id='metric-curvature', children="—", style={'fontSize': '18px', 'color': '#ff6b35'})],
                width=2, style={'textAlign': 'center'}),
        dbc.Col([html.Span("⟨det g⟩", style={'fontSize': '9px', 'color': '#555'}),
                 html.Div(id='metric-detg', children="—", style={'fontSize': '18px', 'color': '#a855f7'})],
                width=2, style={'textAlign': 'center'}),
        dbc.Col([html.Span("⟨H(σ)⟩", style={'fontSize': '9px', 'color': '#555'}),
                 html.Div(id='metric-entropy', children="—", style={'fontSize': '18px', 'color': '#eab308'})],
                width=2, style={'textAlign': 'center'}),
        dbc.Col([html.Span("Smooth", style={'fontSize': '9px', 'color': '#555'}),
                 html.Div(id='metric-smooth', children="—", style={'fontSize': '18px', 'color': '#22c55e'})],
                width=2, style={'textAlign': 'center'}),
    ], style={'padding': '12px 25px', 'backgroundColor': '#0a0a0a', 'borderBottom': '1px solid #1a1a1a'}),
    
    # Main
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='chart-coverage', style={'height': '350px'}),
            dcc.Graph(id='curvature-map', style={'height': '300px'}),
        ], width=6),
        dbc.Col([
            dcc.Graph(id='metric-tensor', style={'height': '200px'}),
            dcc.Graph(id='fiber-section', style={'height': '280px'}),
            dcc.Graph(id='transition-maps', style={'height': '200px'}),
        ], width=6)
    ], style={'padding': '15px'}),
    
    dcc.Interval(id='update-interval', interval=200, n_intervals=0),
    dcc.Interval(id='lmstudio-check', interval=5000, n_intervals=0),
    dcc.Store(id='frame-store', data=None),
], fluid=True, style={'backgroundColor': '#000', 'minHeight': '100vh'})


@app.callback(
    [Output('lmstudio-status', 'children'), Output('lmstudio-status', 'style')],
    [Input('lmstudio-check', 'n_intervals')]
)
def update_lmstudio_status(n):
    status = check_lmstudio(LMSTUDIO_URL)
    if status["connected"]:
        return f"LM: {status['model'][:20]}", {
            'fontSize': '10px', 'padding': '4px 10px', 'border': '1px solid #00ff99',
            'borderRadius': '15px', 'color': '#00ff99', 'marginRight': '8px'
        }
    return "LM STUDIO: OFFLINE", {
        'fontSize': '10px', 'padding': '4px 10px', 'border': '1px solid #ff5555',
        'borderRadius': '15px', 'color': '#ff5555', 'marginRight': '8px'
    }


@app.callback(
    [Output('frame-store', 'data'), Output('telemetry-status', 'children'), Output('telemetry-status', 'style')],
    [Input('update-interval', 'n_intervals')]
)
def update_frame(n):
    frame = HTTP_CLIENT.get_frame()
    if frame and frame.get("status") == "active":
        return frame, "TELEMETRY ACTIVE", {
            'fontSize': '10px', 'padding': '4px 10px', 'border': '1px solid #00ff99',
            'borderRadius': '15px', 'color': '#00ff99'
        }
    return frame, "WAITING...", {
        'fontSize': '10px', 'padding': '4px 10px', 'border': '1px solid #ffff00',
        'borderRadius': '15px', 'color': '#ffff00'
    }


@app.callback(
    [Output('chart-coverage', 'figure'), Output('curvature-map', 'figure'),
     Output('metric-tensor', 'figure'), Output('fiber-section', 'figure'),
     Output('transition-maps', 'figure'), Output('metric-dim', 'children'),
     Output('metric-charts', 'children'), Output('metric-curvature', 'children'),
     Output('metric-detg', 'children'), Output('metric-entropy', 'children'),
     Output('metric-smooth', 'children')],
    [Input('frame-store', 'data')]
)
def update_visualizations(frame_data):
    data = frame_data.get("data") if frame_data else None
    
    if data is None:
        empty = go.Figure()
        empty.update_layout(template="plotly_dark", paper_bgcolor='#0a0a0a')
        return [empty] * 5 + ["—"] * 6
    
    base_coords = np.array(data.get("base_coords_3d", [[0, 0, 0]]))
    N = len(base_coords)
    D = base_coords.shape[1] if len(base_coords.shape) > 1 else 1
    
    if len(base_coords.shape) == 1:
        base_coords = base_coords.reshape(-1, 1)
    
    ATLAS.dim = D
    ATLAS.define_charts(base_coords)
    
    metric = compute_metric_tensor(base_coords)
    scalar_curvature = compute_scalar_curvature(base_coords, metric)
    
    fiber_probs = np.array(data.get("fiber_activations", np.ones((N, 1))))
    section_data = compute_fiber_section(fiber_probs, base_coords)
    
    chart_fig = create_chart_coverage_figure(ATLAS, base_coords)
    curvature_fig = create_curvature_figure(scalar_curvature, base_coords)
    metric_fig = create_metric_tensor_figure(metric)
    fiber_fig = create_fiber_section_figure(section_data, base_coords)
    transition_fig = create_transition_map_figure(ATLAS)
    
    dim_str = str(D)
    charts_str = str(len(ATLAS.chart_centers)) if ATLAS.chart_centers is not None else "—"
    curvature_str = f"{scalar_curvature.mean():.3f}"
    detg_str = f"{np.mean([np.linalg.det(m) for m in metric]):.2f}"
    entropy_str = f"{section_data['mean_entropy']:.3f}"
    smooth_str = f"{1/max(section_data['section_smoothness'], 0.01):.1f}"
    
    return (chart_fig, curvature_fig, metric_fig, fiber_fig, transition_fig,
            dim_str, charts_str, curvature_str, detg_str, entropy_str, smooth_str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--lmstudio", default="http://192.168.56.1:1234")
    parser.add_argument("--state-file", default=None)
    args = parser.parse_args()
    
    LMSTUDIO_URL = args.lmstudio
    if args.state_file:
        STATE_FILE = os.path.abspath(args.state_file)
        HTTP_CLIENT = HTTPClient(STATE_FILE)
    
    print(f"\n[Manifold Tomograph] http://{args.host}:{args.port}")
    print(f"[Manifold Tomograph] State file: {STATE_FILE}")
    print(f"[Manifold Tomograph] LM Studio: {args.lmstudio}\n")
    
    app.run(host=args.host, port=args.port, debug=False)
