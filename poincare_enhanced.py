"""
Enhanced Poincaré Visualization Module for Neural Glass
========================================================

This module provides an interpretable visualization of the model's
semantic position in hyperbolic space during text generation.

Zones:
- ANCHOR (center): High confidence, System 1 thinking
- BALANCED (middle): Weighing options, moderate certainty  
- EXPLORATORY (outer): Deep analysis, System 2 thinking
- BOUNDARY (edge): Semantic instability warning

Author: Jesús Vilela Jato
Date: February 2026
"""

import numpy as np
import plotly.graph_objects as go
import math

def compute_gibbs_temperature(damping: float) -> float:
    """
    Compute effective inverse temperature β from damping parameter.
    Note: High β > 1.87 represents a highly coherent sampling regime.
    """
    if damping > 0 and damping < 1:
        return -math.log(damping / (1 - damping))
    return float('inf')


def create_enhanced_poincare_plot(
    manifold_trace: list,
    curvature: float = -1.0,
    entropy: float = 0.0,
    active_fiber: str = "None",
    constraint_score: float = 1.0,
    damping: float = 0.01,
    show_labels: bool = True
) -> go.Figure:
    """
    Create an enhanced, interpretable Poincaré disk visualization.
    
    Args:
        manifold_trace: List of (x, y) coordinates representing thought trajectory
        curvature: Sectional curvature K
        entropy: Manifold entropy S
        active_fiber: Currently active bundle name
        constraint_score: Constraint satisfaction score (0-1)
        damping: Damping parameter for Gibbs temperature
        show_labels: Whether to show zone labels
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 100)
    
    # --- COGNITIVE ZONES (Background rings) ---
    
    # Zone 1: ANCHOR (Green) - Center
    r_anchor = 0.3
    fig.add_trace(go.Scatter(
        x=r_anchor * np.cos(theta),
        y=r_anchor * np.sin(theta),
        fill='toself',
        fillcolor='rgba(0, 255, 100, 0.15)',
        line=dict(color='rgba(0, 255, 100, 0.4)', width=1, dash='dot'),
        name='Anchor',
        hoverinfo='text',
        hovertext='🟢 ANCHOR ZONE<br>High confidence<br>System 1 thinking<br>Stable semantics'
    ))
    
    # Zone 2: BALANCED (Yellow)
    r_balanced = 0.6
    fig.add_trace(go.Scatter(
        x=r_balanced * np.cos(theta),
        y=r_balanced * np.sin(theta),
        fill='tonext',
        fillcolor='rgba(255, 255, 0, 0.08)',
        line=dict(color='rgba(255, 255, 0, 0.3)', width=1, dash='dot'),
        name='Balanced',
        hoverinfo='text',
        hovertext='🟡 BALANCED ZONE<br>Weighing options<br>Moderate certainty'
    ))
    
    # Zone 3: EXPLORATORY (Orange)
    r_explore = 0.85
    fig.add_trace(go.Scatter(
        x=r_explore * np.cos(theta),
        y=r_explore * np.sin(theta),
        fill='tonext',
        fillcolor='rgba(255, 150, 0, 0.08)',
        line=dict(color='rgba(255, 150, 0, 0.3)', width=1, dash='dot'),
        name='Exploratory',
        hoverinfo='text',
        hovertext='🟠 EXPLORATORY ZONE<br>System 2 thinking<br>High uncertainty'
    ))
    
    # --- BOUNDARY CIRCLE ---
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Boundary',
        hoverinfo='text',
        hovertext='⚠️ STABILITY BOUNDARY<br>Beyond = semantic instability'
    ))
    
    # --- SEMANTIC DIRECTION INDICATORS ---
    if show_labels:
        # Cardinal directions with semantic meaning
        labels = [
            (0, 1.08, 'ABSTRACT', 'top center'),
            (0, -1.08, 'CONCRETE', 'bottom center'),
            (1.08, 0, 'CREATIVE', 'middle right'),
            (-1.08, 0, 'ANALYTICAL', 'middle left'),
        ]
        
        for x, y, text, _ in labels:
            fig.add_annotation(
                x=x, y=y,
                text=f"<b>{text}</b>",
                showarrow=False,
                font=dict(size=8, color='rgba(100, 200, 255, 0.6)'),
                xanchor='center',
                yanchor='middle'
            )
        
        # Zone labels
        zone_labels = [
            (0, 0.15, 'ANCHOR', 'lime'),
            (0, 0.45, 'BALANCED', 'yellow'),
            (0, 0.72, 'EXPLORE', 'orange'),
        ]
        
        for x, y, text, color in zone_labels:
            fig.add_annotation(
                x=x, y=y,
                text=text,
                showarrow=False,
                font=dict(size=7, color=f'rgba({get_rgb(color)}, 0.5)'),
                xanchor='center'
            )
    
    # --- ORIGIN MARKER ---
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=10, color='lime', symbol='diamond',
                   line=dict(color='white', width=1)),
        name='Anchor',
        hoverinfo='text',
        hovertext='◆ SEMANTIC ANCHOR<br>Origin of meaning'
    ))
    
    # --- THOUGHT TRAJECTORY ---
    if manifold_trace and len(manifold_trace) > 0:
        arr = np.array(manifold_trace)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            n_points = len(arr)
            
            # Calculate distances from center for coloring
            distances = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2)
            
            # Color by zone: green (center) → yellow → orange → red (edge)
            def get_zone_color(d):
                if d < 0.3:
                    return 'rgba(0, 255, 100, 0.9)'  # Green
                elif d < 0.6:
                    return 'rgba(255, 255, 0, 0.9)'  # Yellow
                elif d < 0.85:
                    return 'rgba(255, 150, 0, 0.9)'  # Orange
                else:
                    return 'rgba(255, 50, 50, 0.9)'  # Red (danger)
            
            # Trajectory line with gradient effect
            for i in range(len(arr) - 1):
                opacity = 0.3 + 0.7 * i / max(n_points - 1, 1)
                fig.add_trace(go.Scatter(
                    x=[arr[i, 0], arr[i+1, 0]],
                    y=[arr[i, 1], arr[i+1, 1]],
                    mode='lines',
                    line=dict(color=f'rgba(255, 0, 255, {opacity})', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Trajectory points colored by zone
            colors = [get_zone_color(d) for d in distances]
            sizes = [4 + 8 * i/max(n_points, 1) for i in range(n_points)]
            
            hover_texts = []
            for i, d in enumerate(distances):
                if d < 0.3:
                    zone = "ANCHOR"
                elif d < 0.6:
                    zone = "BALANCED"
                elif d < 0.85:
                    zone = "EXPLORATORY"
                else:
                    zone = "BOUNDARY ⚠️"
                hover_texts.append(f'Step {i+1}<br>Zone: {zone}<br>Radius: {d:.2f}')
            
            fig.add_trace(go.Scatter(
                x=arr[:, 0], y=arr[:, 1],
                mode='markers',
                marker=dict(size=sizes, color=colors,
                           line=dict(color='white', width=0.5)),
                name='Steps',
                hoverinfo='text',
                hovertext=hover_texts
            ))
            
            # Current position (prominent star marker)
            if n_points > 0:
                current_d = distances[-1]
                if current_d < 0.3:
                    star_color = 'lime'
                    status = 'ANCHORED'
                elif current_d < 0.6:
                    star_color = 'yellow'
                    status = 'BALANCED'
                elif current_d < 0.85:
                    star_color = 'orange'
                    status = 'EXPLORING'
                else:
                    star_color = 'red'
                    status = 'UNSTABLE'
                
                fig.add_trace(go.Scatter(
                    x=[arr[-1, 0]], y=[arr[-1, 1]],
                    mode='markers',
                    marker=dict(size=14, color=star_color, symbol='star',
                               line=dict(color='white', width=2)),
                    name='Current',
                    hoverinfo='text',
                    hovertext=f'⭐ CURRENT POSITION<br>Status: {status}<br>Bundle: {active_fiber}<br>Radius: {current_d:.3f}'
                ))
    
    # --- COMPUTE GIBBS TEMPERATURE ---
    beta = compute_gibbs_temperature(damping)
    rw_status = "✓" if beta > 1.87 else "✗"
    rw_color = 'lime' if beta > 1.87 else 'orange'
    
    # --- CONSTRAINT INDICATOR ---
    cs_color = 'lime' if constraint_score > 0.5 else ('orange' if constraint_score > 0.2 else 'red')
    
    # --- LAYOUT ---
    fig.update_layout(
        title=dict(
            text="Poincaré Manifold Projection",
            font=dict(size=12, color='cyan'),
            x=0.5
        ),
        template="plotly_dark",
        xaxis=dict(
            range=[-1.25, 1.25], 
            showgrid=False, 
            zeroline=False,
            showticklabels=False, 
            title=None,
            fixedrange=True
        ),
        yaxis=dict(
            range=[-1.25, 1.25], 
            showgrid=False, 
            zeroline=False,
            showticklabels=False, 
            title=None, 
            scaleanchor='x',
            fixedrange=True
        ),
        width=320, 
        height=320,
        margin=dict(l=5, r=5, t=35, b=45),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        annotations=[
            # Bottom metrics
            dict(
                x=0.5, y=-0.12, 
                xref='paper', yref='paper',
                text=f'<b>β={beta:.1f}</b> {rw_status}  |  κ={curvature:.1f}  |  S={entropy:.2f}',
                showarrow=False,
                font=dict(size=9, color=rw_color)
            ),
        ]
    )
    
    return fig


def get_rgb(color_name: str) -> str:
    """Convert color name to RGB values string."""
    colors = {
        'lime': '0, 255, 0',
        'yellow': '255, 255, 0',
        'orange': '255, 150, 0',
        'red': '255, 50, 50',
        'cyan': '0, 255, 255',
        'magenta': '255, 0, 255',
    }
    return colors.get(color_name, '255, 255, 255')


# --- TEST ---
if __name__ == "__main__":
    # Generate sample trajectory
    np.random.seed(42)
    n_steps = 30
    
    # Simulate a thought trajectory that starts centered, explores, returns
    angles = np.linspace(0, 4*np.pi, n_steps) + np.random.randn(n_steps) * 0.3
    radii = 0.2 + 0.4 * np.sin(np.linspace(0, 2*np.pi, n_steps)) + np.random.randn(n_steps) * 0.05
    radii = np.clip(radii, 0.05, 0.9)
    
    trace = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles)]
    
    fig = create_enhanced_poincare_plot(
        manifold_trace=trace,
        curvature=-1.0,
        entropy=2.77,
        active_fiber="Bundle-5",
        constraint_score=0.85,
        damping=0.01
    )
    
    fig.write_html("poincare_test.html")
    print("Saved poincare_test.html")
