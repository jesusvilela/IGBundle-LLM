"""
Enhanced poll_telemetry for Neural Glass - Apply this patch
Run: python apply_poincare_patch.py
"""

ENHANCED_POLL_TELEMETRY = '''
# --- TELEMETRY POLLING ---
def get_zone_color(d):
    """Get color based on radial distance from center."""
    if d < 0.3:
        return 'rgba(0, 255, 100, 0.9)'   # Green - Anchor
    elif d < 0.6:
        return 'rgba(255, 255, 0, 0.9)'   # Yellow - Balanced
    elif d < 0.85:
        return 'rgba(255, 150, 0, 0.9)'   # Orange - Exploratory
    else:
        return 'rgba(255, 50, 50, 0.9)'   # Red - Boundary/Danger

def get_zone_name(d):
    """Get zone name based on radial distance."""
    if d < 0.3:
        return "ANCHOR"
    elif d < 0.6:
        return "BALANCED"
    elif d < 0.85:
        return "EXPLORATORY"
    else:
        return "BOUNDARY ⚠️"

def poll_telemetry():
    """
    Poll telemetry state and return enhanced visualization.
    
    ENHANCED Poincaré Projection with:
    - Cognitive zones (Anchor/Balanced/Exploratory)
    - Semantic direction labels (Abstract/Concrete/Creative/Analytical)
    - Color-coded trajectory by zone
    - Gibbs temperature indicator
    - Hover information for interpretability
    """
    try:
        # Update Gibbs temperature
        damping = TELEMETRY_STATE.get("damping", 0.01)
        beta = compute_gibbs_temperature(damping)
        TELEMETRY_STATE["gibbs_beta"] = beta
        
        # Create ENHANCED Poincaré projection
        fig = go.Figure()
        theta = np.linspace(0, 2*np.pi, 100)
        
        # --- COGNITIVE ZONES (Background rings) ---
        # Zone 1: ANCHOR (Green) - Center - System 1 thinking
        r_anchor = 0.3
        fig.add_trace(go.Scatter(
            x=r_anchor * np.cos(theta),
            y=r_anchor * np.sin(theta),
            fill='toself',
            fillcolor='rgba(0, 255, 100, 0.15)',
            line=dict(color='rgba(0, 255, 100, 0.4)', width=1, dash='dot'),
            name='Anchor',
            hoverinfo='text',
            hovertext='🟢 ANCHOR ZONE<br>High confidence | System 1<br>Stable semantics'
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
        
        # Zone 3: EXPLORATORY (Orange) - System 2 thinking
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
        
        # --- STABILITY BOUNDARY ---
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode='lines',
            line=dict(color='cyan', width=2),
            name='Boundary',
            hoverinfo='text',
            hovertext='⚠️ STABILITY BOUNDARY<br>Beyond = semantic instability'
        ))
        
        # --- SEMANTIC ANCHOR (Origin) ---
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
        if TELEMETRY_STATE["manifold_trace"]:
            arr = np.array(TELEMETRY_STATE["manifold_trace"])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                n_points = len(arr)
                
                # Calculate distances for zone coloring
                distances = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2)
                
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
                    zone = get_zone_name(d)
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
                        hovertext=f'⭐ CURRENT: {status}<br>Bundle: {TELEMETRY_STATE["active_fiber"]}<br>r={current_d:.3f}'
                    ))
        
        # --- LAYOUT WITH ANNOTATIONS ---
        rw_status = "✓" if beta > 1.87 else "✗"
        rw_color = 'lime' if beta > 1.87 else 'orange'
        
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
                # Semantic direction labels
                dict(x=0, y=1.12, text='<b>ABSTRACT</b>', showarrow=False,
                     font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                dict(x=0, y=-1.12, text='<b>CONCRETE</b>', showarrow=False,
                     font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                dict(x=1.12, y=0, text='<b>CREATIVE</b>', showarrow=False,
                     font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                dict(x=-1.12, y=0, text='<b>ANALYTICAL</b>', showarrow=False,
                     font=dict(size=7, color='rgba(100, 200, 255, 0.5)')),
                # Zone labels
                dict(x=0, y=0.15, text='ANCHOR', showarrow=False,
                     font=dict(size=6, color='rgba(0, 255, 100, 0.4)')),
                dict(x=0, y=0.45, text='BALANCED', showarrow=False,
                     font=dict(size=6, color='rgba(255, 255, 0, 0.4)')),
                dict(x=0, y=0.72, text='EXPLORE', showarrow=False,
                     font=dict(size=6, color='rgba(255, 150, 0, 0.4)')),
                # Bottom metrics
                dict(x=0.5, y=-0.12, xref='paper', yref='paper',
                     text=f'<b>β={beta:.1f}</b>{rw_status} | κ={TELEMETRY_STATE["curvature"]:.1f} | S={TELEMETRY_STATE["entropy"]:.2f}',
                     showarrow=False,
                     font=dict(size=9, color=rw_color))
            ]
        )
        
        return (
            f"{TELEMETRY_STATE['curvature']}",
            f"{TELEMETRY_STATE['entropy']}",
            f"{TELEMETRY_STATE['active_fiber']}",
            f"{TELEMETRY_STATE['constraint_score']:.2f}",
            "\\n".join(TELEMETRY_STATE["thought_trace"][-8:]),
            fig
        )
    except Exception as e:
        print(f"Telemetry Poll Error: {e}")
        return ("0.0", "0.0", "Error", "0.0", f"Error: {e}", go.Figure())
'''

import re

def apply_patch():
    app_path = r"H:\LLM-MANIFOLD\igbundle-llm\app_neural_glass.py"
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the poll_telemetry section
    # Pattern: from "# --- TELEMETRY POLLING ---" to "# --- UI BUILD ---"
    pattern = r'# --- TELEMETRY POLLING ---.*?(?=# --- UI BUILD ---)'
    
    replacement = ENHANCED_POLL_TELEMETRY.strip() + '\n\n'
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Backup original
    with open(app_path + '.bak', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Write patched version
    with open(app_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Patched {app_path}")
    print(f"   Backup saved to {app_path}.bak")

if __name__ == "__main__":
    apply_patch()
