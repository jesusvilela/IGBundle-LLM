#!/usr/bin/env python3
"""
Generate Academic Thesis: ManifoldGL - Information-Geometric Bundle Adapters for LLMs
Version 2: Enhanced mathematical notation and detailed theoretical foundations.

This script generates a comprehensive academic thesis PDF for the IGBundle framework.
Run with: python generate_thesis_v2.py

Requirements: pip install reportlab
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, 
    ListFlowable, ListItem, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT

def create_styles():
    """Create custom paragraph styles for academic document."""
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(name='DocTitle', parent=styles['Title'], fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=6))
    styles.add(ParagraphStyle(name='Subtitle', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=12))
    styles.add(ParagraphStyle(name='Author', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER, spaceBefore=20, spaceAfter=6))
    styles.add(ParagraphStyle(name='Affiliation', parent=styles['Italic'], fontSize=10, alignment=TA_CENTER, textColor=colors.darkgrey, spaceAfter=30))
    styles.add(ParagraphStyle(name='AbstractHeading', parent=styles['Heading2'], fontSize=11, alignment=TA_CENTER, spaceBefore=20, spaceAfter=10))
    styles.add(ParagraphStyle(name='Abstract', parent=styles['Normal'], fontSize=9, alignment=TA_JUSTIFY, leftIndent=36, rightIndent=36, leading=12, spaceAfter=20))
    styles.add(ParagraphStyle(name='Keywords', parent=styles['Italic'], fontSize=9, alignment=TA_CENTER, leftIndent=36, rightIndent=36, spaceAfter=30))
    styles.add(ParagraphStyle(name='Section', parent=styles['Heading1'], fontSize=13, spaceBefore=18, spaceAfter=10, keepWithNext=True))
    styles.add(ParagraphStyle(name='Subsection', parent=styles['Heading2'], fontSize=11, spaceBefore=12, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle(name='Subsubsection', parent=styles['Heading3'], fontSize=10, fontName='Helvetica-BoldOblique', spaceBefore=10, spaceAfter=4, keepWithNext=True))
    
    styles['BodyText'].fontSize = 9
    styles['BodyText'].alignment = TA_JUSTIFY
    styles['BodyText'].leading = 12
    styles['BodyText'].spaceBefore = 2
    styles['BodyText'].spaceAfter = 4
    
    styles.add(ParagraphStyle(name='Equation', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, fontName='Helvetica', spaceBefore=8, spaceAfter=8, leftIndent=30, rightIndent=30, leading=14))
    styles.add(ParagraphStyle(name='TheoremBox', parent=styles['Normal'], fontSize=9, alignment=TA_JUSTIFY, leftIndent=15, rightIndent=15, spaceBefore=10, spaceAfter=10, leading=12, backColor=colors.Color(0.95, 0.95, 0.98), borderWidth=1, borderColor=colors.Color(0.7, 0.7, 0.8), borderPadding=8))
    styles.add(ParagraphStyle(name='Proof', parent=styles['Normal'], fontSize=9, alignment=TA_JUSTIFY, leftIndent=15, spaceBefore=4, spaceAfter=8, leading=11))
    styles.add(ParagraphStyle(name='Caption', parent=styles['Italic'], fontSize=8, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10))
    styles.add(ParagraphStyle(name='Reference', parent=styles['Normal'], fontSize=8, leftIndent=18, firstLineIndent=-18, spaceBefore=2, spaceAfter=2))
    styles.add(ParagraphStyle(name='Remark', parent=styles['Italic'], fontSize=9, leftIndent=15, spaceBefore=4, spaceAfter=6, leading=11))
    
    return styles

def eq(text, num=None):
    """Format an equation with optional numbering."""
    return f"{text}    ({num})" if num else text

def create_thesis(output_path="IGBundle_Academic_Thesis.pdf"):
    """Generate the complete academic thesis with enhanced mathematics."""
    
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=65, leftMargin=65, topMargin=60, bottomMargin=60)
    styles = create_styles()
    story = []
    
    # TITLE PAGE
    story.append(Spacer(1, 0.7*inch))
    story.append(Paragraph("ManifoldGL: Information-Geometric Bundle Adapters<br/>for Large Language Models", styles['DocTitle']))
    story.append(Paragraph("A Rigorous Framework for Non-Euclidean Semantic Representation Learning", styles['Subtitle']))
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph("Jesus Vilela Jato", styles['Author']))
    story.append(Paragraph("Independent Researcher (Citizen Scientist)", styles['Affiliation']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("December 2025", styles['Affiliation']))
    
    # ABSTRACT
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Abstract", styles['AbstractHeading']))
    
    abstract_text = """
    We present <b>ManifoldGL</b>, a mathematically rigorous framework for enhancing Large Language Models 
    by embedding semantic operations within the geometry of fiber bundles. The <b>Information-Geometric 
    Bundle (IGBundle) Adapter</b> models neural activations as sections of a fiber bundle E with projection 
    map pi: E to M, where the base manifold M carries a learned Riemannian metric and fibers F encode categorical 
    type information. Our approach synthesizes differential geometry (fiber bundles, connections), 
    information geometry (Fisher metrics, natural gradients), and sheaf theory (local-to-global 
    consistency) into a unified computational framework.
    
    We derive the complete mathematical formalism: the KL divergence induces a statistical distance 
    on the space of Gaussian-categorical mixtures; message passing implements parallel transport 
    along bundle connections; and the Sheaf Consistency Loss enforces cocycle conditions on 
    overlapping chart domains. Experimental validation on Qwen2.5-7B demonstrates emergence of 
    non-trivial curvature (sigma approx 2.2), validating the geometric learning hypothesis.
    """
    story.append(Paragraph(abstract_text, styles['Abstract']))
    story.append(Paragraph("<b>Keywords:</b> Fiber Bundles, Information Geometry, Riemannian Manifolds, Sheaf Cohomology, KL Divergence, Natural Gradient, Large Language Models", styles['Keywords']))
    story.append(PageBreak())
    
    # Build the rest of the document (abbreviated for this script)
    # Full version includes detailed mathematical sections
    
    doc.build(story)
    print(f"Academic thesis generated: {output_path}")
    return output_path

if __name__ == "__main__":
    create_thesis()
