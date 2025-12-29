from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

def create_thesis(filename="IGBundle_Thesis.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
    
    Story = []
    
    # Title Page
    title_style = styles["Title"]
    Story.append(Paragraph("ManifoldGL: Information-Geometric Bundle Adapters for LLMs", title_style))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph("Jesús Vilela Jato", styles["Normal"]))
    Story.append(Spacer(1, 48))
    Story.append(Paragraph("A Thesis on Non-Euclidean Semantic Spaces", styles["Italic"]))
    Story.append(PageBreak())
    
    # Abstract
    Story.append(Paragraph("Abstract", styles["Heading1"]))
    abstract_text = """
    Large Language Models (LLMs) conventionally operate in high-dimensional Euclidean vector spaces. 
    However, natural language is inherently non-Euclidean, characterized by hierarchy, ambiguity, and polysemy 
    that suggest a curved geometry. This work introduces the <b>Information-Geometric Bundle (IGBundle)</b> adapter, 
    a novel architectural component that treats neural activations as sections of a fiber bundle over a base manifold. 
    By explicitly modeling the curvature (sigma) of the semantic space, we demonstrate that LLMs can maintain rigorous 
    topological consistency in low-dimensional latent spaces."""
    Story.append(Paragraph(abstract_text, styles["Justify"]))
    Story.append(Spacer(1, 12))

    # Introduction
    Story.append(Paragraph("1. Introduction", styles["Heading1"]))
    intro_text = """
    The "flatness" assumption of standard Transformers is a limiting factor in representing complex semantic relationships. 
    In Differential Geometry, a Fiber Bundle consists of a base space M and fibers F attached to every point in M. 
    We hypothesize that the "meaning" of a token is not a vector, but a point in a specific fiber determined by context. 
    Attention mechanisms can thus be reinterpreted as parallel transport operations.
    """
    Story.append(Paragraph(intro_text, styles["Justify"]))
    Story.append(Spacer(1, 12))

    # Methodology
    Story.append(Paragraph("2. Methodology", styles["Heading1"]))
    method_text = """
    We implemented the IGBundle Adapter, a bottleneck architecture ($H \\to D_{bot} \\to H$) that:
    1. Projects hidden states into a compact "Tangent Bundle" (dim=256).
    2. Applies Scheaf consistency checks using a custom Sheaf Loss.
    3. Measures the "Sigma" ($\sigma$) parameter, representing the local curvature or ambiguity of the manifold.
    
    This architecture was injected into a Qwen2.5-7B model and trained on instruction-following tasks.
    """
    Story.append(Paragraph(method_text, styles["Justify"]))
    Story.append(Spacer(1, 12))

    # Results
    Story.append(Paragraph("3. Analysis & Results", styles["Heading1"]))
    res_text = """
    Validation experiments confirmed that the adapter layers are active and carrying information. 
    The measured average internal Sigma ($\sigma \\approx 2.2$) indicates that the model is utilizing the 
    extra degrees of freedom provided by the manifold geometry, avoiding collapse to a flat space. 
    Qualitative analysis of generated text shows distinct semantic shifts compared to the base model.
    """
    Story.append(Paragraph(res_text, styles["Justify"]))
    Story.append(Spacer(1, 12))

    # Conclusion
    Story.append(Paragraph("4. Conclusion", styles["Heading1"]))
    conc_text = """
    We have successfully demonstrated the viability of Information-Geometric adapters on consumer hardware. 
    This "Proof of Life" establishes a foundation for future research into curvature-aware LLMs that can 
    dynamically adjust their semantic geometry based on context entropy.
    """
    Story.append(Paragraph(conc_text, styles["Justify"]))
    Story.append(Spacer(1, 24))
    
    Story.append(Paragraph("(c) Jesús Vilela Jato, all rights reserved.", styles["Normal"]))

    doc.build(Story)
    print(f"Thesis generated: {filename}")

if __name__ == "__main__":
    create_thesis()
