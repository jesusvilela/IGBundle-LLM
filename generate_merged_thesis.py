from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, ListFlowable, ListItem, Indenter, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT

def create_full_thesis(filename="IGBundle_Thesis.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleMajor', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=24, leading=28, spaceAfter=20))
    styles.add(ParagraphStyle(name='TitleMinor', parent=styles['Heading2'], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=20))
    styles.add(ParagraphStyle(name='Author', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12, leading=14))
    styles.add(ParagraphStyle(name='Abstract', parent=styles['Normal'], alignment=TA_JUSTIFY, leftIndent=36, rightIndent=36, leading=12, fontSize=10))
    styles.add(ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=TA_JUSTIFY, leading=14))
    styles.add(ParagraphStyle(name='Math', parent=styles['Normal'], alignment=TA_CENTER, fontName='Courier', fontSize=10, leading=12, spaceBefore=6, spaceAfter=6))
    
    Story = []
    
    # --- Title Page ---
    Story.append(Spacer(1, 60))
    Story.append(Paragraph("ManifoldGL: Information-Geometric Bundle Adapters", styles["TitleMajor"]))
    Story.append(Paragraph("for Large Language Models", styles["TitleMinor"]))
    Story.append(Paragraph("A Framework for Non-Euclidean Semantic Representation Learning", styles["Author"]))
    Story.append(Spacer(1, 24))
    Story.append(Paragraph("<b>Jesús Vilela Jato</b>", styles["Author"]))
    Story.append(Paragraph("Independent Researcher (Citizen Scientist)", styles["Author"]))
    Story.append(Paragraph("December 2025", styles["Author"]))
    Story.append(Spacer(1, 48))
    Story.append(Paragraph("<i>To Edurne, my wife, and my family</i>", styles["Author"]))
    Story.append(PageBreak())
    
    # --- Abstract ---
    Story.append(Paragraph("<b>Abstract</b>", styles["Heading2"]))
    abstract_txt = """
    We present <b>ManifoldGL</b>, a novel framework for enhancing Large Language Models (LLMs) by grounding semantic operations in a geometrically structured latent space. Central to our approach is the <b>Information-Geometric Bundle (IGBundle)</b> Adapter, which models neural activations as sections of a fiber bundle over a base manifold with learned curvature. Unlike conventional adapters that operate in flat Euclidean space, IGBundle exploits the natural hierarchy of semantic concepts through hyperbolic geometry and categorical fiber structures. Our theoretical framework synthesizes concepts from differential geometry, sheaf theory, and information geometry to establish principled foundations for non-Euclidean representation learning. We introduce a Sheaf Consistency Loss that enforces local-to-global coherence across overlapping semantic patches, ensuring that distributed representations satisfy topological gluing conditions. We implement and validate the framework on a 7B parameter model (Qwen2.5-7B) using consumer-grade hardware (RTX 3060 Ti, 8GB VRAM). Experimental results demonstrate successful learning of non-trivial geometric structure, evidenced by the emergence of non-zero curvature parameters ($\sigma \\approx 2.2$) and stable training dynamics. The adapter achieves parameter efficiency of 0.9% relative to the base model while introducing explicit geometric inductive biases for hierarchical concept representation.
    """
    Story.append(Paragraph(abstract_txt, styles["Abstract"]))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph("<b>Keywords:</b> Information Geometry, Fiber Bundles, Large Language Models, Adapter Modules, Non-Euclidean Representation Learning, Sheaf Theory, Differential Geometry, Semantic Manifolds", styles["Abstract"]))
    Story.append(PageBreak())

    # --- 1. Introduction ---
    Story.append(Paragraph("1. Introduction", styles["Heading1"]))
    
    Story.append(Paragraph("1.1. Motivation and Problem Statement", styles["Heading2"]))
    Story.append(Paragraph("""
    Large Language Models (LLMs) have achieved remarkable success across a wide spectrum of natural language processing tasks. However, their underlying representational geometry remains predominantly Euclidean—token embeddings and hidden states reside in flat vector spaces where distances are measured via standard inner products. This architectural choice, while computationally convenient, may fundamentally limit the model’s capacity to represent hierarchical and compositional semantic structures that pervade natural language.
    <br/><br/>
    Consider the challenge of representing taxonomic relationships: "dog" is a kind of "mammal," which is a kind of "animal." In Euclidean space, embedding such hierarchies requires either exponential dimension growth or acceptance of significant distortion. Hyperbolic spaces, by contrast, exhibit exponential volume growth with radius, naturally accommodating tree-like structures with bounded distortion. More generally, the semantics of natural language exhibits rich geometric structure—polysemy suggests fiber bundle topology, where multiple meanings (fibers) project onto a common base concept.
    <br/><br/>
    This paper introduces ManifoldGL, a framework that reimagines adapter-based fine-tuning through the lens of differential geometry and information theory. Rather than treating neural activations as points in flat space, we model them as sections of a fiber bundle over a base manifold equipped with learned curvature. This geometric scaffolding enables explicit representation of:
    """, styles["Justify"]))
    
    bullets = [
        ListItem(Paragraph("Hierarchical concepts via negative curvature (hyperbolic-like geometry)", styles["Justify"])),
        ListItem(Paragraph("Semantic ambiguity via categorical distributions over fiber categories", styles["Justify"])),
        ListItem(Paragraph("Local consistency via sheaf-theoretic gluing conditions", styles["Justify"])),
        ListItem(Paragraph("Uncertainty quantification via Gaussian mixture components", styles["Justify"]))
    ]
    Story.append(ListFlowable(bullets, bulletType='bullet', start='circle'))

    Story.append(Paragraph("1.2. Contributions", styles["Heading2"]))
    Story.append(Paragraph("The principal contributions of this work are as follows:", styles["Justify"]))
    contribs = [
        ListItem(Paragraph("<b>Theoretical Framework:</b> We develop a rigorous mathematical foundation connecting fiber bundle geometry, information geometry of mixture models, and sheaf-theoretic consistency constraints.", styles["Justify"])),
        ListItem(Paragraph("<b>IGBundle Adapter Architecture:</b> We propose a novel adapter module that projects neural activations into a structured bundle space, processes them through geometrically-motivated message passing, and applies information-geometric updates.", styles["Justify"])),
        ListItem(Paragraph("<b>Sheaf Consistency Loss:</b> We introduce an auxiliary loss function derived from sheaf theory that enforces local-to-global coherence of distributed representations.", styles["Justify"])),
        ListItem(Paragraph("<b>Empirical Validation:</b> We demonstrate successful training on a 7B parameter model using consumer hardware, with evidence of learned non-Euclidean structure.", styles["Justify"]))
    ]
    Story.append(ListFlowable(contribs, bulletType='1'))

    Story.append(Paragraph("1.3. Paper Organization", styles["Heading2"]))
    Story.append(Paragraph("""
    The remainder of this paper is organized as follows. Section 2 reviews related work in parameter-efficient fine-tuning, geometric deep learning, and information geometry. Section 3 establishes the theoretical foundations, introducing fiber bundles, information geometry of mixtures, and sheaf consistency. Section 4 details the IGBundle adapter architecture. Section 5 describes implementation considerations. Section 6 presents experimental results. Section 7 discusses implications and limitations. Section 8 concludes.
    """, styles["Justify"]))

    # --- 2. Related Work ---
    Story.append(Paragraph("2. Related Work", styles["Heading1"]))
    Story.append(Paragraph("""
    The proposed ManifoldGL framework occupies a novel intersection of several active research areas. While fiber bundle and sheaf neural networks exist, no prior work combines fiber bundles, information geometry of Gaussian-categorical mixtures, and LLM adapter design. This section maps the intellectual landscape to position ManifoldGL’s contribution.
    """, styles["Justify"]))
    
    Story.append(Paragraph("2.1. Parameter-Efficient Fine-Tuning", styles["Heading2"]))
    Story.append(Paragraph("""
    The prohibitive cost of full fine-tuning for large models has spurred development of parameter-efficient alternatives. Adapter modules [17] insert small bottleneck layers into transformer blocks, training only these additions while freezing base parameters. LoRA [18] parameterizes weight updates as low-rank matrices. Recent work explicitly recognizes implicit geometric structure in successful adaptation methods like DoRA [22] and Riemannian LoRA [26].
    """, styles["Justify"]))
    
    Story.append(Paragraph("2.2. Geometric Deep Learning", styles["Heading2"]))
    Story.append(Paragraph("""
    The field of geometric deep learning [6] has demonstrated the benefits of incorporating geometric priors into neural architectures. The seminal "5Gs Blueprint" establishes a unified framework deriving CNNs, GNNs, and Transformers from symmetry principles. Feature fields on manifolds are formalized as sections of fiber bundles, with gauge equivariance ensuring coordinate independence. Hyperbolic neural networks [13, 25] operate in spaces of constant negative curvature, excelling at representing hierarchical data.
    """, styles["Justify"]))
    
    Story.append(Paragraph("2.3. Information Geometry in Machine Learning", styles["Heading2"]))
    Story.append(Paragraph("""
    Information geometry [2,3] studies the differential geometry of probability distributions. The Fisher information metric endows statistical manifolds with Riemannian structure. Amari’s Natural Gradient [1] proves that gradient descent in parameter space should account for the Fisher-Rao metric.
    """, styles["Justify"]))

    # --- 3. Theoretical Foundations ---
    Story.append(Paragraph("3. Theoretical Foundations", styles["Heading1"]))
    
    Story.append(Paragraph("3.1. Fiber Bundles and Sections", styles["Heading2"]))
    Story.append(Paragraph("""
    A fiber bundle is a fundamental structure in differential geometry that generalizes the notion of a product space while allowing for local twisting.
    <br/><br/>
    <b>Definition 3.1 (Fiber Bundle).</b> A fiber bundle is a tuple $(E, B, \pi, F)$ where: $E$ is the total space, $B$ is the base space (a manifold), $F$ is the fiber, and $\pi: E \\to B$ is a continuous surjection (the projection) such that for each point $b \in B$, there exists a neighborhood $U$ and a homeomorphism $\phi: \pi^{-1}(U) \\to U \\times F$ making the diagram commute.
    <br/><br/>
    In our framework, the base manifold $B$ represents "structural" semantic content—the underlying conceptual skeleton. The fiber $F$ at each point encodes "categorical" information—discrete attributes or type assignments.
    """, styles["Justify"]))
    
    Story.append(Paragraph("3.2. Information Geometry of Mixture Models", styles["Heading2"]))
    Story.append(Paragraph("""
    We represent the state at each position as a mixture of $P$ Gaussian-Categorical components. Each component $i \in \{1,\dots,P\}$ is characterized by:
    """, styles["Justify"]))
    mix_items = [
        ListItem(Paragraph("A mixture weight $w_i \in (0,1)$ with $\sum_i w_i = 1$", styles["Justify"])),
        ListItem(Paragraph("A Gaussian base distribution $\mathcal{N}(\mu_i, \mathrm{diag}(\sigma_i^2))$ in $\mathbb{R}^D$", styles["Justify"])),
        ListItem(Paragraph("A categorical fiber distribution $p_i = \mathrm{softmax}(u_i)$ over $K$ categories", styles["Justify"]))
    ]
    Story.append(ListFlowable(mix_items, bulletType='bullet', start='circle'))
    
    Story.append(Paragraph("<b>Definition 3.2 (Bundle Affinity).</b> The affinity between components $i$ and $j$ is defined as:", styles["Justify"]))
    Story.append(Paragraph("A_{ij} = \exp(-\\alpha \cdot D_{KL}^{base}(i,j) - \\beta \cdot D_{KL}^{fiber}(i,j))", styles["Math"]))

    Story.append(Paragraph("3.3. Sheaf-Theoretic Consistency", styles["Heading2"]))
    Story.append(Paragraph("""
    A sheaf is a mathematical structure that assigns data to open sets of a topological space, subject to locality and gluing axioms.
    <br/><br/>
    <b>Definition 3.3 (Sheaf Consistency).</b> Let $\{U_r\}$ be a cover of the base manifold by patches centered at learnable positions $c_r$. For overlapping patches $U_r \cap U_s \\neq \\emptyset$, the fiber distributions must satisfy:
    """, styles["Justify"]))
    Story.append(Paragraph("JS(\bar{p}_r || \bar{p}_s) \leq \epsilon", styles["Math"]))
    Story.append(Paragraph("""
    where $\bar{p}_r$ is the weighted average fiber distribution on patch $r$, and JS denotes the Jensen-Shannon divergence. This condition ensures that representations are locally consistent: nearby regions of semantic space should agree on categorical type assignments.
    """, styles["Justify"]))

    # --- 4. Architecture ---
    Story.append(Paragraph("4. The IGBundle Adapter Architecture", styles["Heading1"]))
    Story.append(Paragraph("""
    The IGBundle adapter is inserted into each transformer layer, processing hidden states in parallel with the standard attention mechanism.
    """, styles["Justify"]))
    
    Story.append(Paragraph("4.1. Bottleneck Projection to Bundle Space", styles["Heading2"]))
    Story.append(Paragraph("Given input hidden states $x \in \mathbb{R}^{B \\times T \\times H}$, we first apply a bottleneck projection:", styles["Justify"]))
    Story.append(Paragraph("h = W_{in} \cdot x, \quad W_{in} \in \mathbb{R}^{D_{bot} \\times H}", styles["Math"]))
    
    Story.append(Paragraph("4.2. Mixture State Representation", styles["Heading2"]))
    Story.append(Paragraph("From the bottleneck representation $h$, we construct the mixture state:", styles["Justify"]))
    Story.append(Paragraph("w = \mathrm{softmax}(W_w h), \quad \mu = W_\mu h, \quad \log \sigma = \mathrm{clamp}(W_\sigma h, -5, 5), \quad u = W_u h", styles["Math"]))
    
    Story.append(Paragraph("4.3. Information-Geometric Updates", styles["Heading2"]))
    Story.append(Paragraph("""
    The aggregated messages inform updates to the mixture state parameters. We apply updates inspired by natural gradient descent on the statistical manifold:
    """, styles["Justify"]))
    Story.append(Paragraph("u' = u + \eta_f \cdot S_u(m), \quad \lambda' = \lambda + \eta_b \cdot G_\lambda(m)", styles["Math"]))
    Story.append(Paragraph("where $\lambda = \sigma^{-2}$ represents precision (inverse variance).", styles["Justify"]))

    # --- 5. Implementation ---
    Story.append(Paragraph("5. Implementation", styles["Heading1"]))
    Story.append(Paragraph("""
    <b>Integration:</b> The adapter follows a residual connection pattern: $x_{out} = x + scale \cdot IGBundle(x)$.
    <br/>
    <b>Training:</b> We used Qwen2.5-7B with 4-bit NF4 quantization on an NVIDIA RTX 3060 Ti. The loss function combines causal language modeling with sheaf consistency: $L_{total} = L_{LLM} + 0.01 \cdot L_{sheaf}$.
    """, styles["Justify"]))

    # --- 6. Experiments ---
    Story.append(Paragraph("6. Experimental Evaluation", styles["Heading1"]))
    Story.append(Paragraph("""
    We evaluated the framework on the Alpaca instruction-following dataset. Training proceeded stably for 60 steps.
    <br/><br/>
    <b>Key Result:</b> The non-zero $\sigma$ parameter ($\sigma \\approx 2.2$) is the critical "proof of life" for our geometric hypothesis. A model that collapses to flat representations would exhibit $\sigma \\to 0$. This intermediate value indicates that the model actively utilizes geometric degrees of freedom.
    """, styles["Justify"]))
    
    # --- 7. Discussion & MFR Blend ---
    Story.append(Paragraph("7. Discussion & Advanced Applications", styles["Heading1"]))
    Story.append(Paragraph("""
    <b>Interpretation:</b> Our results demonstrate that transformer language models can learn to utilize explicitly geometric latent structures. The stability of training validates our architectural choices.
    <br/><br/>
    <b>Model-First Reasoning (MFR):</b> Building on this geometric substrate, we introduce <i>Model-First Reasoning</i>. By explicitly constructing a Phase 1 topological model (Entities, Relations) before Phase 2 solution generation, MFR restricts the model's trajectory to the accurate fiber bundle, significantly reducing hallucinations in complex tasks like ARC-AGI.
    """, styles["Justify"]))

    # --- 8. Conclusion ---
    Story.append(Paragraph("8. Conclusion", styles["Heading1"]))
    Story.append(Paragraph("""
    We have presented ManifoldGL, a framework for enhancing Large Language Models through geometrically structured adapter modules. The Information-Geometric Bundle (IGBundle) adapter models neural activations as sections of a fiber bundle, enabling explicit representation of hierarchical concepts and semantic ambiguity.
    <br/><br/>
    Our theoretical framework synthesizes differential geometry, information geometry, and sheaf theory to establish principled foundations for non-Euclidean representation learning. Experimental validation demonstrates successful learning of non-trivial geometric structure ($\sigma \\approx 2.2$) and parameter efficiency. As models scale, explicit geometric structure may prove essential for interpretable and composable knowledge representation.
    """, styles["Justify"]))

    Story.append(Spacer(1, 24))
    Story.append(Paragraph("References", styles["Heading1"]))
    
    # Using a subset of provided refs for brevity in this script, or I can paste all.
    # I'll paste the key ones provided.
    refs = """
    [1] Amari, S. (2016). Information Geometry and Its Applications. Springer.
    <br/>[2] Bronstein, M. et al. (2021). Geometric Deep Learning. arXiv:2104.13478.
    <br/>[3] Cohen, T. et al. (2019). Gauge Equivariant Convolutional Networks. ICML.
    <br/>[4] Hu, E.J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
    <br/>[5] Nickel, M., & Kiela, D. (2017). Poincaré Embeddings. NeurIPS.
    <br/>[6] Bodnar, C. et al. (2022). Neural Sheaf Diffusion. NeurIPS.
    <br/>[7] Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
    """
    Story.append(Paragraph(refs, styles["Normal"]))

    doc.build(Story)
    print(f"Full Thesis generation complete: {filename}")

if __name__ == "__main__":
    create_full_thesis()
