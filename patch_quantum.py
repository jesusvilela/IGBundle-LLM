import os

replacements = [
    # app_neural_glass.py
    ("app_neural_glass.py", '"gibbs_beta": 4.6,  # Effective inverse temperature (Rajakumar-Watson)', '"gibbs_beta": 4.6,  # Effective inverse temperature'),
    ("app_neural_glass.py", 'Per Rajakumar & Watson (2026): β > 1.87 → classically intractable.', 'Note: High β > 1.87 represents a highly coherent sampling regime.'),
    
    # poincare_enhanced.py
    ("poincare_enhanced.py", 'Per Rajakumar & Watson (2026): β > 1.87 → classically intractable.', 'Note: High β > 1.87 represents a highly coherent sampling regime.'),
    
    # test_scrambling_diagnostic.py
    ("test_scrambling_diagnostic.py", '- Gibbs temperature β: Per Rajakumar & Watson (2026)', '- Gibbs temperature β: Coherence Threshold'),
    ("test_scrambling_diagnostic.py", 'print(f"Rajakumar-Watson threshold: β > 1.87")', 'print(f"Coherence threshold: β > 1.87")'),
    
    # test_scrambling.py
    ("test_scrambling.py", 'print(f"Rajakumar-Watson threshold: β > 1.87")', 'print(f"Coherence threshold: β > 1.87")'),
    ("test_scrambling.py", 'print(f"\\n⚡ QUANTUM ADVANTAGE REGIME: Per Rajakumar & Watson (2026),")', 'print(f"\\n⚡ HIGH COHERENCE REGIME:")'),
    
    # src/igbundle/quantum/scrambling.py
    ("src/igbundle/quantum/scrambling.py", 'Reference: Rajakumar & Watson (2026) - Gibbs Sampling Quantum Advantage', 'Reference: Gibbs Sampling Coherence Limits'),
    ("src/igbundle/quantum/scrambling.py", 'above_hardness_threshold: bool  # β > 1.87 (Rajakumar-Watson)', 'above_hardness_threshold: bool  # β > 1.87'),
    ("src/igbundle/quantum/scrambling.py", 'Per Rajakumar & Watson (2026), β > 1.87 implies classical hardness.', 'β > 1.87 implies a high coherence sampling regime.'),
    
    # docs/POINCARE_VISUALIZATION_GUIDE.md
    ("docs/POINCARE_VISUALIZATION_GUIDE.md", 'Per Rajakumar & Watson (2026):', 'Notes:')
]

for file_rel, old_str, new_str in replacements:
    path = os.path.join(r"h:\LLM-MANIFOLD\igbundle-llm", file_rel.replace("/", "\\"))
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_str in content:
            content = content.replace(old_str, new_str)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Successfully replaced in {file_rel}")
        else:
            print(f"Warning: Exact string not found in {file_rel}")
    else:
        print(f"File not found: {path}")

print("Done patching.")
