
import markdown
from weasyprint import HTML
import os

# Paths
input_md = "submission_package/manuscript.md"
output_pdf = "submission_package/ManifoldGL_Preprint.pdf"

# Read Markdown
with open(input_md, "r", encoding="utf-8") as f:
    text = f.read()

# Convert to HTML
html_content = markdown.markdown(text, extensions=['extra', 'codehilite'])

# Add some basic CSS for styling
css = """
body { font-family: sans-serif; line-height: 1.6; }
h1 { color: #2c3e50; }
h2 { color: #34495e; border-bottom: 1px solid #eee; }
code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
blockquote { border-left: 4px solid #ddd; padding-left: 10px; color: #777; }
"""

# Write to PDF
HTML(string=html_content).write_pdf(output_pdf, stylesheets=[])

print(f"Generated {output_pdf}")
