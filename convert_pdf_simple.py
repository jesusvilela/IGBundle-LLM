
from fpdf import FPDF
import os

# Paths
input_md = "submission_package/manuscript.md"
output_pdf = "submission_package/ManifoldGL_Preprint.pdf"

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'ManifoldGL Preprint', border=False, ln=1, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

# Create PDF
pdf = PDF()
pdf.add_page()
pdf.set_font("helvetica", size=12)

def sanitize(text):
    # Brutal sanitization to latin-1 compatible characters
    return text.encode('latin-1', 'replace').decode('latin-1')

# Read Markdown text
with open(input_md, "r", encoding="utf-8") as f:
    for line in f:
        clean_line = sanitize(line)
        if clean_line.startswith('# '):
            pdf.set_font("helvetica", 'B', 20)
            pdf.cell(0, 10, clean_line.strip('# \n'), ln=True)
            pdf.set_font("helvetica", size=12)
        elif clean_line.startswith('## '):
            pdf.set_font("helvetica", 'B', 16)
            pdf.cell(0, 10, clean_line.strip('# \n'), ln=True)
            pdf.set_font("helvetica", size=12)
        elif clean_line.startswith('### '):
            pdf.set_font("helvetica", 'B', 14)
            pdf.cell(0, 10, clean_line.strip('# \n'), ln=True)
            pdf.set_font("helvetica", size=12)
        else:
            pdf.multi_cell(0, 6, clean_line)

pdf.output(output_pdf)
print(f"Generated {output_pdf}")
