
from mdpdf.converter import Converter
import os

output_pdf = "submission_package/ManifoldGL_Preprint.pdf"
input_md = "submission_package/manuscript.md"

if not os.path.exists("submission_package"):
    os.makedirs("submission_package")

converter = Converter(output_pdf)
converter.convert([input_md])

print(f"Generated {output_pdf}")
