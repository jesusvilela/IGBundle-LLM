# Deployment guide

The project website is served by **GitHub Pages** from the `/docs` folder of `main` (legacy branch deployment). `docs/.nojekyll` disables Jekyll, so Markdown files are served as raw text rather than rendered HTML; link to Markdown through its GitHub URL when rendering matters.

## Enable or verify

1. Open the repository **Settings** tab, then **Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**, branch `main`, folder `/docs`.
3. Save, then confirm that the latest Pages build uses the expected commit.

## Site contents

- `index.html`: research status and evidence overview.
- `wiki.html`: evidence, scope, and limitations, including historical caveats and what a stronger test needs.
- `demo.html`: research-access notes. No hosted demo or inference endpoint is provided.
- `benchmark-results/`: prompt-free result artifacts, provenance, and the offline verification report.
- `llms.txt`: machine-readable summary aligned with the pages above.

## Local preview and checks

Open `docs/index.html` in a browser. After changing evidence artifacts or local site links, regenerate and test the verification report:

```powershell
python scripts\verify_evidence.py --output docs\benchmark-results\verification.json
python -m pytest tests\test_evidence.py
```
