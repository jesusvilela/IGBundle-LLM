
from dash import Dash, html
print("Imported Dash.")
app = Dash(__name__)
app.layout = html.Div("Hello World")
print("Layout set. Running server on 8052...")
try:
    app.run_server(debug=False, port=8052)
except Exception as e:
    print(f"Dash Failed: {e}")
