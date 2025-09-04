import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import requests

# --- Initialize Dash App ---
app = dash.Dash(__name__)
app.title = "Network Anomaly Detection Dashboard"

# --- Dashboard Layout ---
app.layout = html.Div([
    html.H1("Real-Time Network Anomaly Detection", style={"textAlign": "center", "color": "#2c3e50"}),

    dcc.Interval(id="interval-component", interval=3000, n_intervals=0),

    html.Div([
        html.H3("Anomaly Score Trend"),
        dcc.Graph(id="live-graph", figure=go.Figure().update_layout(title="Loading..."))
    ], style={"padding": "10px"}),

    html.Div([
        html.H3("Anomaly Frequency Over Time"),
        dcc.Graph(id="anomaly-frequency")
    ], style={"padding": "10px"}),

    html.Div([
        html.H3("Anomaly Breakdown by Type"),
        dcc.Graph(id="anomaly-category")
    ], style={"padding": "10px"}),

    html.Footer(
        "Dashboard powered by Dash + Plotly | Developed for Final Year Project",
        style={"textAlign": "center", "color": "#aaa", "paddingTop": "10px"}
    )
])
@app.callback(
    [Output("live-graph", "figure"),
     Output("anomaly-frequency", "figure"),
     Output("anomaly-category", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_dashboard(n):
    try:
        response = requests.get("http://127.0.0.1:5000/anomalies")
        if response.status_code == 200:
            data = response.json().get("anomaly_scores", [])
        else:
            print(f"Error fetching anomalies: HTTP {response.status_code} - {response.reason}")
            data = []

        if not data:
            return go.Figure().update_layout(title="No anomalies detected."), go.Figure(), go.Figure()

        df = pd.DataFrame(data, columns=["timestamp", "score", "category"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Live anomaly trend graph
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=df["timestamp"], y=df["score"],
            mode="lines+markers", marker=dict(color="crimson"), name="Anomaly Score"
        ))
        figure.update_layout(title="Anomaly Score Over Time")

        # Anomaly frequency graph
        frequency_graph = go.Figure()
        anomaly_freq = df.groupby(pd.Grouper(key="timestamp", freq="1T")).size()
        frequency_graph.add_trace(go.Bar(x=anomaly_freq.index, y=anomaly_freq.values, name="Anomaly Frequency"))
        frequency_graph.update_layout(title="Anomaly Frequency Over Time")

        # Anomaly category breakdown
        category_counts = df["category"].value_counts()
        category_graph = go.Figure()
        category_graph.add_trace(go.Pie(labels=category_counts.index, values=category_counts.values, name="Categories"))
        category_graph.update_layout(title="Anomaly Breakdown by Type")

        return figure, frequency_graph, category_graph

    except Exception as e:
        print(f"Error updating dashboard: {e}")
        return go.Figure(), go.Figure(), go.Figure()

if __name__ == "__main__":
    app.run_server(debug=False, port=8051)
