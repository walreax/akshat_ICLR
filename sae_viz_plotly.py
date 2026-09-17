"""
Interactive SAE metrics visualization (Plotly)

Reads a JSON file containing SAE / concept metrics (structure like the sample you provided)
and produces an interactive HTML dashboard with dropdowns to switch between top-level records.

Plots included per record:
- Bar chart of attention_entropy by concept
- Bar chart of attention_concentration_index by concept
- 2D scatter of concept centroids (x, y)
- Heatmap of centroid_deviation distances (pairwise)

Usage:
    python sae_viz_plotly.py --input path/to/metrics.json --output sae_metrics.html

The script tries to import plotly and pandas. Install with:
    pip install plotly pandas

"""

from pathlib import Path
import json
import argparse
import math

try:
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    raise RuntimeError(
        "Missing dependency: ensure 'pandas' and 'plotly' are installed (pip install pandas plotly)"
    ) from e


def load_metrics(json_path):
    """Load the nested JSON and flatten into per-record DataFrames.

    Returns:
        records: list of record names (top-level keys)
        metrics_by_record: dict(record_name -> pd.DataFrame) with columns:
            ['concept', 'centroid_x', 'centroid_y', 'attention_entropy', 'attention_concentration_index']
        deviations_by_record: dict(record_name -> pd.DataFrame) square matrix of pairwise distances
    """
    p = Path(json_path)
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # Data might be a list of items (as in sample) or a dict
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [data]
    else:
        raise ValueError("Unsupported JSON top-level type: expected list or dict")

    metrics_by_record = {}
    deviations_by_record = {}
    records = []

    # Each item in items might contain one or more top-level named records
    for item in items:
        if not isinstance(item, dict):
            continue
        for rec_name, rec_val in item.items():
            records.append(rec_name)
            concept_rows = []

            # concept_metrics: {concept_name: {centroid: [x,y], attention_entropy: val, ...}}
            cm = rec_val.get('concept_metrics', {})
            for concept, cvals in cm.items():
                centroid = cvals.get('centroid', [math.nan, math.nan])
                attention_entropy = cvals.get('attention_entropy', None)
                attention_concentration_index = cvals.get('attention_concentration_index', None)
                concept_rows.append(
                    {
                        'concept': concept,
                        'centroid_x': centroid[0] if len(centroid) > 0 else math.nan,
                        'centroid_y': centroid[1] if len(centroid) > 1 else math.nan,
                        'attention_entropy': attention_entropy,
                        'attention_concentration_index': attention_concentration_index,
                    }
                )

            df = pd.DataFrame(concept_rows)
            metrics_by_record[rec_name] = df

            # centroid_deviation: nested pairs with 'distance' values
            cd = rec_val.get('centroid_deviation', {})
            # centroid_deviation keys might be 'a__b' or similar; build matrix
            concepts = sorted(df['concept'].tolist())
            dev_mat = pd.DataFrame(0.0, index=concepts, columns=concepts)

            for pair_key, pair_val in cd.items():
                # try to infer concept names from pair_key like 'hourglass__time'
                if '__' in pair_key:
                    a, b = pair_key.split('__', 1)
                else:
                    # fallback: attempt to read nested keys
                    a = pair_key
                    b = pair_key
                distance = None
                # pair_val may be dict with 'distance'
                if isinstance(pair_val, dict) and 'distance' in pair_val:
                    distance = pair_val['distance']
                # Or it might be numeric directly
                if distance is None and isinstance(pair_val, (int, float)):
                    distance = float(pair_val)
                if distance is not None:
                    if a in dev_mat.index and b in dev_mat.columns:
                        dev_mat.loc[a, b] = float(distance)
                        dev_mat.loc[b, a] = float(distance)
            deviations_by_record[rec_name] = dev_mat

    return records, metrics_by_record, deviations_by_record


def make_dashboard(records, metrics_by_record, deviations_by_record, out_html, show_all=False):
    """Create a Plotly figure with subplots and a dropdown to switch between records.

    Parameters:
        show_all: if True, all records' traces are visible by default and the dropdown includes an "All" option.
    """
    # layout: 2 rows x 2 cols
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.46, 0.54],
        row_heights=[0.45, 0.55],
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "heatmap"}]],
        subplot_titles=(
            "Attention Entropy by Concept",
            "Attention Concentration Index by Concept",
            "Centroid scatter (x vs y)",
            "Centroid Deviation distance matrix",
        ),
    )

    all_traces = []
    visible_flags = []

    for i, rec in enumerate(records):
        df = metrics_by_record.get(rec, pd.DataFrame())
        dev = deviations_by_record.get(rec, pd.DataFrame())

        # ensure sorted concept order to align heatmap
        concepts = df['concept'].tolist()

        # Bar: attention_entropy
        trace_entropy = go.Bar(
            x=df['concept'],
            y=df['attention_entropy'],
            name=f"entropy_{rec}",
            marker=dict(color='darkorange'),
            hovertemplate="%{x}<br>entropy: %{y:.4f}<extra></extra>",
        )
        fig.add_trace(trace_entropy, row=1, col=1)
        all_traces.append(trace_entropy)

        # Bar: attention_concentration_index
        trace_conc = go.Bar(
            x=df['concept'],
            y=df['attention_concentration_index'],
            name=f"conc_{rec}",
            marker=dict(color='teal'),
            hovertemplate="%{x}<br>concentration: %{y:.4f}<extra></extra>",
        )
        fig.add_trace(trace_conc, row=1, col=2)
        all_traces.append(trace_conc)

        # Scatter: centroids
        trace_cent = go.Scatter(
            x=df['centroid_x'],
            y=df['centroid_y'],
            mode='markers+text',
            text=df['concept'],
            textposition='top center',
            marker=dict(size=12, color='royalblue', opacity=0.8),
            name=f"centroids_{rec}",
            hovertemplate="%{text}<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>",
        )
        fig.add_trace(trace_cent, row=2, col=1)
        all_traces.append(trace_cent)

        # Heatmap: centroid deviations
        if not dev.empty:
            z = dev.values
            trace_heat = go.Heatmap(
                z=z,
                x=dev.columns.tolist(),
                y=dev.index.tolist(),
                colorscale='Viridis',
                colorbar=dict(title='distance'),
                zmin=0,
                name=f"dev_{rec}",
                hovertemplate="%{y} vs %{x}<br>distance: %{z:.4f}<extra></extra>",
            )
        else:
            # empty placeholder
            trace_heat = go.Heatmap(
                z=[[0]],
                x=["-"],
                y=["-"],
                colorscale='Viridis',
                showscale=False,
                name=f"dev_{rec}",
            )
        fig.add_trace(trace_heat, row=2, col=2)
        all_traces.append(trace_heat)

        # visibility: either show all or only traces for the first record
        is_visible = show_all or (i == 0)
        visible_flags.extend([is_visible] * 4)

    # Set all traces visibility according to visible_flags
    for t, vis in zip(fig.data, visible_flags):
        t.visible = vis

    # Build dropdown buttons
    buttons = []
    n_traces_per_record = 4
    total_records = len(records)

    # Add an "All" option first
    all_vis = [True] * (n_traces_per_record * total_records)
    buttons.append(dict(label="All", method="update", args=[{"visible": all_vis}, {"title": "SAE metrics: All"}]))

    for i, rec in enumerate(records):
        vis = [False] * (n_traces_per_record * total_records)
        start = i * n_traces_per_record
        for j in range(n_traces_per_record):
            vis[start + j] = True

        btn = dict(
            label=rec,
            method="update",
            args=[{"visible": vis}, {"title": f"SAE metrics: {rec}"}],
        )
        buttons.append(btn)


    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="top",
            )
        ],
        height=800,
        title=f"SAE metrics: {records[0] if records else ''}",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0.0),
    )

    # Tidy axes and subplot layout
    fig.update_xaxes(tickangle=45, automargin=True)
    fig.update_yaxes(automargin=True)

    # Write to HTML
    out_p = Path(out_html)
    fig.write_html(out_p, include_plotlyjs='cdn')
    print(f"Wrote interactive SAE visualization to: {out_p}")


def main():
    parser = argparse.ArgumentParser(description="Plot SAE metrics (interactive Plotly)")
    parser.add_argument('--input', '-i', required=True, help='Path to input JSON file')
    parser.add_argument('--output', '-o', default='sae_metrics.html', help='Output HTML file')
    parser.add_argument('--show-all', action='store_true', help='Show all records by default and enable "All" dropdown')
    args = parser.parse_args()

    records, metrics_by_record, deviations_by_record = load_metrics(args.input)
    if not records:
        raise SystemExit("No records found in the input JSON")

    make_dashboard(records, metrics_by_record, deviations_by_record, args.output, show_all=args.show_all)


if __name__ == '__main__':
    main()
