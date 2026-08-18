import os
import tempfile
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

def generate_radar_chart(factor_scores: dict) -> bytes:
    """Generates a radar chart PNG image using Plotly."""
    categories = list(factor_scores.keys())
    values = list(factor_scores.values())
    
    # Close the loop
    if categories:
        categories.append(categories[0])
        values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Tender Evaluation'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=False
    )
    
    return fig.to_image(format="png", width=600, height=400)

def render_evaluation_report(analysis_data: dict, company_profile: dict) -> bytes:
    """Renders HTML template with Jinja2 and converts to PDF via WeasyPrint."""
    evaluation_data = analysis_data.get("evaluation_data", {})
    extracted_data = analysis_data.get("extracted_data", {})
    
    factor_scores = evaluation_data.get("factor_scores", {})
    png_bytes = generate_radar_chart(factor_scores)
    
    import base64
    b64_image = base64.b64encode(png_bytes).decode('utf-8')
    image_data_uri = f"data:image/png;base64,{b64_image}"
    
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("evaluation_report.html")
    
    hard_gates_list = []
    for gate in evaluation_data.get("hard_gates", []):
        hard_gates_list.append((gate.get("gate_name", "Unknown"), gate.get("passed", False)))
    
    html_content = template.render(
        filename=analysis_data.get("filename", "Tender"),
        created_at=analysis_data.get("created_at"),
        decision=evaluation_data.get("decision"),
        win_probability_score=evaluation_data.get("win_probability_score", 0),
        rationale=evaluation_data.get("rationale", ""),
        image_data_uri=image_data_uri,
        factor_scores=factor_scores,
        hard_gates=hard_gates_list
    )
    
    return HTML(string=html_content).write_pdf()
