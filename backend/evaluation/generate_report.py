#!/usr/bin/env python3
"""
Evaluation Report Generator - Creates human-readable reports from evaluation results.

Aggregates results from multiple evaluation runs and generates
Markdown/HTML reports for review.

Usage:
    python -m evaluation.generate_report \
        --input-dir evaluation/results/ \
        --output evaluation/reports/evaluation_report.md \
        --format markdown
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import jinja2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if not found or invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def load_all_results(input_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all evaluation result files from directory."""
    results = {}
    
    if not input_dir.exists():
        logger.warning(f"Input directory not found: {input_dir}")
        return results
    
    # Find all JSON files
    json_files = list(input_dir.glob("**/*.json"))
    
    for json_file in json_files:
        data = load_json_file(json_file)
        if data:
            # Determine result type from filename
            filename = json_file.stem
            key = filename.lower()
            
            if "ablation" in key:
                key = "ablation"
            elif "benchmark" in key:
                key = "benchmark_" + json_file.parent.name
            elif "perf" in key:
                key = "performance"
            
            results[key] = {
                "data": data,
                "path": str(json_file),
                "filename": json_file.name,
            }
    
    return results


def analyze_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze results and compute summary statistics."""
    analysis = {
        "total_runs": len(results),
        "datasets_evaluated": set(),
        "configs_tested": set(),
        "best_configs": {},
        "trend": {},
    }
    
    # Extract info from different result types
    for key, result_data in results.items():
        data = result_data.get("data", {})
        
        # Main evaluation results
        if "metrics" in data and "dataset" not in data:
            for metric, values in data.get("metrics", {}).items():
                mean = values.get("mean", 0)
                # Track best performing configs
                # (simplified logic - full implementation would compare across configs)
        
        # Benchmark results
        if "dataset" in data:
            analysis["datasets_evaluated"].add(data["dataset"])
        
        # Config information
        if "config" in data:
            config = data["config"]
            config_str = f"{config.get('embedding_model', '?')} + {config.get('reranker_model', '?')}"
            analysis["configs_tested"].add(config_str)
    
    return analysis


def generate_markdown_report(
    results: Dict[str, Dict[str, Any]],
    analysis: Dict[str, Any],
    output_path: Path,
) -> str:
    """Generate a Markdown report."""
    
    report_lines = []
    
    # Header
    report_lines.append("# TenderSync RAG Evaluation Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Total Runs:** {analysis['total_runs']}")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    
    eval_results = results.get("eval_report", {}).get("data", {})
    if eval_results:
        metrics = eval_results.get("metrics", {})
        sample_count = eval_results.get("total_samples", 0)
        
        report_lines.append(f"Evaluation completed on **{sample_count}** samples.")
        report_lines.append("")
        report_lines.append("### Key Metrics")
        report_lines.append("")
        report_lines.append("| Metric | Score | Target | Status |")
        report_lines.append("|--------|-------|--------|--------|")
        
        thresholds = load_json_file(Path(__file__).parent / "config" / "metrics_thresholds.json")
        if thresholds:
            for metric_name, metric_data in metrics.items():
                mean_val = metric_data.get("mean", 0) * 100
                metric_threshold = thresholds.get("thresholds", {}).get(metric_name, {})
                target = metric_threshold.get("target", 0.90) * 100
                min_threshold = metric_threshold.get("min", 0.0) * 100
                
                status = "✅" if mean_val >= min_threshold else "❌"
                if mean_val >= target:
                    status += " (target met)"
                
                report_lines.append(
                    f"| {metric_name.replace('_', ' ').title()} | {mean_val:.1f}% | {target:.0f}% | {status} |"
                )
        else:
            for metric_name, metric_data in metrics.items():
                mean_val = metric_data.get("mean", 0) * 100
                report_lines.append(f"| {metric_name} | {mean_val:.1f}% | - | - |")
        
        report_lines.append("")
    
    # Ablation Study Results
    if "ablation" in results:
        report_lines.append("## Ablation Study")
        report_lines.append("")
        report_lines.append(
            "Comparing different RAG pipeline configurations to measure impact of each component."
        )
        report_lines.append("")
        
        ablation_data = results["ablation"]["data"]
        configs = ablation_data.get("configurations", [])
        comparison = ablation_data.get("comparison", {})
        
        if configs:
            report_lines.append("### Configuration Comparison")
            report_lines.append("")
            report_lines.append("| Configuration | Context Recall | Faithfulness | Answer Relevance | Latency |")
            report_lines.append("|---------------|----------------|--------------|------------------|---------|")
            
            for config in configs:
                name = config.get("name", "unknown")
                metrics = config.get("metrics", {})
                
                cr = metrics.get("context_recall", {}).get("mean", 0) * 100
                fth = metrics.get("faithfulness", {}).get("mean", 0) * 100
                ar = metrics.get("answer_relevance", {}).get("mean", 0) * 100
                lat = config.get("latency_ms", 0)
                
                report_lines.append(f"| {name} | {cr:.1f}% | {fth:.1f}% | {ar:.1f}% | {lat:.0f}ms |")
            
            report_lines.append("")
        
        if comparison.get("configs"):
            report_lines.append("### Improvement vs Baseline")
            report_lines.append("")
            report_lines.append(
                f"*Baseline: {comparison.get('baseline', 'N/A')}*"
            )
            report_lines.append("")
            
            for config_comp in comparison["configs"]:
                name = config_comp.get("name", "unknown")
                report_lines.append(f"**{name}:**")
                
                deltas = config_comp.get("deltas", {})
                for metric_name, delta_data in deltas.items():
                    delta_pct = delta_data.get("delta_pct", 0)
                    improved = "↑" if delta_data.get("improved") else "↓"
                    report_lines.append(
                        f"  - {metric_name.replace('_', ' ').title()}: {improved} {delta_pct:+.1f}% "
                        f"({delta_data.get('actual', 0)*100:.1f}% vs {delta_data.get('baseline', 0)*100:.1f}%)"
                    )
                report_lines.append("")
    
    # Benchmark Results
    benchmark_keys = [k for k in results.keys() if k.startswith("benchmark_")]
    if benchmark_keys:
        report_lines.append("## Standard Benchmark Results")
        report_lines.append("")
        
        for bench_key in benchmark_keys:
            bench_data = results[bench_key]["data"]
            dataset_name = bench_data.get("dataset", bench_key)
            
            report_lines.append(f"### {dataset_name}")
            report_lines.append("")
            
            # Dataset info
            config = bench_data.get("config", {})
            report_lines.append(
                f"- **Samples:** {bench_data.get('num_samples', 'N/A')}"
            )
            report_lines.append(
                f"- **Model:** {config.get('embedding_model', 'N/A')}"
            )
            report_lines.append(
                f"- **Retrieval:** {config.get('retrieval_mode', 'N/A')}"
            )
            report_lines.append("")
            
            metrics = bench_data.get("metrics", {})
            if metrics:
                report_lines.append("| Metric | Mean | Std Dev | Min | Max |")
                report_lines.append("|--------|------|---------|-----|-----|")
                
                for metric_name, metric_data in metrics.items():
                    report_lines.append(
                        f"| {metric_name.replace('_', ' ').title()} | "
                        f"{metric_data.get('mean', 0)*100:.1f}% | "
                        f"{metric_data.get('std', 0)*100:.1f}% | "
                        f"{metric_data.get('min', 0)*100:.1f}% | "
                        f"{metric_data.get('max', 0)*100:.1f}% |"
                    )
                report_lines.append("")
            
            # Individual sample results (summary)
            samples = bench_data.get("samples", [])
            if samples:
                report_lines.append("#### Sample Results (first 10)")
                report_lines.append("")
                report_lines.append("| ID | Query | Metric Scores |")
                report_lines.append("|----|-------|---------------|")
                
                for sample in samples[:10]:
                    query_short = sample.get("query", "")[:50] + "..." if len(sample.get("query", "")) > 50 else sample.get("query", "")
                    scores = " | ".join(
                        f"{k}: {v*100:.0f}%" 
                        for k, v in sample.get("metrics", {}).items()
                    )
                    report_lines.append(f"| {sample.get('id', '')} | {query_short} | {scores} |")
                
                if len(samples) > 10:
                    report_lines.append(f"*... and {len(samples) - 10} more samples*")
                
                report_lines.append("")
    
    # Performance Benchmarks
    if "performance" in results:
        perf_data = results["performance"]["data"]
        report_lines.append("## Performance Benchmarks")
        report_lines.append("")
        report_lines.append("| Metric | Value |")
        report_lines.append("|--------|-------|")
        
        perf_metrics = perf_data.get("metrics", {})
        for metric_name, value in perf_metrics.items():
            if isinstance(value, (int, float)):
                if metric_name.endswith("_ms"):
                    report_lines.append(f"| {metric_name.replace('_', ' ').title()} | {value:.1f}ms |")
                elif metric_name.endswith("_seconds"):
                    report_lines.append(f"| {metric_name.replace('_', ' ').title()} | {value:.3f}s |")
                else:
                    report_lines.append(f"| {metric_name.replace('_', ' ').title()} | {value} |")
            else:
                report_lines.append(f"| {metric_name.replace('_', ' ').title()} | {value} |")
        
        report_lines.append("")
    
    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")
    
    # Generate recommendations based on results
    recommendations = generate_recommendations(results, analysis)
    for rec in recommendations:
        report_lines.append(f"- {rec}")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append(f"*Report generated by TenderSync Evaluation Framework on {datetime.now().isoformat()}*")
    
    report_content = "\n".join(report_lines)
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Report saved to: {output_path}")
    return report_content


def generate_recommendations(
    results: Dict[str, Dict[str, Any]],
    analysis: Dict[str, Any],
) -> List[str]:
    """Generate actionable recommendations based on results."""
    recommendations = []
    
    # Check main evaluation results
    eval_data = results.get("eval_report", {}).get("data", {})
    metrics = eval_data.get("metrics", {})
    
    # Faithfulness recommendation
    faithfulness = metrics.get("faithfulness", {}).get("mean", 0)
    if faithfulness < 0.85:
        recommendations.append(
            "⚠️ **Faithfulness below target (<85%):** Consider adding more context to retrieved chunks "
            "or improving the citation verification step to reduce hallucinations."
        )
    elif faithfulness >= 0.95:
        recommendations.append(
            "✅ **Excellent faithfulness (>95%):** Responses are highly grounded in retrieved context. "
            "Current citation verification approach is working well."
        )
    
    # Context recall recommendation
    context_recall = metrics.get("context_recall", {}).get("mean", 0)
    if context_recall < 0.80:
        recommendations.append(
            "⚠️ **Context recall below target (<80%):** Some relevant information is not being retrieved. "
            "Consider: increasing top-k, improving chunking strategy, or adding BM25 keyword search."
        )
    
    # Check ablation results
    ablation_data = results.get("ablation", {}).get("data", {})
    comparison = ablation_data.get("comparison", {})
    
    if comparison.get("configs"):
        baseline_name = comparison.get("baseline", "")
        best_config = None
        best_improvement = 0
        
        for config in comparison["configs"]:
            deltas = config.get("deltas", {})
            # Simple improvement score (average of deltas)
            avg_delta = sum(d.get("delta", 0) for d in deltas.values()) / max(len(deltas), 1)
            
            if avg_delta > best_improvement:
                best_improvement = avg_delta
                best_config = config
        
        if best_config and best_improvement > 0.05:
            recommendations.append(
                f"📈 **Ablation suggests improvement from:** {best_config.get('name', 'unknown')} "
                f"({best_config.get('description', '')})"
            )
    
    # Default recommendations if none generated
    if not recommendations:
        recommendations.append("✅ All metrics performing within acceptable ranges.")
        recommendations.append("Consider running ablation studies to identify optimization opportunities.")
    
    return recommendations


def generate_html_report(
    results: Dict[str, Dict[str, Any]],
    analysis: Dict[str, Any],
    output_path: Path,
) -> str:
    """Generate an HTML report (simplified, uses Markdown via converter)."""
    # For simplicity, we'll create a basic HTML wrapper around the markdown
    markdown_content = generate_markdown_report(results, analysis, output_path.with_suffix(".md"))
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TenderSync RAG Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #2563eb;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #1e40af;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f3f4f6;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f9fafb;
        }}
        .status-pass {{
            color: #16a34a;
            font-weight: bold;
        }}
        .status-fail {{
            color: #dc2626;
            font-weight: bold;
        }}
        code {{
            background-color: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
        }}
    </style>
</head>
<body>
    {markdown_to_html(markdown_content)}
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to: {output_path}")
    return html_content


def markdown_to_html(markdown: str) -> str:
    """Simple markdown to HTML converter (basic support)."""
    import re
    
    html = markdown
    
    # Headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Tables (simple)
    html = re.sub(
        r'\|(.+)\|',
        lambda m: '<table><tr>' + ''.join(f'<td>{cell.strip()}</td>' for cell in m.group(1).split('|')) + '</tr></table>',
        html
    )
    
    # Lists
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', html, flags=re.DOTALL)
    
    # Paragraphs
    html = re.sub(r'\n\n', r'</p><p>', html)
    html = f'<p>{html}</p>'
    
    # Clean up empty paragraphs
    html = re.sub(r'<p></p>', '', html)
    
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation report from result files"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Directory containing evaluation result JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for report"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["markdown", "html", "both"],
        default="markdown",
        help="Report format"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    logger.info(f"Loading results from: {input_dir}")
    results = load_all_results(input_dir)
    
    logger.info(f"Found {len(results)} result files")
    
    for key in results:
        logger.info(f"  - {key}: {results[key]['filename']}")
    
    # Analyze
    analysis = analyze_results(results)
    
    # Generate report
    if args.format in ["markdown", "both"]:
        md_output = output_path
        if args.format == "both":
            md_output = output_path.with_suffix(".md")
        
        logger.info(f"Generating Markdown report...")
        generate_markdown_report(results, analysis, md_output)
    
    if args.format in ["html", "both"]:
        html_output = output_path
        if args.format == "both":
            html_output = output_path.with_suffix(".html")
        
        logger.info(f"Generating HTML report...")
        generate_html_report(results, analysis, html_output)
    
    logger.info("Report generation complete.")


if __name__ == "__main__":
    main()
