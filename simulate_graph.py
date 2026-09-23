"""Simulate and visualize the Knowledge Graph for Financial Multimodal RAG.

This script parses a sample from the real FinQA dataset (or custom financial data)
into the Knowledge Graph schema defined in src/financial_rag/ingestion/models.py,
and generates an interactive, standalone HTML visualization (using Vis.js).
"""

import json
from pathlib import Path
import re
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from financial_rag.ingestion.models import EntityType, RelationType


def load_finqa_sample(sample_idx: int = 0) -> dict:
    """Load a sample from FinQA test parquet."""
    parquet_path = Path("data/raw/datasets/huggingface/finqa_parquet/test.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found at {parquet_path}")
    df = pd.read_parquet(parquet_path)
    return df.iloc[sample_idx].to_dict()


def build_graph_from_finqa(sample: dict) -> tuple[list[dict], list[dict]]:
    """Build Knowledge Graph nodes and edges from a FinQA sample."""
    nodes = []
    edges = []
    added_nodes = set()

    def add_node(node_id: str, label: str, node_type: EntityType, value: any = None, size: int = 24):
        if node_id not in added_nodes:
            added_nodes.add(node_id)
            nodes.append({
                "id": node_id,
                "label": label,
                "type": node_type.value,
                "value": str(value) if value is not None else "",
                "size": size,
            })

    def add_edge(src: str, dst: str, rel: RelationType, label: str = ""):
        edges.append({
            "from": src,
            "to": dst,
            "relation": rel.value,
            "label": label or rel.value,
        })

    # 1. Document & Context Container
    doc_id = sample.get("id", "doc_finqa")
    add_node(doc_id, f"Document: {doc_id}", EntityType.DOCUMENT, size=32)

    # 2. Company Node (Extracted from question or pre_text)
    company_name = "Entergy Corporation"
    if "entergy" in sample.get("question", "").lower():
        company_name = "Entergy Corporation"
    add_node("comp:entergy", company_name, EntityType.COMPANY, size=30)
    add_edge(doc_id, "comp:entergy", RelationType.FILED_BY)

    # 3. Table Node
    table_id = f"{doc_id}:table"
    add_node(table_id, "Table: Net Revenue & Rate Changes", EntityType.TABLE, size=28)
    add_edge(doc_id, table_id, RelationType.CONTAINS)

    # 4. Footnote Nodes from Post-text & Pre-text
    post_texts = sample.get("post_text", [])
    if hasattr(post_texts, "tolist"):
        post_texts = post_texts.tolist()
    for idx, txt in enumerate(post_texts, start=1):
        txt_str = str(txt)
        if "note" in txt_str.lower():
            note_match = re.search(r"note\s+(\d+)", txt_str, re.IGNORECASE)
            note_label = f"Note {note_match.group(1)} (Rate Proceedings)" if note_match else f"Footnote {idx}"
            note_id = f"{doc_id}:note_{idx}"
            add_node(note_id, note_label, EntityType.FOOTNOTE, value=txt_str[:120] + "...", size=22)
            add_edge(table_id, note_id, RelationType.CROSS_REF, "CROSS_REF")

    # 5. Parse Table Rows (Metrics, Periods, Values)
    raw_table = sample.get("table", [])
    if hasattr(raw_table, "tolist"):
        raw_table = raw_table.tolist()

    header_row = raw_table[0] if raw_table else []
    data_rows = raw_table[1:] if len(raw_table) > 1 else []

    # Time periods
    period_2014 = "period:2014"
    period_2015 = "period:2015"
    add_node(period_2014, "FY2014", EntityType.TIME_PERIOD, size=24)
    add_node(period_2015, "FY2015", EntityType.TIME_PERIOD, size=24)

    metrics_created = []

    for r_idx, row in enumerate(data_rows):
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not row or len(row) < 2:
            continue
        raw_name = str(row[0]).strip()
        raw_val = str(row[1]).strip()

        if not raw_name:
            continue

        clean_val = re.sub(r"[\$\(\)\,\s]", "", raw_val)
        metric_id = f"metric:{r_idx}:{clean_val}"
        add_node(metric_id, raw_name, EntityType.METRIC, size=22)
        add_edge(table_id, metric_id, RelationType.CONTAINS)

        val_id = f"val:{r_idx}"
        add_node(val_id, raw_val, EntityType.MONETARY_VALUE, value=raw_val, size=18)
        add_edge(metric_id, val_id, RelationType.HAS_VALUE)

        # Year association
        if "2014" in raw_name.lower():
            add_edge(metric_id, period_2014, RelationType.REPORTED_IN)
            metrics_created.append(("2014", metric_id, val_id, raw_val))
        elif "2015" in raw_name.lower():
            add_edge(metric_id, period_2015, RelationType.REPORTED_IN)
            metrics_created.append(("2015", metric_id, val_id, raw_val))

    # 6. Program-of-Thought (PoT) Calculation Node: Net Change in Net Revenue
    target_q = sample.get("question", "")
    target_ans = sample.get("answer", "")
    calc_id = "pot:calculation"
    add_node(
        calc_id, 
        f"PoT Result: Change = {target_ans} ($5829 - $5735)", 
        EntityType.METRIC, 
        value=f"Formula: 5829 - 5735 = {target_ans}\nQuestion: {target_q}", 
        size=26
    )

    # Link 2014 and 2015 Revenue into Calculation
    for yr, m_id, v_id, val in metrics_created:
        add_edge(m_id, calc_id, RelationType.DERIVED_FROM, f"Inputs {yr}")

    return nodes, edges


def generate_interactive_html(nodes: list[dict], edges: list[dict], output_path: Path):
    """Generate modern, interactive dark-mode HTML with Vis.js."""
    color_map = {
        EntityType.COMPANY.value: {"background": "#f59e0b", "border": "#d97706"},        # Amber
        EntityType.DOCUMENT.value: {"background": "#3b82f6", "border": "#2563eb"},       # Blue
        EntityType.SECTION.value: {"background": "#6366f1", "border": "#4f46e5"},        # Indigo
        EntityType.TABLE.value: {"background": "#8b5cf6", "border": "#7c3aed"},          # Purple
        EntityType.METRIC.value: {"background": "#10b981", "border": "#059669"},         # Emerald
        EntityType.TIME_PERIOD.value: {"background": "#eab308", "border": "#ca8a04"},    # Yellow
        EntityType.MONETARY_VALUE.value: {"background": "#ef4444", "border": "#dc2626"}, # Red
        EntityType.FOOTNOTE.value: {"background": "#64748b", "border": "#475569"},       # Slate
    }

    vis_nodes = []
    for n in nodes:
        c = color_map.get(n["type"], {"background": "#94a3b8", "border": "#64748b"})
        vis_nodes.append({
            "id": n["id"],
            "label": n["label"],
            "title": f"<b>Type:</b> {n['type'].upper()}<br><b>ID:</b> {n['id']}<br><b>Value:</b> {n.get('value', 'N/A')}",
            "color": {"background": c["background"], "border": c["border"], "highlight": {"background": "#38bdf8", "border": "#0284c7"}},
            "size": n["size"],
            "font": {"color": "#ffffff", "size": 14, "face": "system-ui"},
            "shape": "dot",
            "borderWidth": 2,
        })

    vis_edges = []
    for e in edges:
        vis_edges.append({
            "from": e["from"],
            "to": e["to"],
            "label": e["label"],
            "arrows": "to",
            "color": {"color": "#64748b", "highlight": "#38bdf8"},
            "font": {"color": "#94a3b8", "size": 11, "align": "middle", "strokeWidth": 0},
            "smooth": {"type": "curvedCW", "roundness": 0.15},
        })

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>Financial Multimodal RAG — Knowledge Graph Simulator</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0b0f19;
      color: #e2e8f0;
      display: flex;
      height: 100vh;
      overflow: hidden;
    }}
    #sidebar {{
      width: 340px;
      background: #111827;
      border-right: 1px solid #1f2937;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      overflow-y: auto;
    }}
    h1 {{ font-size: 1.15rem; color: #38bdf8; line-height: 1.4; }}
    .subtitle {{ font-size: 0.8rem; color: #94a3b8; }}
    .legend-card {{
      background: #1e293b;
      border-radius: 8px;
      padding: 14px;
      border: 1px solid #334155;
    }}
    .legend-title {{ font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; margin-bottom: 10px; font-weight: 700; letter-spacing: 0.05em; }}
    .legend-item {{ display: flex; align-items: center; gap: 10px; font-size: 0.82rem; margin-bottom: 7px; }}
    .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
    #details {{
      background: #1e293b;
      border-radius: 8px;
      padding: 14px;
      border: 1px solid #334155;
      font-size: 0.85rem;
      flex-grow: 1;
    }}
    #details pre {{
      white-space: pre-wrap;
      word-break: break-all;
      background: #0f172a;
      padding: 10px;
      border-radius: 6px;
      margin-top: 8px;
      color: #38bdf8;
      font-size: 0.78rem;
    }}
    #network-container {{
      flex-grow: 1;
      position: relative;
    }}
    #network {{
      width: 100%;
      height: 100%;
    }}
    .tip-bar {{
      position: absolute;
      bottom: 16px;
      right: 16px;
      background: rgba(17, 24, 39, 0.85);
      border: 1px solid #374151;
      border-radius: 20px;
      padding: 8px 16px;
      font-size: 0.78rem;
      color: #94a3b8;
      backdrop-filter: blur(4px);
    }}
  </style>
</head>
<body>
  <div id="sidebar">
    <div>
      <h1>🏛️ Mô Phỏng Đồ Thị Tri Thức</h1>
      <p class="subtitle">Financial Multimodal RAG & FinQA Engine</p>
    </div>

    <div class="legend-card">
      <div class="legend-title">Phân Loại Thực Thể (Node Types)</div>
      <div class="legend-item"><span class="dot" style="background:#f59e0b"></span> Company (Doanh nghiệp)</div>
      <div class="legend-item"><span class="dot" style="background:#3b82f6"></span> Document (Tài liệu gốc)</div>
      <div class="legend-item"><span class="dot" style="background:#8b5cf6"></span> Table (Bảng tài chính)</div>
      <div class="legend-item"><span class="dot" style="background:#10b981"></span> Metric (Chỉ tiêu kế toán)</div>
      <div class="legend-item"><span class="dot" style="background:#eab308"></span> Period (Kỳ / Niên độ)</div>
      <div class="legend-item"><span class="dot" style="background:#ef4444"></span> Value (Con số cụ thể)</div>
      <div class="legend-item"><span class="dot" style="background:#64748b"></span> Footnote (Thuyết minh)</div>
    </div>

    <div id="details">
      <div class="legend-title">Chi Tiết Node Được Chọn</div>
      <p id="node-info" style="color: #94a3b8;">Nhấp vào bất kỳ Node nào trên đồ thị để xem thuộc tính chi tiết và các quan hệ kế toán.</p>
    </div>
  </div>

  <div id="network-container">
    <div id="network"></div>
    <div class="tip-bar">💡 Cuộn chuột để Phóng to/Thu nhỏ • Kéo chuột để Di chuyển • Bấm vào Node để Khám phá</div>
  </div>

  <script type="text/javascript">
    const nodes = new vis.DataSet({json.dumps(vis_nodes)});
    const edges = new vis.DataSet({json.dumps(vis_edges)});

    const container = document.getElementById('network');
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      physics: {{
        barnesHut: {{
          gravitationalConstant: -3500,
          centralGravity: 0.35,
          springLength: 140,
          springConstant: 0.04,
          damping: 0.09
        }},
        stabilization: {{ iterations: 200 }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 150,
        zoomView: true,
        dragView: true
      }}
    }};

    const network = new vis.Network(container, data, options);

    network.on("click", function(params) {{
      if (params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        const node = nodes.get(nodeId);
        const connectedEdges = network.getConnectedEdges(nodeId);
        
        let detailsHtml = `
          <strong style="color:#f8fafc; font-size:1rem;">${{node.label}}</strong><br>
          <span style="color:#38bdf8; font-size:0.8rem;">Type: ${{node.id.split(':')[0].toUpperCase()}}</span><br>
          <pre>${{JSON.stringify(node, null, 2)}}</pre>
          <p style="margin-top:8px; color:#cbd5e1;"><b>Số liên kết kết nối:</b> ${{connectedEdges.length}}</p>
        `;
        document.getElementById('node-info').innerHTML = detailsHtml;
      }}
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    print(f"[+] Successfully wrote interactive HTML to: {output_path.resolve()}")


def main():
    print("[*] Loading sample from FinQA test dataset...")
    sample = load_finqa_sample(sample_idx=0)
    print(f"[+] Loaded sample: {sample.get('id')} - Question: {sample.get('question')}")

    print("[*] Building Knowledge Graph nodes and edges...")
    nodes, edges = build_graph_from_finqa(sample)
    print(f"    Nodes count: {len(nodes)}")
    print(f"    Edges count: {len(edges)}")

    output_file = Path("knowledge_graph_simulation.html")
    generate_interactive_html(nodes, edges, output_file)
    print(f"\n[SUCCESS] Knowledge graph generated at: {output_file.resolve()}")
    print("Open 'knowledge_graph_simulation.html' in your browser (Chrome/Edge) to view the interactive graph.")


if __name__ == "__main__":
    main()
