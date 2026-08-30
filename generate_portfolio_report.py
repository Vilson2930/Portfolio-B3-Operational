from pathlib import Path
from datetime import datetime
import math
import pandas as pd
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)

ROOT = Path(__file__).resolve().parent
DATA_LIVE = ROOT / "data_live"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

PORTFOLIO_FILE = DATA_LIVE / "portfolio_current.csv"
TECHNICAL_FILE = DATA_LIVE / "portfolio_technical_current.csv"
PORTFOLIO_AUDIT_FILE = DATA_LIVE / "portfolio_current_audit.csv"
EXTREME_AUDIT_FILE = DATA_LIVE / "portfolio_extreme_audit.csv"
SECTORS_FILE = DATA_LIVE / "selected_sectors_current.csv"

OUT_PDF = REPORTS / "portfolio_b3_operational_report.pdf"

REQUIRED = [PORTFOLIO_FILE]
for p in REQUIRED:
    if not p.exists():
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {p}")

portfolio = pd.read_csv(PORTFOLIO_FILE)
technical = pd.read_csv(TECHNICAL_FILE) if TECHNICAL_FILE.exists() else pd.DataFrame()
portfolio_audit = pd.read_csv(PORTFOLIO_AUDIT_FILE) if PORTFOLIO_AUDIT_FILE.exists() else pd.DataFrame()
extreme_audit = pd.read_csv(EXTREME_AUDIT_FILE) if EXTREME_AUDIT_FILE.exists() else pd.DataFrame()
sectors = pd.read_csv(SECTORS_FILE) if SECTORS_FILE.exists() else pd.DataFrame()

def first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def fmt_pct(x, digits=1):
    try:
        if pd.isna(x):
            return "N/D"
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "N/D"

def fmt_num(x, digits=3):
    try:
        if pd.isna(x):
            return "N/D"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/D"

def normalize_signal(v):
    s = str(v or "").upper().strip()
    replacements = {
        "ENTRADA LIBERADA": "ENTRADA",
        "COMPRA FORTE": "ENTRADA FORTE",
        "COMPRA": "ENTRADA",
        "AGUARDAR MELHOR PONTO": "AGUARDAR",
        "NÃO ENTRAR AGORA": "NÃO COMPRAR AGORA",
        "NAO ENTRAR AGORA": "NÃO COMPRAR AGORA",
        "BLOQUEAR ENTRADA TEMPORARIAMENTE": "NÃO COMPRAR AGORA",
        "SEM CONFIRMAÇÃO TÉCNICA": "SEM CONFIRMAÇÃO",
        "SEM CONFIRMACAO TECNICA": "SEM CONFIRMAÇÃO",
    }
    return replacements.get(s, s if s else "N/D")

# Merge technical layer without changing any score or selection.
merged = portfolio.copy()

if not technical.empty and "TICKER" in technical.columns:
    tech_cols = ["TICKER"]
    for c in [
        "TECHNICAL_SCORE", "TECH_SCORE", "SCORE_TECNICO",
        "SIGNAL", "SINAL", "OPERATIONAL_ACTION", "ACTION",
        "RSI14", "RSI_14", "MACD", "ATR_PCT",
        "MM20", "MM50", "MM200", "RET_20D", "RET_60D",
        "REL_VOLUME", "RELATIVE_VOLUME"
    ]:
        if c in technical.columns and c not in tech_cols:
            tech_cols.append(c)
    merged = merged.merge(
        technical[tech_cols].drop_duplicates("TICKER"),
        on="TICKER",
        how="left"
    )

signal_col = first_existing(
    merged,
    ["OPERATIONAL_ACTION", "ACTION", "SIGNAL", "SINAL"]
)
tech_score_col = first_existing(
    merged,
    ["TECHNICAL_SCORE", "TECH_SCORE", "SCORE_TECNICO"]
)

merged["REPORT_SIGNAL"] = (
    merged[signal_col].map(normalize_signal)
    if signal_col else "N/D"
)

# Styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleB3",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#17365D"),
    spaceAfter=5*mm
))
styles.add(ParagraphStyle(
    name="SubTitleB3",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    leading=13,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#4F4F4F"),
    spaceAfter=6*mm
))
styles.add(ParagraphStyle(
    name="H1B3",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=14,
    leading=17,
    textColor=colors.HexColor("#17365D"),
    spaceBefore=4*mm,
    spaceAfter=3*mm
))
styles.add(ParagraphStyle(
    name="H2B3",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    textColor=colors.HexColor("#17365D"),
    spaceBefore=3*mm,
    spaceAfter=2*mm
))
styles.add(ParagraphStyle(
    name="BodyB3",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=8.6,
    leading=11.5,
    textColor=colors.HexColor("#222222"),
    spaceAfter=2*mm
))
styles.add(ParagraphStyle(
    name="SmallB3",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7.2,
    leading=9.2,
    textColor=colors.HexColor("#333333")
))

PAGE_W, PAGE_H = A4

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2F3"))
    canvas.line(15*mm, 12*mm, PAGE_W-15*mm, 12*mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(15*mm, 7.5*mm, "PORTFOLIO B3 OPERATIONAL | Relatório quantitativo")
    canvas.drawRightString(PAGE_W-15*mm, 7.5*mm, f"Página {doc.page}")
    canvas.restoreState()

def table(data, widths=None, header=True, font_size=7.2):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold" if header else "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), font_size),
        ("LEADING", (0,0), (-1,-1), font_size + 2),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#A6A6A6")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D9E2F3")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#17365D")),
        ]
        for r in range(1, len(data)):
            if r % 2 == 0:
                commands.append(("BACKGROUND", (0,r), (-1,r), colors.HexColor("#F7F9FC")))
    t.setStyle(TableStyle(commands))
    return t

story = []
now = datetime.now()

story.append(Paragraph("PORTFOLIO B3 OPERATIONAL", styles["TitleB3"]))
story.append(Paragraph(
    "Relatório operacional de seleção, alocação e timing de entrada",
    styles["SubTitleB3"]
))
story.append(Paragraph(
    f"Gerado em {now.strftime('%d/%m/%Y %H:%M')}",
    styles["SubTitleB3"]
))

# Executive dashboard
story.append(Paragraph("Resumo executivo", styles["H1B3"]))

n_stocks = int(merged["TICKER"].nunique()) if "TICKER" in merged else len(merged)
n_sectors = int(merged["MACRO_SECTOR"].nunique()) if "MACRO_SECTOR" in merged else 0
allocation_total = pd.to_numeric(
    merged.get("PORTFOLIO_WEIGHT", pd.Series(dtype=float)),
    errors="coerce"
).sum()

signals = merged["REPORT_SIGNAL"].value_counts(dropna=False).to_dict()
dashboard = [
    ["Indicador", "Resultado"],
    ["Ações selecionadas", str(n_stocks)],
    ["Setores selecionados", str(n_sectors)],
    ["Arquitetura", "4 setores × 3 ações"],
    ["Regra setorial", "TOP4_1Y"],
    ["Regra das ações", "80% desconto + 20% fundamentos"],
    ["Alocação", "40% / 30% / 20% / 10%"],
    ["Peso total", fmt_pct(allocation_total, 2)],
]
for label in ["ENTRADA FORTE", "ENTRADA", "AGUARDAR", "NÃO COMPRAR AGORA", "SEM CONFIRMAÇÃO"]:
    if label in signals:
        dashboard.append([label.title(), str(signals[label])])

story.append(table(dashboard, widths=[70*mm, 70*mm], font_size=8))
story.append(Spacer(1, 3*mm))

story.append(Paragraph("Arquitetura do modelo", styles["H1B3"]))
story.append(Paragraph(
    "<b>Seleção de setores:</b> TOP4_1Y. A cada ciclo operacional, os quatro setores são definidos pelo ranking setorial validado, sem hardcode de nomes.",
    styles["BodyB3"]
))
story.append(Paragraph(
    "<b>Seleção de ações:</b> dentro de cada setor selecionado, o motor combina 80% de DISCOUNT_52W com 20% de fundamentos e seleciona as três maiores pontuações elegíveis.",
    styles["BodyB3"]
))
story.append(Paragraph(
    "<b>Alocação de capital:</b> setor rank 1 = 40%, rank 2 = 30%, rank 3 = 20% e rank 4 = 10%. O peso de cada setor é dividido igualmente entre suas três ações.",
    styles["BodyB3"]
))
story.append(Paragraph(
    "<b>Timing:</b> a camada técnica é complementar e não altera a composição nem o FINAL_SCORE. Ela serve apenas para classificar o momento de entrada.",
    styles["BodyB3"]
))

# Sector allocation
story.append(Paragraph("Setores e alocação atual", styles["H1B3"]))
sector_cols = ["TOP4_RANK", "MACRO_SECTOR", "SECTOR_WEIGHT"]
sector_view = (
    merged[sector_cols]
    .drop_duplicates()
    .sort_values("TOP4_RANK")
    if all(c in merged.columns for c in sector_cols)
    else pd.DataFrame()
)

if not sector_view.empty:
    sector_data = [["Rank", "Setor", "Peso do setor", "Peso por ação"]]
    for _, r in sector_view.iterrows():
        sw = float(r["SECTOR_WEIGHT"])
        sector_data.append([
            int(r["TOP4_RANK"]),
            str(r["MACRO_SECTOR"]),
            fmt_pct(sw, 1),
            fmt_pct(sw/3, 2),
        ])
    story.append(table(sector_data, widths=[18*mm, 65*mm, 35*mm, 35*mm], font_size=8))

# Main ranking
story.append(Paragraph("Ranking operacional das 12 ações", styles["H1B3"]))
rank_data = [[
    "Setor", "Rank", "Ticker", "Desconto",
    "Fund.", "Score", "Peso", "Timing"
]]
for _, r in merged.sort_values(["TOP4_RANK", "SECTOR_RANK"]).iterrows():
    rank_data.append([
        str(r.get("MACRO_SECTOR", "")),
        f'{int(r.get("SECTOR_RANK", 0))}',
        str(r.get("TICKER", "")),
        fmt_pct(r.get("DISCOUNT_52W", np.nan), 1),
        fmt_num(r.get("FUND_SCORE", np.nan), 3),
        fmt_num(r.get("FINAL_SCORE", np.nan), 3),
        fmt_pct(r.get("PORTFOLIO_WEIGHT", np.nan), 2),
        str(r.get("REPORT_SIGNAL", "N/D")),
    ])
story.append(table(
    rank_data,
    widths=[35*mm, 12*mm, 19*mm, 23*mm, 21*mm, 20*mm, 22*mm, 30*mm],
    font_size=6.3
))

story.append(Paragraph(
    "Leitura: o ranking estrutural define quais ações fazem parte da carteira. O timing técnico define apenas se o ponto atual está liberado, deve aguardar ou deve permanecer bloqueado temporariamente.",
    styles["SmallB3"]
))

# Sector pages
for sector_rank, g in merged.groupby("TOP4_RANK", sort=True):
    g = g.sort_values("SECTOR_RANK")
    sector_name = str(g["MACRO_SECTOR"].iloc[0])
    sw = float(g["SECTOR_WEIGHT"].iloc[0]) if "SECTOR_WEIGHT" in g else np.nan

    story.append(Paragraph(sector_name, styles["H1B3"]))
    story.append(Paragraph(
        f"Rank setorial: <b>{int(sector_rank)}</b>. Peso do setor: <b>{fmt_pct(sw,1)}</b>. "
        f"Peso por ação: <b>{fmt_pct(sw/3,2)}</b>.",
        styles["BodyB3"]
    ))

    sector_data = [["Prior.", "Ticker", "Desconto", "Fund.", "Score", "Peso", "Timing"]]
    for _, r in g.iterrows():
        sector_data.append([
            int(r.get("SECTOR_RANK", 0)),
            str(r.get("TICKER", "")),
            fmt_pct(r.get("DISCOUNT_52W", np.nan), 1),
            fmt_num(r.get("FUND_SCORE", np.nan), 3),
            fmt_num(r.get("FINAL_SCORE", np.nan), 3),
            fmt_pct(r.get("PORTFOLIO_WEIGHT", np.nan), 2),
            str(r.get("REPORT_SIGNAL", "N/D")),
        ])
    story.append(table(
        sector_data,
        widths=[18*mm, 24*mm, 27*mm, 23*mm, 23*mm, 25*mm, 38*mm],
        font_size=7
    ))

# Individual explanations
story.append(PageBreak())
story.append(Paragraph("Explicação individual das 12 ações", styles["H1B3"]))
story.append(Paragraph(
    "Esta seção apenas traduz os resultados já produzidos pelo motor. Não cria novos critérios, não recalcula o ranking e não altera a carteira.",
    styles["BodyB3"]
))

signal_explanations = {
    "ENTRADA FORTE": "O timing técnico indica condição excepcional dentro das regras do motor e recebe a maior prioridade operacional.",
    "ENTRADA": "O timing técnico está favorável e a entrada está liberada segundo a camada complementar.",
    "AGUARDAR": "A ação permanece selecionada estruturalmente, mas o timing técnico ainda não apresenta condição suficiente para nova entrada.",
    "NÃO COMPRAR AGORA": "A ação permanece estruturalmente selecionada, porém o timing técnico não recomenda nova compra neste momento.",
    "SEM CONFIRMAÇÃO": "A ação permanece selecionada, mas não há dados técnicos suficientes para confirmação do ponto de entrada.",
}

for _, r in merged.sort_values(["TOP4_RANK", "SECTOR_RANK"]).iterrows():
    ticker = str(r.get("TICKER", ""))
    sector = str(r.get("MACRO_SECTOR", ""))
    signal = str(r.get("REPORT_SIGNAL", "N/D"))
    explanation = signal_explanations.get(
        signal,
        "O sinal técnico deve ser interpretado como uma camada complementar ao ranking estrutural."
    )
    tech_txt = (
        f" Score técnico: {fmt_num(r.get(tech_score_col, np.nan), 1)}."
        if tech_score_col else ""
    )

    block = [
        Paragraph(f"<b>{ticker} - {sector}</b>", styles["H2B3"]),
        Paragraph(
            f"<b>Timing:</b> {signal}. {explanation}",
            styles["BodyB3"]
        ),
        Paragraph(
            f"Rank no setor: {int(r.get('SECTOR_RANK', 0))}. "
            f"Desconto 52 semanas: {fmt_pct(r.get('DISCOUNT_52W', np.nan),1)}. "
            f"Fund Score: {fmt_num(r.get('FUND_SCORE', np.nan),3)}. "
            f"FINAL_SCORE: {fmt_num(r.get('FINAL_SCORE', np.nan),3)}. "
            f"Peso na carteira: {fmt_pct(r.get('PORTFOLIO_WEIGHT', np.nan),2)}."
            f"{tech_txt}",
            styles["BodyB3"]
        ),
    ]
    story.append(KeepTogether(block))

# Audit
story.append(PageBreak())
story.append(Paragraph("Auditoria e qualidade", styles["H1B3"]))

if not portfolio_audit.empty:
    audit_data = [["Check", "Valor", "Esperado", "Status"]]
    for _, r in portfolio_audit.iterrows():
        audit_data.append([
            str(r.get("CHECK", "")),
            str(r.get("VALUE", "")),
            str(r.get("EXPECTED", "")),
            str(r.get("STATUS", "")),
        ])
    story.append(Paragraph("Auditoria interna do portfólio", styles["H2B3"]))
    story.append(table(audit_data, widths=[52*mm, 48*mm, 48*mm, 25*mm], font_size=6.5))

if not extreme_audit.empty:
    status_col = first_existing(extreme_audit, ["EXTERNAL_STATUS", "STATUS"])
    diag_col = first_existing(extreme_audit, ["EXTERNAL_DIAGNOSTIC", "DIAGNOSTIC"])
    if status_col:
        reviews = extreme_audit[extreme_audit[status_col].astype(str).str.upper().eq("REVIEW")].copy()
        if not reviews.empty:
            story.append(Paragraph("Reviews externos", styles["H2B3"]))
            rev_data = [["Ticker", "Setor", "Diagnóstico"]]
            for _, r in reviews.iterrows():
                rev_data.append([
                    str(r.get("TICKER", "")),
                    str(r.get("MACRO_SECTOR", "")),
                    str(r.get(diag_col, "REVIEW")) if diag_col else "REVIEW",
                ])
            story.append(table(rev_data, widths=[30*mm, 55*mm, 85*mm], font_size=6.8))
            story.append(Paragraph(
                "Os reviews externos são diagnósticos independentes e não alteram automaticamente a metodologia, a composição ou os pesos da carteira.",
                styles["SmallB3"]
            ))

story.append(Paragraph("Observação metodológica", styles["H1B3"]))
story.append(Paragraph(
    "O relatório separa três decisões: seleção de setores, seleção de ações e momento de entrada. "
    "Uma ação pode permanecer entre as 12 selecionadas e, ao mesmo tempo, receber AGUARDAR, NÃO COMPRAR AGORA ou SEM CONFIRMAÇÃO. "
    "A camada técnica não remove automaticamente a ação da carteira.",
    styles["BodyB3"]
))
story.append(Paragraph(
    "A arquitetura de alocação utilizada é ALLOCATION_V1.0.0: 40% / 30% / 20% / 10% por rank setorial, com divisão igual entre as três ações de cada setor. "
    "O histórico validado permanece preservado e a camada operacional é atualizada com os dados correntes.",
    styles["BodyB3"]
))
story.append(Paragraph(
    "<b>Aviso:</b> relatório quantitativo de apoio à decisão. Não constitui garantia de retorno nem recomendação individualizada de investimento.",
    styles["SmallB3"]
))

doc = SimpleDocTemplate(
    str(OUT_PDF),
    pagesize=A4,
    rightMargin=12*mm,
    leftMargin=12*mm,
    topMargin=14*mm,
    bottomMargin=17*mm,
    title="Portfolio B3 Operational",
    author="Portfolio B3 Operational"
)

doc.build(story, onFirstPage=footer, onLaterPages=footer)

print("=" * 78)
print("PORTFOLIO B3 OPERATIONAL — RELATÓRIO PDF")
print("=" * 78)
print(f"Arquivo : {OUT_PDF}")
print(f"Ações   : {n_stocks}")
print(f"Setores : {n_sectors}")
print(f"Peso    : {allocation_total:.4%}")
print("STATUS  : PDF GERADO COM SUCESSO")
print("=" * 78)
