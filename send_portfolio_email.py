from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
import mimetypes
import os
import smtplib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_LIVE = ROOT / "data_live"
REPORTS = ROOT / "reports"

PDF_FILE = REPORTS / "portfolio_b3_operational_report.pdf"
PORTFOLIO_FILE = DATA_LIVE / "portfolio_current.csv"
TECHNICAL_FILE = DATA_LIVE / "portfolio_technical_current.csv"

EMAIL_USER = os.environ.get("EMAIL_USER", "").strip()
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "").strip()
EMAIL_TO = os.environ.get("EMAIL_TO", "").strip()

if not EMAIL_USER:
    raise RuntimeError("Secret/variável EMAIL_USER ausente.")
if not EMAIL_PASSWORD:
    raise RuntimeError("Secret/variável EMAIL_PASSWORD ausente.")
if not EMAIL_TO:
    raise RuntimeError("Secret/variável EMAIL_TO ausente.")
if not PDF_FILE.exists():
    raise FileNotFoundError(f"PDF não encontrado: {PDF_FILE}")
if not PORTFOLIO_FILE.exists():
    raise FileNotFoundError(f"Portfólio não encontrado: {PORTFOLIO_FILE}")

portfolio = pd.read_csv(PORTFOLIO_FILE)
technical = pd.read_csv(TECHNICAL_FILE) if TECHNICAL_FILE.exists() else pd.DataFrame()

def first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

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

merged = portfolio.copy()

signal_col = None
tech_score_col = None

if not technical.empty and "TICKER" in technical.columns:
    signal_col = first_existing(
        technical,
        ["OPERATIONAL_ACTION", "ACTION", "SIGNAL", "SINAL"]
    )
    tech_score_col = first_existing(
        technical,
        ["TECHNICAL_SCORE", "TECH_SCORE", "SCORE_TECNICO"]
    )
    cols = ["TICKER"]
    for c in [signal_col, tech_score_col]:
        if c and c not in cols:
            cols.append(c)
    merged = merged.merge(
        technical[cols].drop_duplicates("TICKER"),
        on="TICKER",
        how="left"
    )

merged["REPORT_SIGNAL"] = (
    merged[signal_col].map(normalize_signal)
    if signal_col and signal_col in merged.columns
    else "N/D"
)

signal_order = [
    "ENTRADA FORTE",
    "ENTRADA",
    "AGUARDAR",
    "NÃO COMPRAR AGORA",
    "SEM CONFIRMAÇÃO",
    "N/D",
]

counts = merged["REPORT_SIGNAL"].value_counts().to_dict()

def pct(x):
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "N/D"

def num(x):
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "N/D"

# Highest-priority current opportunities: timing first, then FINAL_SCORE.
priority_map = {
    "ENTRADA FORTE": 0,
    "ENTRADA": 1,
    "AGUARDAR": 2,
    "SEM CONFIRMAÇÃO": 3,
    "NÃO COMPRAR AGORA": 4,
    "N/D": 5,
}
merged["_PRIORITY"] = merged["REPORT_SIGNAL"].map(priority_map).fillna(9)

top = merged.sort_values(
    ["_PRIORITY", "FINAL_SCORE"],
    ascending=[True, False]
).head(12)

date_txt = datetime.now().strftime("%d/%m/%Y")

rows_signals = ""
for label in signal_order:
    if label in counts:
        rows_signals += (
            f"<tr><td style='padding:6px 10px;border-bottom:1px solid #ddd;'>"
            f"<b>{label.title()}</b></td>"
            f"<td style='padding:6px 10px;border-bottom:1px solid #ddd;text-align:right;'>"
            f"{counts[label]}</td></tr>"
        )

rows_top = ""
for _, r in top.iterrows():
    rows_top += (
        "<tr>"
        f"<td style='padding:6px;border-bottom:1px solid #ddd;'><b>{r.get('TICKER','')}</b></td>"
        f"<td style='padding:6px;border-bottom:1px solid #ddd;'>{r.get('MACRO_SECTOR','')}</td>"
        f"<td style='padding:6px;border-bottom:1px solid #ddd;'><b>{r.get('REPORT_SIGNAL','N/D')}</b></td>"
        f"<td style='padding:6px;border-bottom:1px solid #ddd;text-align:right;'>{num(r.get('FINAL_SCORE'))}</td>"
        f"<td style='padding:6px;border-bottom:1px solid #ddd;text-align:right;'>{pct(r.get('PORTFOLIO_WEIGHT'))}</td>"
        "</tr>"
    )

html = f"""
<html>
<body style="font-family:Arial,sans-serif;color:#222;">
<h2 style="color:#17365D;">Portfolio B3 Operational</h2>
<p><b>Resultado operacional — {date_txt}</b></p>

<p>O motor concluiu a atualização dos dados, seleção dos quatro setores, escolha das
<b>12 ações</b>, aplicação da alocação <b>40/30/20/10</b> e classificação do timing técnico.
O relatório completo está no <b>PDF anexado</b>.</p>

<h3 style="color:#17365D;">Resumo dos sinais</h3>
<table style="border-collapse:collapse;min-width:360px;">
{rows_signals}
</table>

<h3 style="color:#17365D;">Carteira atual</h3>
<table style="border-collapse:collapse;width:100%;max-width:780px;">
<tr style="background:#D9E2F3;color:#17365D;">
<th style="padding:7px;text-align:left;">Ticker</th>
<th style="padding:7px;text-align:left;">Setor</th>
<th style="padding:7px;text-align:left;">Timing</th>
<th style="padding:7px;text-align:right;">Score final</th>
<th style="padding:7px;text-align:right;">Peso</th>
</tr>
{rows_top}
</table>

<h3 style="color:#17365D;">Como interpretar</h3>
<p><b>Entrada Forte:</b> maior prioridade operacional segundo a camada técnica.</p>
<p><b>Entrada:</b> ponto de entrada liberado.</p>
<p><b>Aguardar:</b> empresa selecionada, porém o timing ainda precisa melhorar.</p>
<p><b>Não comprar agora:</b> empresa permanece estruturalmente selecionada, mas a entrada está bloqueada no momento.</p>
<p><b>Sem confirmação:</b> dados técnicos insuficientes para confirmar o ponto de entrada.</p>

<p>O PDF traz metodologia, pesos, ranking completo, explicação individual e auditorias.</p>

<p style="font-size:12px;color:#666;">
Relatório quantitativo de apoio à decisão. Não constitui garantia de retorno nem recomendação individualizada de investimento.
</p>
</body>
</html>
"""

msg = EmailMessage()
msg["Subject"] = f"Portfolio B3 Operational — {date_txt}"
msg["From"] = EMAIL_USER
msg["To"] = EMAIL_TO
msg.set_content(
    f"Portfolio B3 Operational — {date_txt}\n"
    "O relatório completo está no PDF anexado."
)
msg.add_alternative(html, subtype="html")

ctype, encoding = mimetypes.guess_type(str(PDF_FILE))
maintype, subtype = (ctype or "application/pdf").split("/", 1)

with open(PDF_FILE, "rb") as f:
    msg.add_attachment(
        f.read(),
        maintype=maintype,
        subtype=subtype,
        filename=PDF_FILE.name
    )

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(EMAIL_USER, EMAIL_PASSWORD)
    smtp.send_message(msg)

print("=" * 78)
print("PORTFOLIO B3 OPERATIONAL — ENVIO DE E-MAIL")
print("=" * 78)
print(f"De      : {EMAIL_USER}")
print(f"Para    : {EMAIL_TO}")
print(f"Anexo   : {PDF_FILE.name}")
print("STATUS  : E-MAIL ENVIADO COM SUCESSO")
print("=" * 78)
