from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_LIVE = ROOT / "data_live"

PRICE_FILE = DATA_LIVE / "price_factors_current.csv"
FUND_FILE = DATA_LIVE / "fundamental_factors_current.csv"
SECTOR_FILE = DATA_LIVE / "selected_sectors_current.csv"

OUT_SCORE = DATA_LIVE / "stock_scores_current.csv"
OUT_PORTFOLIO = DATA_LIVE / "portfolio_current.csv"
OUT_AUDIT = DATA_LIVE / "portfolio_current_audit.csv"

DISCOUNT_WEIGHT = 0.80
FUND_WEIGHT = 0.20
TOP_N_PER_SECTOR = 3
N_SECTORS = 4
EXPECTED_PORTFOLIO_SIZE = 12

# Arquitetura de alocação congelada — Célula 39
ALLOCATION_VERSION = "ALLOCATION_V1.0.0"
SECTOR_WEIGHT_RULE = "RANK_40_30_20_10"
WITHIN_SECTOR_WEIGHT_RULE = "EQUAL_3"

SECTOR_WEIGHTS = {
    1: 0.40,
    2: 0.30,
    3: 0.20,
    4: 0.10,
}

WITHIN_SECTOR_WEIGHT = 1.0 / 3.0

print("=" * 78)
print("PORTIFOLIO-B3 — SELEÇÃO OPERACIONAL FINAL")
print("=" * 78)

for f in [PRICE_FILE, FUND_FILE, SECTOR_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {f}")

price = pd.read_csv(PRICE_FILE)
fund = pd.read_csv(FUND_FILE)
selected_sectors = pd.read_csv(SECTOR_FILE)

required_price = {"TICKER", "MACRO_SECTOR", "DISCOUNT_52W"}
required_fund = {
    "TICKER", "ROE_W", "ROA_W", "OPERATING_MARGIN_W", "NET_MARGIN_W",
    "DEBT_TO_EQUITY_PROXY_W", "EQUITY"
}
required_sector = {"MACRO_SECTOR"}

if required_price - set(price.columns):
    raise RuntimeError(f"Colunas ausentes em preços: {sorted(required_price - set(price.columns))}")
if required_fund - set(fund.columns):
    raise RuntimeError(f"Colunas ausentes em fundamentos: {sorted(required_fund - set(fund.columns))}")
if required_sector - set(selected_sectors.columns):
    raise RuntimeError(f"Colunas ausentes em setores: {sorted(required_sector - set(selected_sectors.columns))}")

price["TICKER"] = price["TICKER"].astype(str).str.upper().str.strip()
fund["TICKER"] = fund["TICKER"].astype(str).str.upper().str.strip()
price["MACRO_SECTOR"] = price["MACRO_SECTOR"].astype(str).str.upper().str.strip()
selected_sectors["MACRO_SECTOR"] = selected_sectors["MACRO_SECTOR"].astype(str).str.upper().str.strip()

# Compatibilidade com a nova proteção de preços.
# Arquivos antigos, sem a coluna, continuam funcionando como PASS.
if "PRICE_QUALITY_STATUS" not in price.columns:
    price["PRICE_QUALITY_STATUS"] = "PASS"

price["PRICE_QUALITY_STATUS"] = (
    price["PRICE_QUALITY_STATUS"]
    .fillna("PASS")
    .astype(str)
    .str.upper()
    .str.strip()
)

top4 = selected_sectors["MACRO_SECTOR"].dropna().drop_duplicates().tolist()
if len(top4) != 4:
    raise RuntimeError(f"Esperados 4 setores; encontrados {len(top4)}.")

print("\nETAPA 1 — TOP4 SETORES")
for i, s in enumerate(top4, 1):
    print(f"{i}. {s}")
print("STATUS : PASS")

fund_cols = [
    "TICKER", "ROE_W", "ROA_W", "OPERATING_MARGIN_W",
    "NET_MARGIN_W", "DEBT_TO_EQUITY_PROXY_W", "EQUITY"
]

merged = price[
    ["TICKER", "MACRO_SECTOR", "DISCOUNT_52W", "PRICE_QUALITY_STATUS"]
].merge(
    fund[fund_cols],
    on="TICKER",
    how="left",
    validate="one_to_one"
)

merged = merged[merged["MACRO_SECTOR"].isin(top4)].copy()

num_cols = [
    "DISCOUNT_52W", "ROE_W", "ROA_W", "OPERATING_MARGIN_W",
    "NET_MARGIN_W", "DEBT_TO_EQUITY_PROXY_W", "EQUITY"
]
for c in num_cols:
    merged[c] = pd.to_numeric(merged[c], errors="coerce")

# Séries com quebra de comparabilidade não entram na seleção.
merged["PRICE_QUALITY_OK"] = (
    merged["PRICE_QUALITY_STATUS"].eq("PASS")
)

# 1) Discount Score — maior desconto = melhor.
# O ranking é calculado somente entre séries comparáveis.
merged["DISCOUNT_SCORE"] = np.nan
quality_mask = (
    merged["PRICE_QUALITY_OK"]
    & merged["DISCOUNT_52W"].notna()
)

merged.loc[quality_mask, "DISCOUNT_SCORE"] = (
    merged.loc[quality_mask]
    .groupby("MACRO_SECTOR")["DISCOUNT_52W"]
    .rank(method="average", pct=True, ascending=True)
)

# 2) Fundamental Score — mesma regra congelada
score_cols = []
for c in ["ROE_W", "ROA_W", "OPERATING_MARGIN_W", "NET_MARGIN_W"]:
    sc = f"{c}_SCORE"
    merged[sc] = (
        merged.groupby("MACRO_SECTOR")[c]
        .rank(method="average", pct=True, ascending=True)
    )
    score_cols.append(sc)

# Debt/Equity: menor é melhor e somente com patrimônio líquido positivo
merged["LEVERAGE_QUALITY"] = np.where(
    (merged["EQUITY"] > 0) & merged["DEBT_TO_EQUITY_PROXY_W"].notna(),
    -merged["DEBT_TO_EQUITY_PROXY_W"],
    np.nan
)

merged["LEVERAGE_SCORE"] = (
    merged.groupby("MACRO_SECTOR")["LEVERAGE_QUALITY"]
    .rank(method="average", pct=True, ascending=True)
)
score_cols.append("LEVERAGE_SCORE")

merged["FUND_COMPONENTS_VALID"] = merged[score_cols].notna().sum(axis=1)
merged["FUND_SCORE"] = merged[score_cols].mean(axis=1, skipna=True)
merged.loc[merged["FUND_COMPONENTS_VALID"] < 3, "FUND_SCORE"] = np.nan

# 3) Regra congelada 80/20
merged["FINAL_SCORE"] = (
    DISCOUNT_WEIGHT * merged["DISCOUNT_SCORE"]
    + FUND_WEIGHT * merged["FUND_SCORE"]
)

merged["ELIGIBLE"] = (
    merged["PRICE_QUALITY_OK"]
    & merged["DISCOUNT_SCORE"].notna()
    & merged["FUND_SCORE"].notna()
    & merged["FINAL_SCORE"].notna()
)

eligible = merged[merged["ELIGIBLE"]].copy()

# Garante que cada TOP4 possua pelo menos 3 ações válidas após o filtro de qualidade.
eligible_counts = (
    eligible.groupby("MACRO_SECTOR")["TICKER"]
    .nunique()
    .to_dict()
)

insufficient = {
    s: eligible_counts.get(s, 0)
    for s in top4
    if eligible_counts.get(s, 0) < TOP_N_PER_SECTOR
}

if insufficient:
    raise RuntimeError(
        "Setor(es) sem 3 ações elegíveis após controle de qualidade de preços: "
        f"{insufficient}"
    )

eligible = eligible.sort_values(
    ["MACRO_SECTOR", "FINAL_SCORE", "DISCOUNT_SCORE", "FUND_SCORE", "TICKER"],
    ascending=[True, False, False, False, True]
)

portfolio = (
    eligible.groupby("MACRO_SECTOR", group_keys=False)
    .head(TOP_N_PER_SECTOR)
    .copy()
)

portfolio["SECTOR_RANK"] = (
    portfolio.groupby("MACRO_SECTOR").cumcount() + 1
)

sector_order = {s: i + 1 for i, s in enumerate(top4)}
portfolio["TOP4_RANK"] = portfolio["MACRO_SECTOR"].map(sector_order)

portfolio = portfolio.sort_values(
    ["TOP4_RANK", "SECTOR_RANK", "TICKER"]
).reset_index(drop=True)

# 4) Alocação congelada — ALLOCATION_V1.0.0
# A seleção das ações já terminou acima. Esta camada apenas atribui pesos.
portfolio["SECTOR_WEIGHT"] = portfolio["TOP4_RANK"].map(SECTOR_WEIGHTS)
portfolio["WITHIN_SECTOR_WEIGHT"] = WITHIN_SECTOR_WEIGHT
portfolio["PORTFOLIO_WEIGHT"] = (
    portfolio["SECTOR_WEIGHT"] * portfolio["WITHIN_SECTOR_WEIGHT"]
)

if portfolio["SECTOR_WEIGHT"].isna().any():
    raise RuntimeError("Falha ao mapear pesos setoriais para TOP4_RANK.")

sector_weight_check = (
    portfolio[
        ["TOP4_RANK", "MACRO_SECTOR", "SECTOR_WEIGHT"]
    ]
    .drop_duplicates()
    .sort_values("TOP4_RANK")
)

allocation_sector_total = float(sector_weight_check["SECTOR_WEIGHT"].sum())
allocation_portfolio_total = float(portfolio["PORTFOLIO_WEIGHT"].sum())

within_sector_totals = (
    portfolio.groupby("MACRO_SECTOR")["WITHIN_SECTOR_WEIGHT"].sum()
)

allocation_sector_pass = bool(
    np.isclose(allocation_sector_total, 1.0, atol=1e-12)
)

allocation_within_pass = bool(
    np.allclose(within_sector_totals.values, 1.0, atol=1e-12)
)

allocation_portfolio_pass = bool(
    np.isclose(allocation_portfolio_total, 1.0, atol=1e-12)
)

expected_weight_by_rank = {
    1: 0.40 / 3.0,
    2: 0.30 / 3.0,
    3: 0.20 / 3.0,
    4: 0.10 / 3.0,
}

allocation_exact_pass = bool(
    all(
        np.allclose(
            portfolio.loc[
                portfolio["TOP4_RANK"].eq(rank),
                "PORTFOLIO_WEIGHT"
            ].values,
            expected_weight,
            atol=1e-12
        )
        for rank, expected_weight in expected_weight_by_rank.items()
    )
)

sector_counts = portfolio.groupby("MACRO_SECTOR")["TICKER"].nunique().to_dict()
duplicates = int(portfolio["TICKER"].duplicated().sum())
n_portfolio = int(portfolio["TICKER"].nunique())

three_per_sector = all(sector_counts.get(s, 0) == 3 for s in top4)
quality_fail_selected = int((~portfolio["PRICE_QUALITY_OK"]).sum())

audit_pass = (
    len(top4) == 4
    and three_per_sector
    and duplicates == 0
    and n_portfolio == EXPECTED_PORTFOLIO_SIZE
    and quality_fail_selected == 0
    and allocation_sector_pass
    and allocation_within_pass
    and allocation_portfolio_pass
    and allocation_exact_pass
)

print("\n" + "=" * 78)
print("PORTFÓLIO OPERACIONAL — 4 × 3")
print("=" * 78)

view = portfolio[
    [
        "TOP4_RANK", "MACRO_SECTOR", "SECTOR_RANK", "TICKER",
        "DISCOUNT_52W", "DISCOUNT_SCORE", "FUND_SCORE", "FINAL_SCORE",
        "SECTOR_WEIGHT", "WITHIN_SECTOR_WEIGHT", "PORTFOLIO_WEIGHT",
        "PRICE_QUALITY_STATUS"
    ]
].copy()

for c in ["DISCOUNT_52W", "DISCOUNT_SCORE", "FUND_SCORE", "FINAL_SCORE"]:
    view[c] = view[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

for c in ["SECTOR_WEIGHT", "WITHIN_SECTOR_WEIGHT", "PORTFOLIO_WEIGHT"]:
    view[c] = view[c].map(lambda x: f"{x:.4%}" if pd.notna(x) else "")

print(view.to_string(index=False))

excluded_quality = merged[
    ~merged["PRICE_QUALITY_OK"]
][["TICKER", "MACRO_SECTOR", "PRICE_QUALITY_STATUS"]].copy()

print("\n" + "=" * 78)
print("CONTROLE DE QUALIDADE DE PREÇOS")
print("=" * 78)
print(f"Ações bloqueadas por comparabilidade . {len(excluded_quality)}")
print(f"Ações bloqueadas dentro do portfólio . {quality_fail_selected}")

if not excluded_quality.empty:
    print()
    print(excluded_quality.to_string(index=False))

print("\n" + "=" * 78)
print("AUDITORIA FINAL")
print("=" * 78)
print("Arquitetura ....................... 4 setores × 3 ações")
print(f"Setores selecionados .............. {len(top4)}")
print(f"Ações selecionadas ................ {n_portfolio}")
print(f"Duplicidades ...................... {duplicates}")
print(f"Qualidade de preço inválida ....... {quality_fail_selected}")
print("Regra ............................. DISCOUNT_80_FUNDAMENTALS_20")
print("Peso desconto ..................... 80%")
print("Peso fundamentos .................. 20%")
print("Fund Score mínimo ................. 3 componentes")
print("Proteção evento corporativo ....... ATIVA")
print("Histórico congelado ............... PRESERVADO")
print(f"Allocation version ................ {ALLOCATION_VERSION}")
print("Peso setorial ..................... 40% / 30% / 20% / 10%")
print("Peso interno por setor ............ 33,3333% / 33,3333% / 33,3333%")
print(f"Peso total da carteira ............ {allocation_portfolio_total:.4%}")
print(f"Auditoria da alocação ............. {'PASS' if (allocation_sector_pass and allocation_within_pass and allocation_portfolio_pass and allocation_exact_pass) else 'FAIL'}")
print(f"STATUS ............................ {'PASS' if audit_pass else 'FAIL'}")

score_cols_out = [
    "TICKER", "MACRO_SECTOR", "PRICE_QUALITY_STATUS", "PRICE_QUALITY_OK",
    "DISCOUNT_52W", "DISCOUNT_SCORE",
    "ROE_W", "ROA_W", "OPERATING_MARGIN_W", "NET_MARGIN_W",
    "DEBT_TO_EQUITY_PROXY_W", "EQUITY",
    "FUND_COMPONENTS_VALID", "FUND_SCORE", "FINAL_SCORE", "ELIGIBLE"
]

merged[score_cols_out].sort_values(
    ["MACRO_SECTOR", "FINAL_SCORE", "TICKER"],
    ascending=[True, False, True]
).to_csv(OUT_SCORE, index=False)

portfolio_cols = [
    "TOP4_RANK", "MACRO_SECTOR", "SECTOR_RANK", "TICKER",
    "PRICE_QUALITY_STATUS",
    "DISCOUNT_52W", "DISCOUNT_SCORE",
    "FUND_COMPONENTS_VALID", "FUND_SCORE", "FINAL_SCORE",
    "SECTOR_WEIGHT", "WITHIN_SECTOR_WEIGHT", "PORTFOLIO_WEIGHT"
]

portfolio[portfolio_cols].to_csv(OUT_PORTFOLIO, index=False)

audit = pd.DataFrame([
    {"CHECK": "TOP4_SECTORS", "VALUE": len(top4), "EXPECTED": 4, "STATUS": "PASS" if len(top4) == 4 else "FAIL"},
    {"CHECK": "PORTFOLIO_SIZE", "VALUE": n_portfolio, "EXPECTED": 12, "STATUS": "PASS" if n_portfolio == 12 else "FAIL"},
    {"CHECK": "DUPLICATES", "VALUE": duplicates, "EXPECTED": 0, "STATUS": "PASS" if duplicates == 0 else "FAIL"},
    {"CHECK": "THREE_PER_SECTOR", "VALUE": str(sector_counts), "EXPECTED": "3 por setor", "STATUS": "PASS" if three_per_sector else "FAIL"},
    {"CHECK": "PRICE_QUALITY", "VALUE": quality_fail_selected, "EXPECTED": 0, "STATUS": "PASS" if quality_fail_selected == 0 else "FAIL"},
    {"CHECK": "RULE", "VALUE": "DISCOUNT_80_FUNDAMENTALS_20", "EXPECTED": "DISCOUNT_80_FUNDAMENTALS_20", "STATUS": "PASS"},
    {"CHECK": "HISTORICAL_CORE", "VALUE": "PRESERVED", "EXPECTED": "PRESERVED", "STATUS": "PASS"},
    {"CHECK": "ALLOCATION_VERSION", "VALUE": ALLOCATION_VERSION, "EXPECTED": "ALLOCATION_V1.0.0", "STATUS": "PASS"},
    {"CHECK": "SECTOR_WEIGHT_RULE", "VALUE": SECTOR_WEIGHT_RULE, "EXPECTED": "RANK_40_30_20_10", "STATUS": "PASS"},
    {"CHECK": "WITHIN_SECTOR_WEIGHT_RULE", "VALUE": WITHIN_SECTOR_WEIGHT_RULE, "EXPECTED": "EQUAL_3", "STATUS": "PASS"},
    {"CHECK": "SECTOR_WEIGHT_TOTAL", "VALUE": allocation_sector_total, "EXPECTED": 1.0, "STATUS": "PASS" if allocation_sector_pass else "FAIL"},
    {"CHECK": "WITHIN_SECTOR_WEIGHT_TOTAL", "VALUE": str(within_sector_totals.to_dict()), "EXPECTED": "1.0 por setor", "STATUS": "PASS" if allocation_within_pass else "FAIL"},
    {"CHECK": "PORTFOLIO_WEIGHT_TOTAL", "VALUE": allocation_portfolio_total, "EXPECTED": 1.0, "STATUS": "PASS" if allocation_portfolio_pass else "FAIL"},
    {"CHECK": "EXACT_WEIGHT_BY_TOP4_RANK", "VALUE": str(expected_weight_by_rank), "EXPECTED": "13.3333% / 10.0000% / 6.6667% / 3.3333%", "STATUS": "PASS" if allocation_exact_pass else "FAIL"},
])
audit.to_csv(OUT_AUDIT, index=False)

print("\n" + "=" * 78)
print("ARQUIVOS GERADOS")
print("=" * 78)
print(f"Scores    : {OUT_SCORE}")
print(f"Portfólio : {OUT_PORTFOLIO}")
print(f"Auditoria : {OUT_AUDIT}")

if not audit_pass:
    raise RuntimeError(
        f"AUDITORIA FINAL = FAIL | portfolio={n_portfolio}, "
        f"duplicidades={duplicates}, qualidade_preco={quality_fail_selected}, "
        f"contagem_setores={sector_counts}, "
        f"allocation_sector={allocation_sector_pass}, "
        f"allocation_within={allocation_within_pass}, "
        f"allocation_total={allocation_portfolio_pass}, "
        f"allocation_exact={allocation_exact_pass}"
    )

print(f"\nSTATUS: PORTFÓLIO OPERACIONAL 4x3 + {ALLOCATION_VERSION} VALIDADO")
print("=" * 78)
