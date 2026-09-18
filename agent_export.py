# ============================================================
# agent_export.py
# PORTFOLIO-B3-OPERATIONAL
#
# EXPORTADOR PARA O INVESTMENT CIO AGENT
#
# IMPORTANTE:
# - NÃO recalcula a seleção do Portfolio-B3
# - NÃO altera TOP4_1Y
# - NÃO altera DISCOUNT_80_FUNDAMENTALS_20
# - NÃO altera sinais técnicos
# - NÃO altera pesos
# - NÃO substitui nenhum arquivo operacional
#
# Apenas lê os outputs oficiais já produzidos pelo robô
# e cria uma representação estruturada para o CIO.
# ============================================================

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_LIVE = ROOT / "data_live"
OUTPUT_DIR = ROOT / "outputs"

PORTFOLIO_FILE = DATA_LIVE / "portfolio_current.csv"

TECHNICAL_FILE = (
    DATA_LIVE
    / "portfolio_technical_current.csv"
)

TECHNICAL_AUDIT_FILE = (
    DATA_LIVE
    / "portfolio_technical_audit.csv"
)

EXTREME_AUDIT_FILE = (
    DATA_LIVE
    / "portfolio_extreme_audit.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "agent_output_raw.json"
)


SOURCE_SYSTEM = "PORTFOLIO_B3_OPERATIONAL"
EXPORT_VERSION = "1.0"

EXPECTED_PORTFOLIO_SIZE = 12


# ============================================================
# SERIALIZAÇÃO SEGURA
# ============================================================

def json_safe(value: Any) -> Any:
    """
    Converte objetos pandas/numpy/datetime para tipos
    compatíveis com JSON.
    """

    if value is None:
        return None

    if isinstance(value, (str, bool)):
        return value

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, (datetime, pd.Timestamp)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return [
            json_safe(item)
            for item in value.tolist()
        ]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    return str(value)


# ============================================================
# LEITURA SEGURA DE CSV
# ============================================================

def read_csv_optional(
    path: Path,
) -> pd.DataFrame:
    """
    Lê CSV quando existir.

    Arquivos complementares ausentes não impedem
    necessariamente a exportação da carteira principal.
    """

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            path,
            low_memory=False,
        )

        return df

    except Exception as exc:
        print(
            f"[WARNING] Não foi possível ler "
            f"{path.name}: {exc}"
        )

        return pd.DataFrame()


# ============================================================
# NORMALIZAÇÃO DE TICKER
# ============================================================

def normalize_ticker_column(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    out = df.copy()

    if "TICKER" in out.columns:
        out["TICKER"] = (
            out["TICKER"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    return out


# ============================================================
# DATAFRAME -> REGISTROS JSON
# ============================================================

def dataframe_records(
    df: pd.DataFrame,
) -> list[dict]:

    if df.empty:
        return []

    records = df.to_dict(
        orient="records"
    )

    return [
        json_safe(record)
        for record in records
    ]


# ============================================================
# VALIDAÇÃO DA CARTEIRA PRINCIPAL
# ============================================================

def validate_portfolio(
    portfolio: pd.DataFrame,
) -> dict:

    missing_fields = []
    warnings = []

    if portfolio.empty:
        return {
            "valid": False,
            "number_of_stocks": 0,
            "number_of_sectors": 0,
            "sector_counts": {},
            "duplicate_tickers": 0,
            "missing_fields": [
                "portfolio_current.csv"
            ],
            "warnings": [
                "Carteira operacional ausente ou vazia."
            ],
        }

    required = [
        "TICKER",
        "MACRO_SECTOR",
        "DISCOUNT_52W",
        "FINAL_SCORE",
    ]

    for column in required:
        if column not in portfolio.columns:
            missing_fields.append(column)

    duplicate_tickers = 0

    if "TICKER" in portfolio.columns:
        duplicate_tickers = int(
            portfolio["TICKER"]
            .duplicated()
            .sum()
        )

        if duplicate_tickers:
            warnings.append(
                f"{duplicate_tickers} ticker(s) duplicado(s)."
            )

    number_of_stocks = len(portfolio)

    if number_of_stocks != EXPECTED_PORTFOLIO_SIZE:
        warnings.append(
            "Quantidade de ações diferente da arquitetura "
            f"esperada: {number_of_stocks} encontradas; "
            f"{EXPECTED_PORTFOLIO_SIZE} esperadas."
        )

    sector_counts = {}

    if "MACRO_SECTOR" in portfolio.columns:

        sector_counts = (
            portfolio
            .groupby("MACRO_SECTOR")
            .size()
            .astype(int)
            .to_dict()
        )

    number_of_sectors = len(
        sector_counts
    )

    if number_of_sectors != 4:
        warnings.append(
            "Quantidade de setores diferente da arquitetura "
            f"esperada: {number_of_sectors} encontrados; "
            "4 esperados."
        )

    if sector_counts:

        invalid_sector_counts = {
            sector: count
            for sector, count
            in sector_counts.items()
            if int(count) != 3
        }

        if invalid_sector_counts:
            warnings.append(
                "Composição setorial diferente de "
                f"3 ações por setor: "
                f"{invalid_sector_counts}"
            )

    valid = (
        not missing_fields
        and duplicate_tickers == 0
        and number_of_stocks
        == EXPECTED_PORTFOLIO_SIZE
        and number_of_sectors == 4
        and all(
            int(count) == 3
            for count
            in sector_counts.values()
        )
    )

    return {
        "valid": bool(valid),
        "number_of_stocks": int(
            number_of_stocks
        ),
        "number_of_sectors": int(
            number_of_sectors
        ),
        "sector_counts": json_safe(
            sector_counts
        ),
        "duplicate_tickers": int(
            duplicate_tickers
        ),
        "missing_fields": missing_fields,
        "warnings": warnings,
    }


# ============================================================
# RESUMO DOS SINAIS TÉCNICOS
# ============================================================

def build_signal_summary(
    technical: pd.DataFrame,
) -> dict:

    if technical.empty:
        return {}

    possible_signal_columns = [
        "SIGNAL",
        "ENTRY_SIGNAL",
        "OPERATIONAL_SIGNAL",
        "SIGNAL_STATUS",
        "TECHNICAL_SIGNAL",
        "SINAL",
        "SINAL_OPERACIONAL",
    ]

    signal_column = None

    for column in possible_signal_columns:
        if column in technical.columns:
            signal_column = column
            break

    if signal_column is None:
        return {}

    series = (
        technical[signal_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    if series.empty:
        return {}

    counts = (
        series
        .value_counts()
        .to_dict()
    )

    return {
        "source_column": signal_column,
        "counts": json_safe(counts),
    }


# ============================================================
# RESUMO SETORIAL
# ============================================================

def build_sector_summary(
    portfolio: pd.DataFrame,
) -> list[dict]:

    if (
        portfolio.empty
        or "MACRO_SECTOR"
        not in portfolio.columns
    ):
        return []

    result = []

    for sector, group in portfolio.groupby(
        "MACRO_SECTOR",
        sort=True,
    ):

        tickers = []

        if "TICKER" in group.columns:
            tickers = (
                group["TICKER"]
                .astype(str)
                .tolist()
            )

        row = {
            "sector": str(sector),
            "number_of_stocks": int(
                len(group)
            ),
            "tickers": tickers,
        }

        if "FINAL_SCORE" in group.columns:

            scores = pd.to_numeric(
                group["FINAL_SCORE"],
                errors="coerce",
            )

            if scores.notna().any():
                row[
                    "mean_final_score"
                ] = float(
                    scores.mean()
                )

        if "DISCOUNT_52W" in group.columns:

            discounts = pd.to_numeric(
                group["DISCOUNT_52W"],
                errors="coerce",
            )

            if discounts.notna().any():
                row[
                    "mean_discount_52w"
                ] = float(
                    discounts.mean()
                )

        result.append(
            json_safe(row)
        )

    return result


# ============================================================
# MERGE SOMENTE PARA EXPOSIÇÃO
# ============================================================

def build_positions(
    portfolio: pd.DataFrame,
    technical: pd.DataFrame,
) -> list[dict]:
    """
    Consolida os outputs existentes por ticker.

    IMPORTANTE:
    este merge NÃO calcula novos sinais.
    Apenas coloca no mesmo registro informações
    que já foram produzidas pelo motor.
    """

    if portfolio.empty:
        return []

    base = normalize_ticker_column(
        portfolio
    )

    if (
        not technical.empty
        and "TICKER" in technical.columns
        and "TICKER" in base.columns
    ):

        tech = normalize_ticker_column(
            technical
        )

        technical_columns = [
            column
            for column in tech.columns
            if column != "TICKER"
        ]

        rename_map = {}

        for column in technical_columns:

            if column in base.columns:
                rename_map[column] = (
                    f"TECHNICAL_{column}"
                )

        if rename_map:
            tech = tech.rename(
                columns=rename_map
            )

        base = base.merge(
            tech,
            on="TICKER",
            how="left",
            validate="one_to_one",
        )

    return dataframe_records(
        base
    )


# ============================================================
# RESUMO DA AUDITORIA TÉCNICA
# ============================================================

def build_technical_audit_summary(
    audit: pd.DataFrame,
) -> dict:

    if audit.empty:
        return {
            "available": False,
            "rows": 0,
        }

    summary = {
        "available": True,
        "rows": int(len(audit)),
    }

    status_columns = [
        "STATUS",
        "AUDIT_STATUS",
        "TECHNICAL_STATUS",
        "RESULT",
    ]

    for column in status_columns:

        if column in audit.columns:

            counts = (
                audit[column]
                .fillna("MISSING")
                .astype(str)
                .value_counts()
                .to_dict()
            )

            summary[
                f"{column.lower()}_counts"
            ] = json_safe(counts)

    return summary


# ============================================================
# RESUMO DA AUDITORIA DE PREÇOS
# ============================================================

def build_extreme_audit_summary(
    audit: pd.DataFrame,
) -> dict:

    if audit.empty:
        return {
            "available": False,
            "rows": 0,
        }

    summary = {
        "available": True,
        "rows": int(len(audit)),
    }

    for column in [
        "ENGINE_STATUS",
        "EXTERNAL_STATUS",
        "PRICE_QUALITY_STATUS",
    ]:

        if column in audit.columns:

            counts = (
                audit[column]
                .fillna("MISSING")
                .astype(str)
                .value_counts()
                .to_dict()
            )

            summary[
                f"{column.lower()}_counts"
            ] = json_safe(counts)

    return summary


# ============================================================
# EXPORTAÇÃO
# ============================================================

def export_agent_output(
    generated_at: datetime | None = None,
) -> str:

    print()
    print("=" * 78)
    print(
        "PORTFOLIO-B3-OPERATIONAL — "
        "EXPORT INVESTMENT CIO AGENT"
    )
    print("=" * 78)

    if generated_at is None:
        generated_at = datetime.now(
            timezone.utc
        )

    # --------------------------------------------------------
    # Carteira oficial
    # --------------------------------------------------------

    if not PORTFOLIO_FILE.exists():

        raise FileNotFoundError(
            "Arquivo operacional obrigatório não encontrado: "
            f"{PORTFOLIO_FILE}"
        )

    portfolio = pd.read_csv(
        PORTFOLIO_FILE,
        low_memory=False,
    )

    portfolio = normalize_ticker_column(
        portfolio
    )

    # --------------------------------------------------------
    # Outputs complementares
    # --------------------------------------------------------

    technical = read_csv_optional(
        TECHNICAL_FILE
    )

    technical = normalize_ticker_column(
        technical
    )

    technical_audit = read_csv_optional(
        TECHNICAL_AUDIT_FILE
    )

    technical_audit = (
        normalize_ticker_column(
            technical_audit
        )
    )

    extreme_audit = read_csv_optional(
        EXTREME_AUDIT_FILE
    )

    extreme_audit = (
        normalize_ticker_column(
            extreme_audit
        )
    )

    # --------------------------------------------------------
    # Auditoria estrutural
    # --------------------------------------------------------

    portfolio_audit = (
        validate_portfolio(
            portfolio
        )
    )

    signal_summary = (
        build_signal_summary(
            technical
        )
    )

    sector_summary = (
        build_sector_summary(
            portfolio
        )
    )

    technical_audit_summary = (
        build_technical_audit_summary(
            technical_audit
        )
    )

    extreme_audit_summary = (
        build_extreme_audit_summary(
            extreme_audit
        )
    )

    positions = build_positions(
        portfolio=portfolio,
        technical=technical,
    )

    # --------------------------------------------------------
    # Arquivos disponíveis
    # --------------------------------------------------------

    source_files = {
        "portfolio_current": {
            "path": str(
                PORTFOLIO_FILE.relative_to(ROOT)
            ),
            "available": PORTFOLIO_FILE.exists(),
        },
        "portfolio_technical_current": {
            "path": str(
                TECHNICAL_FILE.relative_to(ROOT)
            ),
            "available": TECHNICAL_FILE.exists(),
        },
        "portfolio_technical_audit": {
            "path": str(
                TECHNICAL_AUDIT_FILE.relative_to(ROOT)
            ),
            "available": TECHNICAL_AUDIT_FILE.exists(),
        },
        "portfolio_extreme_audit": {
            "path": str(
                EXTREME_AUDIT_FILE.relative_to(ROOT)
            ),
            "available": EXTREME_AUDIT_FILE.exists(),
        },
    }

    # --------------------------------------------------------
    # Payload
    # --------------------------------------------------------

    payload = {

        "source_system":
            SOURCE_SYSTEM,

        "export_version":
            EXPORT_VERSION,

        "generated_at":
            generated_at.isoformat(),

        "engine": {

            "name":
                "Portfolio-B3-Operational",

            "role":
                "BRAZIL_EQUITY_SELECTION",

            "asset_class":
                "BRAZIL_EQUITIES",

            "portfolio_size":
                EXPECTED_PORTFOLIO_SIZE,

            "architecture":
                "4_SECTORS_X_3_STOCKS",

            "sector_rule":
                "TOP4_1Y",

            "stock_rule":
                "DISCOUNT_80_FUNDAMENTALS_20",

            "technical_layer":
                "COMPLEMENTARY_NON_OVERRIDE",
        },

        "portfolio_audit":
            portfolio_audit,

        "signal_summary":
            signal_summary,

        "sector_summary":
            sector_summary,

        "technical_audit":
            technical_audit_summary,

        "price_audit":
            extreme_audit_summary,

        "positions":
            positions,

        "raw_outputs": {

            "portfolio":
                dataframe_records(
                    portfolio
                ),

            "technical":
                dataframe_records(
                    technical
                ),

            "technical_audit":
                dataframe_records(
                    technical_audit
                ),

            "price_audit":
                dataframe_records(
                    extreme_audit
                ),
        },

        "source_files":
            source_files,

        "policy": {

            "selection_recalculated":
                False,

            "technical_signal_recalculated":
                False,

            "weights_recalculated":
                False,

            "source_decisions_preserved":
                True,

            "portfolio_selection_overridden":
                False,

            "technical_layer_overrides_selection":
                False,

            "broker_execution_allowed":
                False,
        },
    }

    # --------------------------------------------------------
    # Gravação
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_safe(payload),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print()
    print(
        f"Arquivo criado: "
        f"{OUTPUT_FILE.relative_to(ROOT)}"
    )

    print(
        "Ações exportadas: "
        f"{portfolio_audit['number_of_stocks']}"
    )

    print(
        "Setores: "
        f"{portfolio_audit['number_of_sectors']}"
    )

    print(
        "Estrutura válida: "
        f"{portfolio_audit['valid']}"
    )

    print(
        "Camada técnica disponível: "
        f"{not technical.empty}"
    )

    print(
        "Auditoria técnica disponível: "
        f"{not technical_audit.empty}"
    )

    print(
        "Auditoria de preços disponível: "
        f"{not extreme_audit.empty}"
    )

    print("=" * 78)
    print(
        "EXPORTAÇÃO PARA O INVESTMENT CIO "
        "CONCLUÍDA"
    )
    print("=" * 78)

    return str(
        OUTPUT_FILE
    )


# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    export_agent_output()
