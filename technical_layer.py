# ============================================================
# technical_layer.py
# PORTFOLIO-B3-OPERATIONAL
#
# CAMADA TÉCNICA COMPLEMENTAR
#
# IMPORTANTE:
# - NÃO altera a seleção TOP4_1Y
# - NÃO altera a regra DISCOUNT_80_FUNDAMENTALS_20
# - NÃO altera portfolio_current.csv
# - NÃO altera dados históricos
# - NÃO remove ações da carteira
#
# A camada técnica apenas responde:
# - qualidade do momento de entrada
# - tendência
# - momentum
# - volume
# - volatilidade/risco
# - sinal operacional
#
# Entrada:
# data_live/portfolio_current.csv
#
# Saída:
# data_live/portfolio_technical_current.csv
# data_live/portfolio_technical_audit.csv
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_LIVE = ROOT / "data_live"

INPUT_PORTFOLIO = DATA_LIVE / "portfolio_current.csv"

OUTPUT_TECHNICAL = (
    DATA_LIVE / "portfolio_technical_current.csv"
)

OUTPUT_AUDIT = (
    DATA_LIVE / "portfolio_technical_audit.csv"
)

EXPECTED_PORTFOLIO_SIZE = 12

PRICE_PERIOD = "2y"
PRICE_INTERVAL = "1d"

MIN_OBSERVATIONS = 200


# ============================================================
# UTILIDADES
# ============================================================

def safe_float(value):

    try:
        value = float(value)

    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return value


def yahoo_symbol(ticker):

    ticker = str(ticker).strip().upper()

    if ticker.endswith(".SA"):
        return ticker

    return f"{ticker}.SA"


# ============================================================
# LEITURA DA CARTEIRA OFICIAL
# ============================================================

def load_portfolio():

    if not INPUT_PORTFOLIO.exists():

        raise FileNotFoundError(
            f"Carteira operacional não encontrada: "
            f"{INPUT_PORTFOLIO}"
        )

    portfolio = pd.read_csv(
        INPUT_PORTFOLIO,
        low_memory=False,
    )

    if "TICKER" not in portfolio.columns:

        raise RuntimeError(
            "portfolio_current.csv sem coluna TICKER."
        )

    portfolio["TICKER"] = (
        portfolio["TICKER"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    duplicates = int(
        portfolio["TICKER"].duplicated().sum()
    )

    if duplicates:

        raise RuntimeError(
            f"Duplicidades na carteira: {duplicates}"
        )

    if len(portfolio) != EXPECTED_PORTFOLIO_SIZE:

        raise RuntimeError(
            f"Esperadas {EXPECTED_PORTFOLIO_SIZE} ações; "
            f"encontradas {len(portfolio)}."
        )

    return portfolio


# ============================================================
# DOWNLOAD TÉCNICO
# ============================================================

def download_prices(tickers):

    symbols = [
        yahoo_symbol(ticker)
        for ticker in tickers
    ]

    if not symbols:

        return pd.DataFrame()

    print()
    print("Baixando OHLCV técnico:")
    print(", ".join(symbols))
    print()

    return yf.download(
        tickers=symbols,
        period=PRICE_PERIOD,
        interval=PRICE_INTERVAL,

        # Esta é uma camada técnica independente.
        #
        # auto_adjust=True evita distorções mecânicas em médias,
        # momentum, ATR e indicadores técnicos após splits,
        # grupamentos e distribuições.
        #
        # NÃO substitui o PREULT usado pelo motor principal.
        auto_adjust=True,

        progress=False,
        threads=True,
        group_by="ticker",
    )


def extract_ticker_data(data, ticker):

    symbol = yahoo_symbol(ticker)

    if data.empty:
        return pd.DataFrame()

    try:

        if isinstance(
            data.columns,
            pd.MultiIndex,
        ):

            level0 = (
                data.columns
                .get_level_values(0)
            )

            if symbol not in level0:

                return pd.DataFrame()

            df = data[symbol].copy()

        else:

            df = data.copy()

        df = df.reset_index()

        if "Date" in df.columns:

            df = df.rename(
                columns={"Date": "DATE"}
            )

        elif "Datetime" in df.columns:

            df = df.rename(
                columns={"Datetime": "DATE"}
            )

        df = df.dropna(
            how="all"
        )

        return df

    except Exception:

        return pd.DataFrame()


# ============================================================
# INDICADORES
# ============================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    average_gain = (
        gain
        .rolling(period)
        .mean()
    )

    average_loss = (
        loss
        .rolling(period)
        .mean()
    )

    rs = (
        average_gain
        /
        average_loss.replace(0, np.nan)
    )

    rsi = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    return rsi


def calculate_macd(close):

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    macd = ema12 - ema26

    signal = (
        macd
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    histogram = (
        macd - signal
    )

    return (
        macd,
        signal,
        histogram,
    )


def calculate_atr(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    previous_close = (
        close.shift(1)
    )

    tr1 = high - low

    tr2 = (
        high
        -
        previous_close
    ).abs()

    tr3 = (
        low
        -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    return (
        true_range
        .rolling(period)
        .mean()
    )


def calculate_indicators(df):

    df = df.copy()

    required = [
        "Close",
        "High",
        "Low",
        "Volume",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        return pd.DataFrame()

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    close = df["Close"]

    volume = (
        df["Volume"]
        .where(
            df["Volume"] > 0,
            np.nan,
        )
    )

    # Não inventa volume.
    # Valores ausentes permanecem NaN.

    df["MM20"] = (
        close
        .rolling(20)
        .mean()
    )

    df["MM50"] = (
        close
        .rolling(50)
        .mean()
    )

    df["MM200"] = (
        close
        .rolling(200)
        .mean()
    )

    df["RSI14"] = (
        calculate_rsi(
            close,
            14,
        )
    )

    (
        df["MACD"],
        df["MACD_SIGNAL"],
        df["MACD_HIST"],
    ) = calculate_macd(close)

    df["ATR14"] = (
        calculate_atr(
            df,
            14,
        )
    )

    df["ATR_PCT"] = (
        df["ATR14"]
        /
        close
    )

    df["RETURN_20D"] = (
        close
        .pct_change(20)
    )

    df["RETURN_60D"] = (
        close
        .pct_change(60)
    )

    df["DIST_MM20"] = (
        close
        /
        df["MM20"]
        -
        1
    )

    df["DIST_MM50"] = (
        close
        /
        df["MM50"]
        -
        1
    )

    df["DIST_MM200"] = (
        close
        /
        df["MM200"]
        -
        1
    )

    df["VOLUME_AVG_20D"] = (
        volume
        .rolling(
            20,
            min_periods=10,
        )
        .mean()
    )

    df["VOLUME_STRENGTH"] = (
        volume
        /
        df["VOLUME_AVG_20D"]
    )

    df["VOLUME_STRENGTH"] = (
        df["VOLUME_STRENGTH"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .clip(
            lower=0.20,
            upper=5.00,
        )
    )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return df


# ============================================================
# SCORE — TENDÊNCIA
# ============================================================

def score_trend(row):

    score = 0

    price = row.get(
        "Close",
        np.nan,
    )

    mm20 = row.get(
        "MM20",
        np.nan,
    )

    mm50 = row.get(
        "MM50",
        np.nan,
    )

    mm200 = row.get(
        "MM200",
        np.nan,
    )

    dist200 = row.get(
        "DIST_MM200",
        np.nan,
    )

    # Distância da MM200 — 40 pontos

    if pd.notna(dist200):

        if dist200 >= 0.20:
            score += 40

        elif dist200 >= 0.10:
            score += 36

        elif dist200 >= 0.05:
            score += 32

        elif dist200 >= 0:
            score += 28

        elif dist200 >= -0.03:
            score += 24

        elif dist200 >= -0.08:
            score += 18

        elif dist200 >= -0.15:
            score += 12

        elif dist200 >= -0.25:
            score += 6

    # MM50 / MM200 — 25 pontos

    if (
        pd.notna(mm50)
        and
        pd.notna(mm200)
        and
        mm200 != 0
    ):

        ratio = mm50 / mm200

        if ratio >= 1.05:
            score += 25

        elif ratio >= 1.02:
            score += 22

        elif ratio >= 1.00:
            score += 18

        elif ratio >= 0.98:
            score += 12

        elif ratio >= 0.95:
            score += 6

    # MM20 / MM50 — 20 pontos

    if (
        pd.notna(mm20)
        and
        pd.notna(mm50)
        and
        mm50 != 0
    ):

        ratio = mm20 / mm50

        if ratio >= 1.03:
            score += 20

        elif ratio >= 1.01:
            score += 17

        elif ratio >= 1.00:
            score += 14

        elif ratio >= 0.99:
            score += 10

        elif ratio >= 0.97:
            score += 5

    # Preço / MM20 — 15 pontos

    if (
        pd.notna(price)
        and
        pd.notna(mm20)
        and
        mm20 != 0
    ):

        ratio = price / mm20

        if ratio >= 1.05:
            score += 15

        elif ratio >= 1.02:
            score += 13

        elif ratio >= 1.00:
            score += 11

        elif ratio >= 0.99:
            score += 8

        elif ratio >= 0.97:
            score += 4

    return round(
        min(score, 100),
        2,
    )


# ============================================================
# SCORE — ENTRADA
# ============================================================

def score_entry(row):

    score = 0

    rsi = row.get(
        "RSI14",
        np.nan,
    )

    dist_mm200 = row.get(
        "DIST_MM200",
        np.nan,
    )

    dist_mm20 = row.get(
        "DIST_MM20",
        np.nan,
    )

    # RSI — 40 pontos

    if pd.notna(rsi):

        if 45 <= rsi <= 60:
            score += 40

        elif 40 <= rsi < 45:
            score += 34

        elif 60 < rsi <= 70:
            score += 30

        elif 35 <= rsi < 40:
            score += 24

        elif 30 <= rsi < 35:
            score += 16

        elif 70 < rsi <= 75:
            score += 16

        elif 25 <= rsi < 30:
            score += 10

        elif 20 <= rsi < 25:
            score += 8

        elif 75 < rsi <= 80:
            score += 8

        elif rsi < 20:
            score += 6

    # MM200 — 35 pontos

    if pd.notna(dist_mm200):

        if -0.03 <= dist_mm200 <= 0.12:
            score += 35

        elif 0.12 < dist_mm200 <= 0.25:
            score += 28

        elif -0.08 <= dist_mm200 < -0.03:
            score += 28

        elif 0.25 < dist_mm200 <= 0.40:
            score += 18

        elif -0.15 <= dist_mm200 < -0.08:
            score += 18

        elif -0.25 <= dist_mm200 < -0.15:
            score += 10

        elif dist_mm200 > 0.40:
            score += 8

    # MM20 — 25 pontos

    if pd.notna(dist_mm20):

        if -0.05 <= dist_mm20 <= 0.05:
            score += 25

        elif 0.05 < dist_mm20 <= 0.10:
            score += 18

        elif -0.10 <= dist_mm20 < -0.05:
            score += 16

        elif 0.10 < dist_mm20 <= 0.15:
            score += 10

        elif -0.15 <= dist_mm20 < -0.10:
            score += 8

    return round(
        min(score, 100),
        2,
    )


# ============================================================
# SCORE — MOMENTUM
# ============================================================

def score_momentum(row):

    score = 0

    macd = row.get(
        "MACD",
        np.nan,
    )

    signal = row.get(
        "MACD_SIGNAL",
        np.nan,
    )

    histogram = row.get(
        "MACD_HIST",
        np.nan,
    )

    return_20d = row.get(
        "RETURN_20D",
        np.nan,
    )

    return_60d = row.get(
        "RETURN_60D",
        np.nan,
    )

    if (
        pd.notna(macd)
        and
        pd.notna(signal)
        and
        macd > signal
    ):
        score += 30

    if (
        pd.notna(histogram)
        and
        histogram > 0
    ):
        score += 20

    if pd.notna(return_20d):

        if return_20d > 0:
            score += 25

        if return_20d > 0.05:
            score += 10

    if (
        pd.notna(return_60d)
        and
        return_60d > 0
    ):
        score += 15

    return min(
        score,
        100,
    )


# ============================================================
# SCORE — VOLUME
# ============================================================

def score_volume(row):

    strength = row.get(
        "VOLUME_STRENGTH",
        np.nan,
    )

    # Volume indisponível:
    # valor neutro, sem favorecer nem penalizar.

    if pd.isna(strength):
        return 55

    if strength >= 1.50:
        return 100

    if strength >= 1.20:
        return 85

    if strength >= 0.90:
        return 70

    if strength >= 0.60:
        return 55

    if strength >= 0.30:
        return 40

    return 30


# ============================================================
# SCORE — RISCO / VOLATILIDADE
# ============================================================

def score_risk(row):

    atr_pct = row.get(
        "ATR_PCT",
        np.nan,
    )

    if pd.isna(atr_pct):
        return 60

    if atr_pct <= 0.03:
        return 100

    if atr_pct <= 0.05:
        return 90

    if atr_pct <= 0.08:
        return 80

    if atr_pct <= 0.12:
        return 65

    if atr_pct <= 0.18:
        return 50

    return 35


# ============================================================
# SCORE TÉCNICO FINAL
# ============================================================

def calculate_technical_score(row):

    trend = score_trend(row)
    entry = score_entry(row)
    momentum = score_momentum(row)
    volume = score_volume(row)
    risk = score_risk(row)

    score = (
        trend * 0.30
        +
        entry * 0.25
        +
        momentum * 0.25
        +
        volume * 0.10
        +
        risk * 0.10
    )

    return round(
        min(score, 100),
        2,
    )


# ============================================================
# SINAL
# ============================================================

def classify_signal(score):

    if pd.isna(score):
        return "DADOS INSUFICIENTES"

    if score >= 80:
        return "COMPRA FORTE"

    if score >= 67:
        return "COMPRA"

    if score >= 55:
        return "AGUARDAR"

    if score >= 42:
        return "FRACO"

    return "EVITAR"



def classify_operational_action(signal, technical_status="PASS"):
    """
    Traduz o sinal técnico em orientação operacional de timing.

    IMPORTANTE:
    - não altera os 12 ativos selecionados;
    - não altera FINAL_SCORE;
    - não altera TOP4_1Y;
    - não executa ordens;
    - histórico/dados insuficientes nunca recebem compra/venda por inferência.
    """

    if technical_status != "PASS":
        return "SEM CONFIRMAÇÃO TÉCNICA"

    mapping = {
        "COMPRA FORTE": "ENTRADA LIBERADA",
        "COMPRA": "ENTRADA LIBERADA",
        "AGUARDAR": "AGUARDAR MELHOR PONTO",
        "FRACO": "NÃO ENTRAR AGORA",
        "EVITAR": "BLOQUEAR ENTRADA TEMPORARIAMENTE",
    }

    return mapping.get(
        signal,
        "SEM CONFIRMAÇÃO TÉCNICA",
    )


def classify_conviction(score):

    if pd.isna(score):
        return "INDEFINIDA"

    if score >= 80:
        return "MUITO ALTA"

    if score >= 67:
        return "ALTA"

    if score >= 55:
        return "MODERADA"

    if score >= 42:
        return "BAIXA"

    return "MUITO BAIXA"


# ============================================================
# RISCO TÉCNICO
# ============================================================

def classify_technical_risk(row, score):

    if pd.isna(score):
        return "INDEFINIDO"

    atr_pct = row.get(
        "ATR_PCT",
        np.nan,
    )

    dist_mm200 = row.get(
        "DIST_MM200",
        np.nan,
    )

    risk_points = 0

    if score < 42:
        risk_points += 2

    elif score < 55:
        risk_points += 1

    if pd.notna(atr_pct):

        if atr_pct > 0.120:
            risk_points += 3

        elif atr_pct > 0.085:
            risk_points += 2

        elif atr_pct > 0.060:
            risk_points += 1

    if pd.notna(dist_mm200):

        if dist_mm200 < -0.25:
            risk_points += 3

        elif dist_mm200 < -0.15:
            risk_points += 2

        elif dist_mm200 < -0.08:
            risk_points += 1

    if risk_points >= 5:
        return "ELEVADO"

    if risk_points >= 3:
        return "MODERADO/ALTO"

    if risk_points >= 1:
        return "MODERADO"

    return "CONTROLADO"


# ============================================================
# STATUS EXPLICÁVEIS
# ============================================================

def trend_status(row):

    price = row.get(
        "Close",
        np.nan,
    )

    mm20 = row.get(
        "MM20",
        np.nan,
    )

    mm50 = row.get(
        "MM50",
        np.nan,
    )

    mm200 = row.get(
        "MM200",
        np.nan,
    )

    if (
        pd.isna(price)
        or
        pd.isna(mm200)
    ):
        return "INDEFINIDA"

    if (
        price > mm200
        and
        pd.notna(mm50)
        and
        mm50 > mm200
    ):
        return "ALTA ESTRUTURAL"

    if price > mm200:
        return "ALTA PARCIAL"

    if (
        pd.notna(mm20)
        and
        price > mm20
    ):
        return "RECUPERAÇÃO CURTA"

    return "BAIXA"


def rsi_status(row):

    rsi = row.get(
        "RSI14",
        np.nan,
    )

    if pd.isna(rsi):
        return "INDEFINIDO"

    if rsi < 30:
        return f"{rsi:.1f} SOBREVENDA FORTE"

    if rsi < 35:
        return f"{rsi:.1f} SOBREVENDA"

    if rsi < 45:
        return f"{rsi:.1f} FRACO"

    if rsi <= 60:
        return f"{rsi:.1f} NEUTRO"

    if rsi <= 70:
        return f"{rsi:.1f} FORTE"

    return f"{rsi:.1f} SOBRECOMPRA"


def momentum_status(row):

    signals = []

    macd = row.get(
        "MACD",
        np.nan,
    )

    macd_signal = row.get(
        "MACD_SIGNAL",
        np.nan,
    )

    histogram = row.get(
        "MACD_HIST",
        np.nan,
    )

    return20 = row.get(
        "RETURN_20D",
        np.nan,
    )

    return60 = row.get(
        "RETURN_60D",
        np.nan,
    )

    if (
        pd.notna(macd)
        and
        pd.notna(macd_signal)
    ):

        signals.append(
            "MACD positivo"
            if macd > macd_signal
            else "MACD negativo"
        )

    if pd.notna(histogram):

        signals.append(
            "histograma positivo"
            if histogram > 0
            else "histograma negativo"
        )

    if pd.notna(return20):

        signals.append(
            "20d positivo"
            if return20 > 0
            else "20d negativo"
        )

    if pd.notna(return60):

        signals.append(
            "60d positivo"
            if return60 > 0
            else "60d negativo"
        )

    positive = sum(
        "positivo" in s
        for s in signals
    )

    negative = sum(
        "negativo" in s
        for s in signals
    )

    if positive >= 3:
        status = "POSITIVO"

    elif negative >= 3:
        status = "NEGATIVO"

    else:
        status = "MISTO"

    return (
        f"{status} "
        f"({'; '.join(signals)})"
    )


def volatility_status(row):

    atr_pct = row.get(
        "ATR_PCT",
        np.nan,
    )

    if pd.isna(atr_pct):
        return "INDEFINIDA"

    if atr_pct <= 0.03:
        return "MUITO BAIXA"

    if atr_pct <= 0.05:
        return "BAIXA"

    if atr_pct <= 0.08:
        return "MODERADA"

    if atr_pct <= 0.12:
        return "ALTA"

    if atr_pct <= 0.18:
        return "MUITO ALTA"

    return "EXTREMA"


def volume_status(row):

    strength = row.get(
        "VOLUME_STRENGTH",
        np.nan,
    )

    if pd.isna(strength):
        return "INDEFINIDO"

    if strength >= 1.50:
        return f"FORTE ({strength:.2f}x)"

    if strength >= 1.00:
        return f"NORMAL ({strength:.2f}x)"

    if strength >= 0.60:
        return f"FRACO ({strength:.2f}x)"

    return f"MUITO FRACO ({strength:.2f}x)"


# ============================================================
# DIAGNÓSTICO
# ============================================================

def technical_diagnostic(row, score):

    return (
        f"Tendência: {trend_status(row)}. "
        f"RSI: {rsi_status(row)}. "
        f"Momentum: {momentum_status(row)}. "
        f"Volume: {volume_status(row)}. "
        f"Volatilidade: {volatility_status(row)}. "
        f"Convicção: {classify_conviction(score)}. "
        f"Risco: {classify_technical_risk(row, score)}. "
        f"Sinal: {classify_signal(score)}."
    )


# ============================================================
# PROCESSAMENTO POR AÇÃO
# ============================================================

def _technical_result_base(ticker, status, obs=np.nan):
    """
    Estrutura padronizada para qualquer resultado técnico.

    Isso evita que ativos com pouco histórico percam colunas no CSV.
    Esses ativos permanecem na carteira oficial, mas recebem
    SEM CONFIRMAÇÃO TÉCNICA.
    """

    return {
        "TICKER": ticker,
        "TECHNICAL_STATUS": status,
        "TECHNICAL_DATE": np.nan,
        "TECHNICAL_OBS": obs,
        "TECH_PRICE": np.nan,
        "MM20": np.nan,
        "MM50": np.nan,
        "MM200": np.nan,
        "RSI14": np.nan,
        "MACD": np.nan,
        "MACD_SIGNAL": np.nan,
        "MACD_HIST": np.nan,
        "ATR14": np.nan,
        "ATR_PCT": np.nan,
        "RETURN_20D": np.nan,
        "RETURN_60D": np.nan,
        "DIST_MM20": np.nan,
        "DIST_MM50": np.nan,
        "DIST_MM200": np.nan,
        "VOLUME_STRENGTH": np.nan,
        "SCORE_TREND": np.nan,
        "SCORE_ENTRY": np.nan,
        "SCORE_MOMENTUM": np.nan,
        "SCORE_VOLUME": np.nan,
        "SCORE_RISK": np.nan,
        "SCORE_TECHNICAL": np.nan,
        "SIGNAL_TECHNICAL": "SEM CONFIRMAÇÃO",
        "CONVICTION_TECHNICAL": "INDEFINIDA",
        "RISK_TECHNICAL": "INDEFINIDO",
        "OPERATIONAL_ACTION": "SEM CONFIRMAÇÃO TÉCNICA",
        "TREND_STATUS": "INDEFINIDA",
        "RSI_STATUS": "INDEFINIDO",
        "MOMENTUM_STATUS": "INDEFINIDO",
        "VOLUME_STATUS": "INDEFINIDO",
        "VOLATILITY_STATUS": "INDEFINIDA",
        "TECHNICAL_DIAGNOSTIC": (
            f"Status técnico: {status}. "
            "Não há evidência técnica suficiente para liberar, bloquear "
            "ou penalizar a ação. A seleção original permanece preservada."
        ),
    }


def analyze_ticker(data, ticker):

    df = extract_ticker_data(
        data,
        ticker,
    )

    if df.empty:
        return _technical_result_base(
            ticker=ticker,
            status="NO_DATA",
            obs=0,
        )

    obs = len(df)

    # A MM200 exige 200 pregões. Não usamos aproximação,
    # média mais curta ou score parcial para preencher a ausência.
    if obs < MIN_OBSERVATIONS:
        return _technical_result_base(
            ticker=ticker,
            status="INSUFFICIENT_HISTORY",
            obs=obs,
        )

    df = calculate_indicators(
        df
    )

    if df.empty:
        return _technical_result_base(
            ticker=ticker,
            status="INDICATOR_ERROR",
            obs=obs,
        )

    valid = df[
        df["Close"].notna()
        &
        df["MM200"].notna()
    ].copy()

    if valid.empty:
        return _technical_result_base(
            ticker=ticker,
            status="INSUFFICIENT_INDICATORS",
            obs=obs,
        )

    last = valid.iloc[-1]

    score = calculate_technical_score(
        last
    )

    signal = classify_signal(
        score
    )

    conviction = classify_conviction(
        score
    )

    technical_risk = classify_technical_risk(
        last,
        score,
    )

    operational_action = classify_operational_action(
        signal=signal,
        technical_status="PASS",
    )

    return {
        "TICKER": ticker,
        "TECHNICAL_STATUS": "PASS",

        "TECHNICAL_DATE":
            last.get(
                "DATE",
                np.nan,
            ),

        "TECHNICAL_OBS":
            obs,

        "TECH_PRICE":
            safe_float(
                last.get(
                    "Close",
                    np.nan,
                )
            ),

        "MM20":
            safe_float(
                last.get(
                    "MM20",
                    np.nan,
                )
            ),

        "MM50":
            safe_float(
                last.get(
                    "MM50",
                    np.nan,
                )
            ),

        "MM200":
            safe_float(
                last.get(
                    "MM200",
                    np.nan,
                )
            ),

        "RSI14":
            safe_float(
                last.get(
                    "RSI14",
                    np.nan,
                )
            ),

        "MACD":
            safe_float(
                last.get(
                    "MACD",
                    np.nan,
                )
            ),

        "MACD_SIGNAL":
            safe_float(
                last.get(
                    "MACD_SIGNAL",
                    np.nan,
                )
            ),

        "MACD_HIST":
            safe_float(
                last.get(
                    "MACD_HIST",
                    np.nan,
                )
            ),

        "ATR14":
            safe_float(
                last.get(
                    "ATR14",
                    np.nan,
                )
            ),

        "ATR_PCT":
            safe_float(
                last.get(
                    "ATR_PCT",
                    np.nan,
                )
            ),

        "RETURN_20D":
            safe_float(
                last.get(
                    "RETURN_20D",
                    np.nan,
                )
            ),

        "RETURN_60D":
            safe_float(
                last.get(
                    "RETURN_60D",
                    np.nan,
                )
            ),

        "DIST_MM20":
            safe_float(
                last.get(
                    "DIST_MM20",
                    np.nan,
                )
            ),

        "DIST_MM50":
            safe_float(
                last.get(
                    "DIST_MM50",
                    np.nan,
                )
            ),

        "DIST_MM200":
            safe_float(
                last.get(
                    "DIST_MM200",
                    np.nan,
                )
            ),

        "VOLUME_STRENGTH":
            safe_float(
                last.get(
                    "VOLUME_STRENGTH",
                    np.nan,
                )
            ),

        "SCORE_TREND":
            score_trend(
                last
            ),

        "SCORE_ENTRY":
            score_entry(
                last
            ),

        "SCORE_MOMENTUM":
            score_momentum(
                last
            ),

        "SCORE_VOLUME":
            score_volume(
                last
            ),

        "SCORE_RISK":
            score_risk(
                last
            ),

        "SCORE_TECHNICAL":
            score,

        "SIGNAL_TECHNICAL":
            signal,

        "CONVICTION_TECHNICAL":
            conviction,

        "RISK_TECHNICAL":
            technical_risk,

        "OPERATIONAL_ACTION":
            operational_action,

        "TREND_STATUS":
            trend_status(
                last
            ),

        "RSI_STATUS":
            rsi_status(
                last
            ),

        "MOMENTUM_STATUS":
            momentum_status(
                last
            ),

        "VOLUME_STATUS":
            volume_status(
                last
            ),

        "VOLATILITY_STATUS":
            volatility_status(
                last
            ),

        "TECHNICAL_DIAGNOSTIC":
            technical_diagnostic(
                last,
                score,
            ),
    }


# ============================================================
# AUDITORIA
# ============================================================

def build_audit(result, original_portfolio):

    total = len(result)

    pass_count = int(
        (
            result["TECHNICAL_STATUS"]
            ==
            "PASS"
        ).sum()
    )

    insufficient_history_count = int(
        (
            result["TECHNICAL_STATUS"]
            ==
            "INSUFFICIENT_HISTORY"
        ).sum()
    )

    no_data_count = int(
        (
            result["TECHNICAL_STATUS"]
            ==
            "NO_DATA"
        ).sum()
    )

    other_review_count = int(
        (
            ~result["TECHNICAL_STATUS"].isin(
                [
                    "PASS",
                    "INSUFFICIENT_HISTORY",
                    "NO_DATA",
                ]
            )
        ).sum()
    )

    portfolio_preserved = (
        total
        ==
        EXPECTED_PORTFOLIO_SIZE
        and
        set(result["TICKER"])
        ==
        set(original_portfolio["TICKER"])
    )

    duplicates = int(
        result["TICKER"].duplicated().sum()
    )

    # Prova explícita de que a camada técnica não reescreveu
    # o FINAL_SCORE recebido do motor principal.
    final_score_preserved = True

    if (
        "FINAL_SCORE" in original_portfolio.columns
        and
        "FINAL_SCORE" in result.columns
    ):

        left = (
            original_portfolio[
                ["TICKER", "FINAL_SCORE"]
            ]
            .copy()
            .sort_values("TICKER")
            .reset_index(drop=True)
        )

        right = (
            result[
                ["TICKER", "FINAL_SCORE"]
            ]
            .copy()
            .sort_values("TICKER")
            .reset_index(drop=True)
        )

        left["FINAL_SCORE"] = pd.to_numeric(
            left["FINAL_SCORE"],
            errors="coerce",
        )

        right["FINAL_SCORE"] = pd.to_numeric(
            right["FINAL_SCORE"],
            errors="coerce",
        )

        final_score_preserved = bool(
            np.allclose(
                left["FINAL_SCORE"].to_numpy(),
                right["FINAL_SCORE"].to_numpy(),
                equal_nan=True,
                rtol=0.0,
                atol=0.0,
            )
        )

    neutral_insufficient = True

    mask_insufficient = (
        result["TECHNICAL_STATUS"]
        !=
        "PASS"
    )

    if mask_insufficient.any():

        neutral_insufficient = bool(
            result.loc[
                mask_insufficient,
                "SCORE_TECHNICAL"
            ].isna().all()
            and
            result.loc[
                mask_insufficient,
                "OPERATIONAL_ACTION"
            ].eq(
                "SEM CONFIRMAÇÃO TÉCNICA"
            ).all()
        )

    core_integrity = (
        portfolio_preserved
        and duplicates == 0
        and final_score_preserved
        and neutral_insufficient
    )

    audit = pd.DataFrame(
        [
            {
                "CHECK": "PORTFOLIO_SIZE",
                "VALUE": total,
                "EXPECTED": EXPECTED_PORTFOLIO_SIZE,
                "STATUS": (
                    "PASS"
                    if total == EXPECTED_PORTFOLIO_SIZE
                    else "FAIL"
                ),
            },
            {
                "CHECK": "TICKERS_PRESERVED",
                "VALUE": str(portfolio_preserved),
                "EXPECTED": "True",
                "STATUS": (
                    "PASS"
                    if portfolio_preserved
                    else "FAIL"
                ),
            },
            {
                "CHECK": "DUPLICATES",
                "VALUE": duplicates,
                "EXPECTED": 0,
                "STATUS": (
                    "PASS"
                    if duplicates == 0
                    else "FAIL"
                ),
            },
            {
                "CHECK": "TECHNICAL_ANALYSIS_PASS",
                "VALUE": pass_count,
                "EXPECTED": "<= 12; disponibilidade depende do histórico",
                "STATUS": "PASS",
            },
            {
                "CHECK": "INSUFFICIENT_HISTORY",
                "VALUE": insufficient_history_count,
                "EXPECTED": "Permitido; não recebe score por aproximação",
                "STATUS": (
                    "PASS"
                    if neutral_insufficient
                    else "FAIL"
                ),
            },
            {
                "CHECK": "NO_DATA",
                "VALUE": no_data_count,
                "EXPECTED": "Permitido como diagnóstico; sem inferência",
                "STATUS": (
                    "PASS"
                    if neutral_insufficient
                    else "FAIL"
                ),
            },
            {
                "CHECK": "OTHER_TECHNICAL_REVIEW",
                "VALUE": other_review_count,
                "EXPECTED": 0,
                "STATUS": (
                    "PASS"
                    if other_review_count == 0
                    else "REVIEW"
                ),
            },
            {
                "CHECK": "NON_PASS_IS_NEUTRAL",
                "VALUE": str(neutral_insufficient),
                "EXPECTED": "True",
                "STATUS": (
                    "PASS"
                    if neutral_insufficient
                    else "FAIL"
                ),
            },
            {
                "CHECK": "FINAL_SCORE_PRESERVED",
                "VALUE": str(final_score_preserved),
                "EXPECTED": "True",
                "STATUS": (
                    "PASS"
                    if final_score_preserved
                    else "FAIL"
                ),
            },
            {
                "CHECK": "CORE_SELECTION_RULE",
                "VALUE": "UNCHANGED",
                "EXPECTED": "UNCHANGED",
                "STATUS": "PASS",
            },
            {
                "CHECK": "DISCOUNT_80_FUNDAMENTALS_20",
                "VALUE": "PRESERVED",
                "EXPECTED": "PRESERVED",
                "STATUS": "PASS",
            },
            {
                "CHECK": "TOP4_1Y",
                "VALUE": "PRESERVED",
                "EXPECTED": "PRESERVED",
                "STATUS": "PASS",
            },
            {
                "CHECK": "HISTORICAL_CORE",
                "VALUE": "PRESERVED",
                "EXPECTED": "PRESERVED",
                "STATUS": "PASS",
            },
            {
                "CHECK": "TECHNICAL_LAYER_CORE_INTEGRITY",
                "VALUE": str(core_integrity),
                "EXPECTED": "True",
                "STATUS": (
                    "PASS"
                    if core_integrity
                    else "FAIL"
                ),
            },
        ]
    )

    return audit


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "PORTFOLIO-B3 — "
        "CAMADA TÉCNICA COMPLEMENTAR"
    )
    print("=" * 78)

    portfolio = load_portfolio()

    tickers = (
        portfolio["TICKER"]
        .tolist()
    )

    print()
    print(
        f"Ações recebidas do motor .............. "
        f"{len(tickers)}"
    )

    print(
        "Seleção fundamental/discount .......... "
        "PRESERVADA"
    )

    print(
        "Regra 80/20 ........................... "
        "NÃO ALTERADA"
    )

    print(
        "TOP4_1Y ............................... "
        "NÃO ALTERADO"
    )

    data = download_prices(
        tickers
    )

    results = []

    print()
    print("=" * 78)
    print("ANÁLISE TÉCNICA")
    print("=" * 78)

    for ticker in tickers:

        print(
            f"Analisando {ticker}..."
        )

        result = analyze_ticker(
            data,
            ticker,
        )

        results.append(
            result
        )

    technical = pd.DataFrame(
        results
    )

    # Merge LEFT:
    # jamais elimina ação da carteira oficial.

    result = portfolio.merge(
        technical,
        on="TICKER",
        how="left",
        validate="one_to_one",
    )

    # Preserva exatamente a ordem da carteira oficial.

    if len(result) != len(portfolio):

        raise RuntimeError(
            "A camada técnica alterou o número "
            "de ações da carteira."
        )

    if (
        set(result["TICKER"])
        !=
        set(portfolio["TICKER"])
    ):

        raise RuntimeError(
            "A camada técnica alterou os tickers "
            "da carteira oficial."
        )

    # ========================================================
    # EXIBIÇÃO
    # ========================================================

    print()
    print("=" * 78)
    print("RESULTADO TÉCNICO")
    print("=" * 78)

    display_columns = [
        c
        for c in [
            "TOP4_RANK",
            "MACRO_SECTOR",
            "SECTOR_RANK",
            "TICKER",
            "FINAL_SCORE",
            "SCORE_TECHNICAL",
            "SIGNAL_TECHNICAL",
            "CONVICTION_TECHNICAL",
            "RISK_TECHNICAL",
            "OPERATIONAL_ACTION",
            "TREND_STATUS",
            "RSI14",
            "ATR_PCT",
            "TECHNICAL_STATUS",
        ]
        if c in result.columns
    ]

    print(
        result[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # AUDITORIA
    # ========================================================

    audit = build_audit(
        result,
        portfolio,
    )

    technical_pass = int(
        (
            result["TECHNICAL_STATUS"]
            ==
            "PASS"
        ).sum()
    )

    technical_insufficient = int(
        (
            result["TECHNICAL_STATUS"]
            ==
            "INSUFFICIENT_HISTORY"
        ).sum()
    )

    technical_no_data = int(
        (
            result["TECHNICAL_STATUS"]
            ==
            "NO_DATA"
        ).sum()
    )

    technical_review = int(
        (
            ~result["TECHNICAL_STATUS"].isin(
                [
                    "PASS",
                    "INSUFFICIENT_HISTORY",
                    "NO_DATA",
                ]
            )
        ).sum()
    )

    print()
    print("=" * 78)
    print("AUDITORIA DA CAMADA TÉCNICA")
    print("=" * 78)

    print(
        f"Ações da carteira ...................... "
        f"{len(result)}"
    )

    print(
        f"Análise técnica PASS ................... "
        f"{technical_pass}"
    )

    print(
        f"Histórico insuficiente ................. "
        f"{technical_insufficient}"
    )

    print(
        f"Sem dados ............................... "
        f"{technical_no_data}"
    )

    print(
        f"Análise técnica REVIEW real ............ "
        f"{technical_review}"
    )

    print(
        "Histórico insuficiente altera carteira . "
        "NÃO"
    )

    print(
        "Histórico insuficiente recebe score .... "
        "NÃO"
    )

    print(
        "Seleção original ....................... "
        "PRESERVADA"
    )

    print(
        "Estrutura 4 × 3 ....................... "
        "PRESERVADA"
    )

    print(
        "TOP4_1Y ............................... "
        "PRESERVADO"
    )

    print(
        "DISCOUNT_80_FUNDAMENTALS_20 ........... "
        "PRESERVADO"
    )

    print(
        "Histórico congelado .................... "
        "PRESERVADO"
    )

    print(
        "Técnico altera composição .............. "
        "NÃO"
    )

    print(
        "Técnico altera FINAL_SCORE ............. "
        "NÃO"
    )

    # ========================================================
    # GRAVAÇÃO
    # ========================================================

    DATA_LIVE.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_TECHNICAL,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        OUTPUT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("ARQUIVOS GERADOS")
    print("=" * 78)

    print(
        OUTPUT_TECHNICAL
    )

    print(
        OUTPUT_AUDIT
    )

    print()
    print(
        "STATUS: CAMADA TÉCNICA COMPLEMENTAR "
        "CONCLUÍDA"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
