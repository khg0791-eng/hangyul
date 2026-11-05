# 일목균형표 보수적 매수 신호 스캐너 (웹 배포용)
# 로컬 실행: streamlit run app.py
# requirements.txt 내용:
#   streamlit
#   yfinance
#   finance-datareader
#   plotly
#   pandas
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

try:
    import FinanceDataReader as fdr
except ImportError:
    fdr = None

st.set_page_config(page_title="일목균형표 스캐너", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- (선택) 비밀번호
# Streamlit Cloud > Settings > Secrets 에  APP_PASSWORD = "원하는비번"  추가하면 활성화
try:
    _pw = st.secrets.get("APP_PASSWORD")
except Exception:
    _pw = None
if _pw:
    if st.session_state.get("auth") != True:
        inp = st.text_input("비밀번호", type="password")
        if inp == _pw:
            st.session_state["auth"] = True
            st.rerun()
        st.stop()

# ---------------------------------------------------------------- 데이터
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, market: str, years: int = 3) -> pd.DataFrame:
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    if market == "한국" and fdr is not None:
        df = fdr.DataReader(ticker, start)
    else:
        if market == "한국":
            ticker = ticker + ".KS"
        df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=86400, show_spinner=False)
def kr_listing() -> pd.DataFrame:
    if fdr is None:
        return pd.DataFrame(columns=["Code", "Name"])
    try:
        return fdr.StockListing("KRX")[["Code", "Name"]]
    except Exception:
        return pd.DataFrame(columns=["Code", "Name"])

def resolve_kr(q: str):
    """종목명 또는 코드 -> (코드, 이름)"""
    q = q.strip()
    if q.isdigit():
        return q.zfill(6), q.zfill(6)
    lst = kr_listing()
    hit = lst[lst["Name"] == q]
    if hit.empty:
        hit = lst[lst["Name"].str.contains(q, case=False, na=False, regex=False)]
    if hit.empty:
        raise ValueError(f"'{q}' 종목을 찾을 수 없습니다. 종목코드로 입력해 보세요.")
    return hit.iloc[0]["Code"], hit.iloc[0]["Name"]

# ---------------------------------------------------------------- 일목균형표
def ichimoku(df, t=9, k=26, s=52, shift=26):
    hi, lo = df["High"], df["Low"]
    df["tenkan"] = (hi.rolling(t).max() + lo.rolling(t).min()) / 2
    df["kijun"] = (hi.rolling(k).max() + lo.rolling(k).min()) / 2
    # 선행스팬은 26일 앞에 그려지므로, '오늘 시점의 구름'은 26일 전에 계산된 값
    df["spanA"] = ((df["tenkan"] + df["kijun"]) / 2).shift(shift)
    df["spanB"] = ((hi.rolling(s).max() + lo.rolling(s).min()) / 2).shift(shift)
    df["chikou"] = df["Close"].shift(-shift)
    df["cloud_top"] = df[["spanA", "spanB"]].max(axis=1)
    df["cloud_bot"] = df[["spanA", "spanB"]].min(axis=1)
    df["cloud_thick"] = (df["cloud_top"] - df["cloud_bot"]) / df["Close"] * 100
    return df

# ---------------------------------------------------------------- 캔들 패턴
def candle_patterns(df):
    body = (df["Close"] - df["Open"]).abs()
    rng = (df["High"] - df["Low"]).replace(0, float("nan"))
    upper = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower = df[["Open", "Close"]].min(axis=1) - df["Low"]
    bull = df["Close"] > df["Open"]
    prev_bear = df["Close"].shift() < df["Open"].shift()
    df["bull_engulf"] = (bull & prev_bear & (df["Open"] <= df["Close"].shift())
                         & (df["Close"] >= df["Open"].shift()))
    df["hammer"] = (lower > body * 2) & (upper < body * 0.5) & (body / rng > 0.1)
    df["strong_bull"] = bull & (body / rng > 0.6)
    return df

# ---------------------------------------------------------------- 신호
def signals(df, min_thick, cross_window, require_chikou, require_candle):
    df["gc"] = (df["tenkan"] > df["kijun"]) & (df["tenkan"].shift() <= df["kijun"].shift())
    df["gc_recent"] = df["gc"].rolling(cross_window).max().fillna(0).astype(bool)
    bull_cloud = df["spanA"] > df["spanB"]
    above = (df["tenkan"] > df["cloud_top"]) & (df["kijun"] > df["cloud_top"]) & (df["Close"] > df["cloud_top"])
    thick = df["cloud_thick"] >= min_thick
    cond = df["gc"] & bull_cloud & above & thick
    if require_chikou:
        cond &= df["Close"] > df["Close"].shift(26)
    if require_candle:
        pat = (df["bull_engulf"] | df["hammer"] | df["strong_bull"]).rolling(3).max().fillna(0).astype(bool)
        cond &= pat
    df["buy"] = cond
    return df

def checklist(df, min_thick):
    r = df.iloc[-1]
    return {
        "전환선 > 기준선": bool(r.tenkan > r.kijun),
        "최근 골든크로스 발생": bool(r.gc_recent),
        "양운 (선행A > 선행B)": bool(r.spanA > r.spanB),
        f"구름 두께 ≥ {min_thick}%": bool(r.cloud_thick >= min_thick),
        "종가가 구름 위": bool(r.Close > r.cloud_top),
        "전환·기준선 모두 구름 위": bool(r.tenkan > r.cloud_top and r.kijun > r.cloud_top),
        "후행스팬 > 26일 전 종가": bool(r.Close > df["Close"].iloc[-27]) if len(df) > 27 else False,
    }

def analyze(ticker, market, years, p):
    df = load_data(ticker, market, years)
    df = ichimoku(df.copy())
    df = candle_patterns(df)
    return signals(df, p["min_thick"], p["cross_window"], p["require_chikou"], p["require_candle"])

# ---------------------------------------------------------------- 차트
def make_chart(df, name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.8, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low,
                                 close=df.Close, name="캔들",
                                 increasing_line_color="#e04848", decreasing_line_color="#3a6fd8"), 1, 1)
    for up, color in [(True, "rgba(224,72,72,0.22)"), (False, "rgba(58,111,216,0.22)")]:
        a = df["spanA"].where((df.spanA > df.spanB) == up)
        b = df["spanB"].where((df.spanA > df.spanB) == up)
        fig.add_trace(go.Scatter(x=df.index, y=b, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=a, fill="tonexty", fillcolor=color,
                                 line=dict(width=0.5, color=color), name="양운" if up else "음운"), 1, 1)
    for col, c, n in [("tenkan", "#ff9800", "전환선"), ("kijun", "#1565c0", "기준선"),
                      ("chikou", "#2e7d32", "후행스팬")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], line=dict(color=c, width=1.3), name=n), 1, 1)
    sig = df[df.buy]
    fig.add_trace(go.Scatter(x=sig.index, y=sig.Low * 0.97, mode="markers",
                             marker=dict(symbol="triangle-up", size=14, color="gold",
                                         line=dict(width=1, color="black")), name="매수신호"), 1, 1)
    fig.add_trace(go.Bar(x=df.index, y=df.Volume, marker_color="gray", name="거래량"), 2, 1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, title=name,
                      legend=dict(orientation="h"), margin=dict(t=50, b=10))
    return fig

# ---------------------------------------------------------------- 사이드바
PRESETS = {
    "한국": "삼성전자\nSK하이닉스\nNAVER\n현대차\n셀트리온\nKB금융\n기아\nPOSCO홀딩스",
    "미국": "AAPL\nMSFT\nNVDA\nGOOGL\nAMZN\nMETA\nTSLA\nAVGO",
}

st.title("📈 일목균형표 보수적 매수 신호 스캐너")
st.caption("전략: 전환선이 기준선을 상향 돌파(골든크로스) + 두꺼운 양운 위에서 발생 시 매수")

with st.sidebar:
    market = st.radio("시장", ["한국", "미국"], horizontal=True)
    years = st.slider("조회 기간(년)", 1, 5, 2)
    st.subheader("보수적 필터")
    min_thick = st.slider("최소 구름 두께(% of 종가)", 0.0, 20.0, 5.0, 0.5)
    cross_window = st.slider("'최근' 골든크로스 기준(일)", 1, 20, 5)
    require_chikou = st.checkbox("후행스팬 확인 (종가 > 26일 전 종가)", True)
    require_candle = st.checkbox("캔들 확인 (장대양봉/장악형/망치형)", False)
    st.caption("※ 참고용 도구이며 투자 판단과 책임은 본인에게 있습니다.")

params = dict(min_thick=min_thick, cross_window=cross_window,
              require_chikou=require_chikou, require_candle=require_candle)

tab1, tab2 = st.tabs(["🔍 종목 분석", "📋 관심종목 스캐너"])

# ---------------------------------------------------------------- 탭1: 개별 분석
with tab1:
    q = st.text_input("종목명 또는 코드" if market == "한국" else "티커",
                      "삼성전자" if market == "한국" else "AAPL", key=f"q_{market}")
    try:
        if market == "한국":
            code, name = resolve_kr(q)
            label = f"{name}({code})"
        else:
            code = q.strip().upper()
            label = code
        with st.spinner("데이터 불러오는 중..."):
            df = analyze(code, market, years, params)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

    last = df.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("현재가", f"{last.Close:,.2f}")
    c2.metric("구름 두께", f"{last.cloud_thick:.1f}%")
    c3.metric("구름 상태", "양운" if last.spanA > last.spanB else "음운")
    c4.metric("매수신호", "🟢 발생" if df["buy"].tail(cross_window).any() else "⚪ 없음")
    c5.metric("기준선(손절 참고)", f"{last.kijun:,.2f}",
              f"{(last.Close / last.kijun - 1) * 100:.1f}%")

    st.plotly_chart(make_chart(df.tail(250), f"{label} 일목균형표"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("현재 체크리스트")
        cl = checklist(df, min_thick)
        for k, v in cl.items():
            st.write(("✅ " if v else "❌ ") + k)
        st.progress(sum(cl.values()) / len(cl), text=f"{sum(cl.values())}/{len(cl)} 충족")
    with right:
        st.subheader("과거 매수신호 & 이후 수익률")
        sig = df[df.buy].copy()
        if sig.empty:
            st.info("조건을 만족하는 신호가 없습니다. 구름 두께 기준을 낮추거나 조회 기간을 늘려보세요.")
        else:
            for n in (20, 60):
                sig[f"+{n}일(%)"] = [
                    (df.Close.iloc[df.index.get_loc(i) + n] / df.Close.loc[i] - 1) * 100
                    if df.index.get_loc(i) + n < len(df) else None for i in sig.index]
            sig["구름두께(%)"] = sig["cloud_thick"]
            st.dataframe(sig[["Close", "구름두께(%)", "+20일(%)", "+60일(%)"]].round(2))

# ---------------------------------------------------------------- 탭2: 스캐너
with tab2:
    st.write("관심종목을 한 줄에 하나씩 입력하면, 내 기준으로 한 번에 점검합니다. "
             "(한국: 종목명/코드, 미국: 티커)")
    wl = st.text_area("관심종목", PRESETS[market], height=200, key=f"wl_{market}")
    only_signal = st.checkbox("매수신호 발생 종목만 보기", False)

    if st.button("스캔 시작", type="primary"):
        items = [x.strip() for x in wl.splitlines() if x.strip()]
        rows, fails = [], []
        bar = st.progress(0.0)
        for i, item in enumerate(items):
            try:
                if market == "한국":
                    code, name = resolve_kr(item)
                else:
                    code, name = item.upper(), item.upper()
                d = analyze(code, market, max(years, 2), params)
                r = d.iloc[-1]
                cl = checklist(d, min_thick)
                rows.append({
                    "종목": name, "코드": code,
                    "신호": "🟢 발생" if d["buy"].tail(cross_window).any() else "⚪",
                    "충족": sum(cl.values()),
                    "현재가": round(float(r.Close), 2),
                    "구름두께(%)": round(float(r.cloud_thick), 1),
                    "양운": "O" if r.spanA > r.spanB else "X",
                    "구름 위": "O" if r.Close > r.cloud_top else "X",
                    "기준선 이격(%)": round(float((r.Close / r.kijun - 1) * 100), 1),
                })
            except Exception as e:
                fails.append(f"{item}: {e}")
            bar.progress((i + 1) / len(items))
        bar.empty()
        if rows:
            res = pd.DataFrame(rows).sort_values(["신호", "충족"], ascending=[False, False])
            if only_signal:
                res = res[res["신호"] == "🟢 발생"]
            st.dataframe(res, hide_index=True, use_container_width=True)
            st.caption("충족 = 체크리스트 7개 중 충족 개수 / 기준선 이격이 작을수록 손절 기준이 가까워 보수적")
        if fails:
            with st.expander(f"불러오기 실패 {len(fails)}건"):
                st.write("\n".join(fails))
