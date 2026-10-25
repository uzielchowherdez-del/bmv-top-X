
import os, yfinance as yf, pandas as pd, numpy as np, requests, datetime
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import warnings; warnings.filterwarnings('ignore')

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# --- CONFIGURACIÓN DE RIESGO MACRO ---
# Calendario Banxico 2025-2026 (jueves 13:00 hrs) - actualiza cada año
DIAS_BANXICO = [
    "2026-02-05","2026-03-26","2026-05-14","2026-06-25","2026-08-06","2026-09-24","2026-11-05","2026-12-17",
    "2025-02-06","2025-03-27","2025-05-15","2025-06-26","2025-08-07","2025-09-25","2025-11-06","2025-12-18"
]

NEGATIVAS = ["demanda","fraude","pérdida","perdida","baja","recorte","caída","caida","investigación","multa","quiebra","huelga","demora","falla","escándalo","lawsuit","fraud","loss","downgrade","cut","miss","probe"]
POSITIVAS = ["récord","record","alza","sube","ganancia","crecimiento","expansión","contrato","aprobación","incremento","utilidad","beat","upgrade","growth","gain"]

def get_macro_state():
    """Analiza USD/MXN y mercado para filtrar días malos"""
    estado = {"riesgo":"BAJO", "usdmxn_chg":0, "mxx_chg":0, "es_banxico":False, "mensaje":""}
    try:
        # USD/MXN
        usd = yf.Ticker("USDMXN=X").history(period="5d")
        if len(usd)>=2:
            usd_chg = (usd['Close'].iloc[-1] - usd['Close'].iloc[-2]) / usd['Close'].iloc[-2] * 100
            estado["usdmxn_chg"] = round(float(usd_chg),2)
        
        # IPC México ^MXX
        try:
            mxx = yf.Ticker("^MXX").history(period="5d")
            if len(mxx)>=2:
                mxx_chg = (mxx['Close'].iloc[-1] - mxx['Close'].iloc[-2]) / mxx['Close'].iloc[-2] * 100
                estado["mxx_chg"] = round(float(mxx_chg),2)
        except:
            estado["mxx_chg"] = 0

        # ¿Hoy es día de Banxico?
        hoy = datetime.date.today().isoformat()
        if hoy in DIAS_BANXICO:
            estado["es_banxico"] = True
            estado["mensaje"] += "📢 HOY ANUNCIO BANXICO 13:00 hrs. "

        # Lógica de riesgo
        if estado["usdmxn_chg"] >= 1.2:
            estado["riesgo"] = "EXTREMO"
            estado["mensaje"] += f"💥 USD/MXN +{estado['usdmxn_chg']}% disparo. "
        elif estado["usdmxn_chg"] >= 0.7:
            estado["riesgo"] = "ALTO"
            estado["mensaje"] += f"⚠️ USD/MXN +{estado['usdmxn_chg']}% sube fuerte. "
        elif estado["usdmxn_chg"] <= -0.7:
            estado["riesgo"] = "FAVORABLE"
            estado["mensaje"] += f"✅ USD/MXN {estado['usdmxn_chg']}% dólar débil, favorece BMV. "
        
        if estado["mxx_chg"] <= -1.0:
            estado["riesgo"] = "ALTO" if estado["riesgo"]!="EXTREMO" else "EXTREMO"
            estado["mensaje"] += f"📉 IPC {estado['mxx_chg']}% ayer mal. "

        if not estado["mensaje"]:
            estado["mensaje"] = f"Día normal. USD/MXN {estado['usdmxn_chg']}% | IPC {estado['mxx_chg']}%"

    except Exception as e:
        estado["mensaje"] = f"Macro no disponible: {e}"
    
    return estado

def analizar_noticia(ticker):
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news and ".MX" in ticker:
            news = yf.Ticker(ticker.replace(".MX","")).news
        if not news:
            return 0, ""
        titulos=[]
        score=0
        for n in news[:2]:
            title = n.get('title','') if isinstance(n, dict) else ""
            if not title and isinstance(n, dict):
                title = n.get('content',{}).get('title','') if isinstance(n.get('content'), dict) else ""
            if not title: continue
            titulos.append(title)
            tl=title.lower()
            score += sum(1 for w in POSITIVAS if w in tl) - sum(1 for w in NEGATIVAS if w in tl)
        sentimiento = 0
        if score <= -2: sentimiento=-1
        elif score==-1: sentimiento=-0.5
        elif score>=2: sentimiento=1
        elif score==1: sentimiento=0.5
        return sentimiento, " | ".join(titulos)[:150]
    except:
        return 0, ""

tickers = ["AAPL.MX","AMXB.MX","WALMEX.MX","GFNORTEO.MX","GMEXICOB.MX","CEMEXCPO.MX","GAPB.MX","VOLARA.MX","BIMBOA.MX","AC.MX","GCARSOA1.MX","PINFRA.MX","TLEVISACPO.MX","TSLA.MX","NVDA.MX","MSFT.MX","META.MX","AMZN.MX","GOOGL.MX","GRUMAB.MX","KIMBERA.MX"]

def rsi(s,p=14):
    delta=s.diff(); gain=delta.where(delta>0,0).rolling(p).mean(); loss=-delta.where(delta<0,0).rolling(p).mean(); rs=gain/loss; return 100-(100/(1+rs))

all_data={}
for t in tickers:
    try:
        df=yf.Ticker(t).history(period="2y", auto_adjust=True)
        if len(df)>100: all_data[t]=df
    except: pass

rows=[]
for t,df in all_data.items():
    d=df.copy()
    d['Ret_1d']=d['Close'].pct_change(1); d['Ret_5d']=d['Close'].pct_change(5)
    d['Gap']=(d['Open']-d['Close'].shift(1))/d['Close'].shift(1)
    d['Vol_MA20']=d['Volume'].rolling(20).mean(); d['Vol_Ratio']=d['Volume']/d['Vol_MA20']
    d['Dist_MA20']=(d['Close']-d['Close'].rolling(20).mean())/d['Close'].rolling(20).mean()
    d['RSI']=rsi(d['Close']); d['Target']=(d['High']>=d['Open']*1.012).astype(int)
    d.dropna(inplace=True); d['Ticker']=t; rows.append(d)

master=pd.concat(rows)
features=['Ret_1d','Ret_5d','Gap','Vol_Ratio','Dist_MA20','RSI']
X=master[features]; y=master['Target']
base_model=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=0.03,min_child_weight=10,reg_alpha=1,reg_lambda=2,subsample=0.7,colsample_bytree=0.7,random_state=42,eval_metric='logloss')
calibrated=CalibratedClassifierCV(base_model, method='sigmoid', cv=TimeSeriesSplit(3))
calibrated.fit(X,y)
proba_raw=calibrated.predict_proba(X)[:,1]
p10,p90=np.percentile(proba_raw,[10,90])

macro = get_macro_state()

resultados=[]
for t,df in all_data.items():
    d=df.copy()
    d['Ret_1d']=d['Close'].pct_change(1); d['Ret_5d']=d['Close'].pct_change(5)
    d['Gap']=(d['Open']-d['Close'].shift(1))/d['Close'].shift(1)
    d['Vol_MA20']=d['Volume'].rolling(20).mean(); d['Vol_Ratio']=d['Volume']/d['Vol_MA20']
    d['Dist_MA20']=(d['Close']-d['Close'].rolling(20).mean())/d['Close'].rolling(20).mean()
    d['RSI']=rsi(d['Close']); d=d.dropna()
    if d.empty: continue
    last=d.iloc[-1]
    raw=calibrated.predict_proba([[last['Ret_1d'],last['Ret_5d'],last['Gap'],last['Vol_Ratio'],last['Dist_MA20'],last['RSI']]])[0,1]
    prob=52 + (np.clip(raw,p10,p90)-p10)/(p90-p10+1e-9)*16
    sent, noticia = analizar_noticia(t)
    
    comprable=True; motivo="OK"; prob_aj=prob
    if last['RSI']>75: comprable=False; motivo=f"RSI {last['RSI']:.0f}"
    elif sent==-1: comprable=False; motivo=f"Noticia negativa"; prob_aj-=10
    elif sent==-0.5: prob_aj-=5
    elif sent==1: prob_aj+=3
    
    # Ajuste macro
    if macro["riesgo"]=="EXTREMO":
        comprable=False; motivo=f"Macro EXTREMO: {macro['mensaje']}"
        prob_aj-=15
    elif macro["riesgo"]=="ALTO":
        prob_aj-=7
    elif macro["riesgo"]=="FAVORABLE":
        prob_aj+=4
    
    if comprable:
        resultados.append({'Ticker':t,'Prob':prob,'Prob_Ajustada':max(45,min(75,prob_aj)),'Close':last['Close'],'RSI':last['RSI'],'Noticia':noticia,'Motivo':motivo,'Sent':sent})

resultados=sorted(resultados,key=lambda x: x['Prob_Ajustada'],reverse=True)

# MENSAJE TELEGRAM
hoy_str = pd.Timestamp.now().date()
msg = f"📊 *TOP BMV HOY {hoy_str} v7 MACRO*\n"
msg += f"Tasa base: {master['Target'].mean()*100:.1f}%\n"
msg += f"Macro: {macro['riesgo']} | {macro['mensaje']}\n"
msg += f"USD/MXN: {macro['usdmxn_chg']}% | IPC: {macro['mxx_chg']}%\n\n"

if macro["riesgo"]=="EXTREMO":
    msg += f"🚫 *HOY NO OPERES - DÍA DE ALTO RIESGO*\n{macro['mensaje']}\n\nProtege capital. Ahorras comisiones.\n"
    # Aun así muestra top pero bloqueado
    for r in resultados[:3]:
        msg += f"❌ {r['Ticker']} {r['Prob_Ajustada']:.1f}% BLOQUEADO - {r['Motivo']}\n"
else:
    top_validos = [r for r in resultados if r['Prob_Ajustada']>=58][:5]
    if top_validos:
        for r in top_validos:
            msg += f"*{r['Ticker']}* {r['Prob_Ajustada']:.1f}% (base {r['Prob']:.1f}%) | ${r['Close']:.2f} | Venta ${r['Close']*1.012:.2f} | Stop ${r['Close']*0.992:.2f}\n"
            if r['Sent']!=0 and r['Noticia']:
                msg += f"  📰 {r['Noticia'][:90]}\n"
            msg += "\n"
        if macro["es_banxico"]:
            msg += "⚠️ *Hoy Banxico 13:00* - Cierra todo antes de las 12:50 si vas ganando. Hey no tiene stop auto.\n"
    else:
        msg += "Hoy sin oportunidades >58% después de filtro macro. No operes.\n"

if macro["riesgo"]!="EXTREMO":
    msg += "\n⚠️ Hey sin stop auto. Alerta TradingView en Stop."

if TELEGRAM_TOKEN and CHAT_ID:
    url=f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id":CHAT_ID,"text":msg,"parse_mode":"Markdown"})
print(msg)
