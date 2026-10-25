
import os, yfinance as yf, pandas as pd, numpy as np, requests
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import warnings; warnings.filterwarnings('ignore')

# Secrets desde GitHub
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    raise Exception("Faltan TELEGRAM_TOKEN y CHAT_ID en los Secrets de GitHub")

tickers = ["AAPL.MX","AMXB.MX","WALMEX.MX","GFNORTEO.MX","GMEXICOB.MX","CEMEXCPO.MX","GAPB.MX","VOLARA.MX","BIMBOA.MX","AC.MX","GCARSOA1.MX","PINFRA.MX","TLEVISACPO.MX","TSLA.MX","NVDA.MX","MSFT.MX","META.MX","AMZN.MX","GOOGL.MX","GRUMAB.MX","KIMBERA.MX","CUERVO.MX","GFINBURO.MX","BBAJIOO.MX","GAPB.MX"]

def rsi(s,p=14):
    delta=s.diff()
    gain=delta.where(delta>0,0).rolling(p).mean()
    loss=-delta.where(delta<0,0).rolling(p).mean()
    rs=gain/loss
    return 100-(100/(1+rs))

all_data={}
for t in tickers:
    try:
        df=yf.Ticker(t).history(period="2y", auto_adjust=True)
        if len(df)>100:
            all_data[t]=df
    except:
        pass

rows=[]
for t,df in all_data.items():
    d=df.copy()
    d['Ret_1d']=d['Close'].pct_change(1)
    d['Ret_5d']=d['Close'].pct_change(5)
    d['Gap']=(d['Open']-d['Close'].shift(1))/d['Close'].shift(1)
    d['Vol_MA20']=d['Volume'].rolling(20).mean()
    d['Vol_Ratio']=d['Volume']/d['Vol_MA20']
    d['Dist_MA20']=(d['Close']-d['Close'].rolling(20).mean())/d['Close'].rolling(20).mean()
    d['RSI']=rsi(d['Close'])
    d['Target']=(d['High']>=d['Open']*1.012).astype(int)
    d.dropna(inplace=True)
    d['Ticker']=t
    rows.append(d)

if not rows:
    msg="⚠️ Hoy no se pudo descargar BMV. Revisa Yahoo."
else:
    master=pd.concat(rows)
    features=['Ret_1d','Ret_5d','Gap','Vol_Ratio','Dist_MA20','RSI']
    X=master[features]; y=master['Target']
    
    base_model=XGBClassifier(n_estimators=150,max_depth=2,learning_rate=0.03,min_child_weight=10,reg_alpha=1,reg_lambda=2,subsample=0.7,colsample_bytree=0.7,random_state=42,eval_metric='logloss')
    calibrated=CalibratedClassifierCV(base_model, method='sigmoid', cv=TimeSeriesSplit(3))
    calibrated.fit(X,y)
    proba_raw=calibrated.predict_proba(X)[:,1]
    p10,p90=np.percentile(proba_raw,[10,90])
    
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
        comprable = not (last['RSI']>75 or last['RSI']<25 or last['Vol_Ratio']<0.6 or abs(last['Gap'])>0.05)
        if comprable and prob>=58:
            resultados.append((t,prob,last['Close'],last['RSI']))
    
    resultados=sorted(resultados,key=lambda x: x[1],reverse=True)[:5]
    
    if resultados:
        msg=f"📊 *TOP BMV HOY {pd.Timestamp.now().date()}*\nTasa base mercado: {master['Target'].mean()*100:.1f}%\n\n"
        for t,p,c,rsi_val in resultados:
            msg+=f"*{t}* {p:.1f}% | Cierre ${c:.2f} | Venta +1.2% ${c*1.012:.2f} | Stop manual ${c*0.992:.2f} | RSI {rsi_val:.0f}\n"
        msg+="\n⚠️ Hey no tiene stop auto. Pon alerta en TradingView en precio stop."
    else:
        msg=f"📊 Hoy {pd.Timestamp.now().date()} sin oportunidades >58%. No operes. Ahorras comisiones. Tasa base: {master['Target'].mean()*100:.1f}%"

# Enviar a Telegram
url=f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
r=requests.post(url, json={"chat_id":CHAT_ID,"text":msg,"parse_mode":"Markdown"})
print(f"Enviado: {r.status_code} - {msg}")
