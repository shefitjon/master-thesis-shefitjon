import os, json, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts')
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'engineered_sessions_no_leakage.csv')
LANG = 'tr' if '--tr' in sys.argv else 'en'
T = {'h1': ('1.  THE HONEST RESULT  —  recomputed live, right now', '1.  DÜRÜST SONUÇ  —  şu anda canlı yeniden hesaplandı'), 'held': ('Held-out sessions (never trained on)', 'Ayrılmış oturumlar (hiç eğitilmedi)'), 'f1': ('F1-score (Logistic Regression)', 'F1 skoru (Lojistik Regresyon)'), 'auc': ('ROC-AUC', 'ROC-AUC'), 'def': ('This is the number I defend. The leaky pipeline reported 0.997.', 'Savunduğum sayı bu. Sızıntılı hat 0,997 bildirmişti.'), 'h2': ('2.  BROWSE BEHAVIOR, NOT PRICE  —  two real sessions, opposite verdicts', '2.  FİYAT DEĞİL, GEZİNME DAVRANIŞI  —  iki gerçek oturum, zıt kararlar'), 'aband': ('ABANDONER', 'TERK EDEN'), 'purch': ('PURCHASER', 'SATIN ALAN'), 'row': ('held-out session row', 'ayrılmış oturum satırı'), 'views': ('  views before cart', '  sepet öncesi görüntüleme'), 'evts': ('  total events before cart', '  sepet öncesi toplam olay'), 'bint': ('  browse intensity', '  gezinme yoğunluğu'), 'cval': ('  cart value', '  sepet değeri'), 'pp': ('  MODEL P(purchase)', '  MODEL P(satın alma)'), 'verd': ('The verdict tracks the browsing pattern — not the price tag.', 'Karar, fiyat etiketini değil, gezinme örüntüsünü izliyor.'), 'h3': ('3.  THE INTERVENTION  —  a real Gemini message + judge scores', '3.  MÜDAHALE  —  gerçek bir Gemini mesajı + yargıç puanları'), 'scval': ('Session cart value', 'Oturum sepet değeri'), 'spp': ('Model P(purchase)', 'Model P(satın alma)'), 'hr': ('high-risk', 'yüksek riskli'), 'jcl': ('Judge avg — clarity', 'Yargıç ort. — açıklık'), 'jre': ('Judge avg — relevance', 'Yargıç ort. — ilgililik'), 'jur': ('Judge avg — urgency (weakest)', 'Yargıç ort. — aciliyet (en zayıf)'), 'jpe': ('Judge avg — personalization', 'Yargıç ort. — kişiselleştirme'), 'jov': ('Overall (150 sessions)', 'Genel (150 oturum)'), 'loop': ('Predict -> Explain -> Generate -> Evaluate. The loop is closed.', 'Tahmin -> Açıkla -> Üret -> Değerlendir. Döngü kapandı.')}

def t(k):
    return T[k][1 if LANG == 'tr' else 0]

class C:
    R = '\x1b[0m'
    B = '\x1b[1m'
    RED = '\x1b[91m'
    GRN = '\x1b[92m'
    YEL = '\x1b[93m'
    CYN = '\x1b[96m'
    GRY = '\x1b[90m'

def head(s):
    print(f'\n{C.B}{C.CYN}{'=' * 70}\n{s}\n{'=' * 70}{C.R}')

def kv(k, v, col=C.GRN):
    print(f'  {k:<34} {col}{C.B}{v}{C.R}')

def main():
    feats = json.load(open(f'{ART}/feature_names.json'))
    lr = joblib.load(f'{ART}/model_lr.pkl')
    idx = json.load(open(f'{ART}/holdout_indices.json'))
    catmap = json.load(open(f'{ART}/category_mapping.json'))
    n2c = {v: int(k) for k, v in catmap.items()}
    df = pd.read_csv(CSV)
    df['main_category'] = df['main_category'].astype(str).map(n2c)
    X = df[feats].values.astype(float)
    y = df['target_purchase'].values
    test = np.array(idx['test_indices'])
    col = {f: i for i, f in enumerate(feats)}
    proba = lr.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    head(t('h1'))
    kv(t('held'), f'{len(test):,}')
    kv(t('f1'), f'{f1_score(y[test], pred[test]):.4f}')
    kv(t('auc'), f'{roc_auc_score(y[test], proba[test]):.4f}')
    print(f'  {C.GRY}{t('def')}{C.R}')
    head(t('h2'))
    purch = test[y[test] == 1][np.argmax(proba[test[y[test] == 1]])]
    aband = test[y[test] == 0][np.argmin(proba[test[y[test] == 0]])]

    def show(i, label, colr):
        r = X[i]
        print(f'\n  {colr}{C.B}{label}{C.R}  ({t('row')} {i})')
        kv(t('views'), int(r[col['views_before_cart']]), colr)
        kv(t('evts'), int(r[col['total_events_before_cart']]), colr)
        kv(t('bint'), f'{r[col['browse_intensity_pre_cart']]:.2f}', colr)
        kv(t('cval'), f'${r[col['initial_cart_value']]:.2f}', colr)
        kv(t('pp'), f'{proba[i] * 100:.2f}%', colr)
    show(aband, t('aband'), C.RED)
    show(purch, t('purch'), C.GRN)
    print(f'\n  {C.YEL}{C.B}{t('verd')}{C.R}')
    head(t('h3'))
    # Judge results from the thesis generator-size ablation: 150 sessions per
    # model, scored by Flash-Latest. Gemini 2.5 Pro is the primary generator,
    # Flash-Lite the smaller-model comparison. Urgency is the weakest dimension;
    # Flash-Lite only "wins" personalization by parroting the cart dollar value.
    print(f'\n  {C.GRN}{C.B}“That is a great choice after looking at 7 options; your electronics order is saved with free 30-day returns.”{C.R}')
    print(f'  {C.GRY}— Gemini 2.5 Pro, grounded in the session LIME reason{C.R}')
    print()
    kv(t('jcl'), '9.83 (Pro)  vs  9.84 (Flash-Lite)')
    kv(t('jre'), '8.25 (Pro)  vs  7.65 (Flash-Lite)')
    kv(t('jur'), '3.31 (Pro)  vs  3.22 (Flash-Lite)', C.YEL)
    kv(t('jpe'), '7.33 (Pro)  vs  8.67 (Flash-Lite)')
    kv(t('jov'), '~ 7.2 / 10')
    print(f'\n  {C.GRY}{t('loop')}{C.R}\n')
    if '--widget-params' in sys.argv:
        clf = lr.named_steps['clf']
        sca = lr.named_steps['scaler']
        params = {'intercept': float(clf.intercept_[0]), 'features': {}}
        for f in ['views_before_cart', 'total_events_before_cart', 'browse_intensity_pre_cart']:
            i = col[f]
            params['features'][f] = {'coef': float(clf.coef_[0][i]), 'mean': float(sca.mean_[i]), 'scale': float(sca.scale_[i])}
        print('WIDGET_PARAMS=' + json.dumps(params))
if __name__ == '__main__':
    main()
