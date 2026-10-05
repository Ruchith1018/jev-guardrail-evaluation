"""Shared helpers for the analysis scripts: loads results/raw_results.csv and items.jsonl,
and provides Wilson intervals, AUC and rate helpers. Import with `from analyze import *`."""
import pandas as pd, numpy as np, json, math, sys
pd.set_option("display.width",260,"display.max_columns",50,"display.max_rows",500)
import os
R=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")+os.sep
d=pd.read_csv(R+"raw_results.csv")
items={}
for l in open(R+"items.jsonl",encoding="utf8"):
    j=json.loads(l); items[j["id"]]=j
d["text"]=d.id.map(lambda i: items[i]["text"])
d["meta"]=d.id.map(lambda i: items[i]["meta"])

def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; dd=1+z*z/n; c=(p+z*z/(2*n))/dd; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/dd
    return (100*(c-h),100*(c+h))
def auc(y,s):
    y=np.asarray(y); s=np.asarray(s,float); npos=(y==1).sum(); nneg=(y==0).sum()
    if npos==0 or nneg==0: return np.nan
    r=pd.Series(s).rank().values; return (r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg)
def rate(y,flag):
    y=np.asarray(y); f=np.asarray(flag).astype(bool)
    tp=((y==1)&f).sum(); fn=((y==1)&~f).sum(); fp=((y==0)&f).sum(); tn=((y==0)&~f).sum()
    a=wilson(tp,tp+fn); b=wilson(fp,fp+tn)
    return dict(n=len(y),att=int(tp+fn),safe=int(fp+tn),tp=int(tp),fp=int(fp),catch=100*tp/(tp+fn) if tp+fn else np.nan,c_lo=a[0],c_hi=a[1],
                fb=100*fp/(fp+tn) if fp+tn else np.nan,f_lo=b[0],f_hi=b[1])
def line(name,g,col="jev",th=0.5):
    g=g[g[col].notna()]
    if g.empty: return None
    r=rate(g.label.values,(g[col].values>=th) if col!="regex" else (g[col].values>=1)); r["name"]=name; r["auc"]=auc(g.label.values,g[col].values); return r
def show(rows,cols=("name","n","att","safe","catch","c_lo","c_hi","fb","f_lo","f_hi","auc")):
    t = pd.DataFrame([r for r in rows if r])[list(cols)]
    t[[c for c in cols if c not in ("name", "auc")]] = t[[c for c in cols if c not in ("name", "auc")]].round(1)
    if "auc" in t: t["auc"] = t["auc"].round(3)
    print(t.to_string(index=False))
