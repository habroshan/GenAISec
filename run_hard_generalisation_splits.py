                      
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
import numpy as np
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from train_phase3_model_comparison_leakage_safe import (
    TEXT_COL, LABEL_COL, SESSION_COL, TURN_COL, NUMERIC_FEATURES,
    load_json, enrich_missing_numeric_features, numeric_matrix, labels,
    build_cumulative_text, build_turn_feature_transformers, make_turn_features,
    make_sequence_pack, RNNClassifier, train_torch_model, set_seed, ensure_dir
)

def metric(y,p,t):
    yp=(p>=t).astype(int)
    return {"accuracy":float(accuracy_score(y,yp)),"precision":float(precision_score(y,yp,zero_division=0)),
            "recall":float(recall_score(y,yp,zero_division=0)),"f1":float(f1_score(y,yp,zero_division=0)),
            "roc_auc":float(roc_auc_score(y,p)) if len(set(y.tolist()))>1 else None,"threshold":float(t),"n":int(len(y))}

def best_t(y,p):
    ts=np.linspace(.05,.95,181)
    return float(ts[int(np.argmax([f1_score(y,(p>=t).astype(int),zero_division=0) for t in ts]))])

def sessions(rows):
    d=defaultdict(list)
    for r in rows: d[int(r[SESSION_COL])].append(r)
    for k in d: d[k]=sorted(d[k],key=lambda x:int(x[TURN_COL]))
    return d

def flat(d,ids):
    out=[]
    for sid in ids: out.extend(d[sid])
    return sorted(out,key=lambda r:(int(r[SESSION_COL]),int(r[TURN_COL])))

def sess_label(rs): return max(int(r[LABEL_COL]) for r in rs)

def train_val_ids(d, ids, seed):
    rng=np.random.default_rng(seed)
    pos=[i for i in ids if sess_label(d[i])==1]; neg=[i for i in ids if sess_label(d[i])==0]
    rng.shuffle(pos); rng.shuffle(neg)
    def split(x):
        n=max(1,int(round(.15*len(x)))) if len(x)>1 else 0
        return x[n:],x[:n]
    ptr,pv=split(pos); ntr,nv=split(neg)
    return set(ptr+ntr),set(pv+nv)

def lr_eval(tr,va,te):
    yt,yv,ye=labels(tr),labels(va),labels(te)
    v=TfidfVectorizer(max_features=20000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
    Xt=v.fit_transform(build_cumulative_text(tr)); Xv=v.transform(build_cumulative_text(va)); Xe=v.transform(build_cumulative_text(te))
    s=StandardScaler()
    Nt=s.fit_transform(numeric_matrix(tr)); Nv=s.transform(numeric_matrix(va)); Ne=s.transform(numeric_matrix(te))
    Xt,Xv,Xe=hstack([Xt,Nt]),hstack([Xv,Nv]),hstack([Xe,Ne])
    c=LogisticRegression(max_iter=2000,class_weight="balanced",solver="liblinear",random_state=42).fit(Xt,yt)
    pv=c.predict_proba(Xv)[:,1]; t=best_t(yv,pv)
    pe=c.predict_proba(Xe)[:,1]
    return metric(ye,pe,t)

def gru_eval(tr,va,te,args,name):
    v,s=build_turn_feature_transformers(tr,max_features=args.tfidf_features)
    trf, vaf, tef=make_turn_features(tr,v,s),make_turn_features(va,v,s),make_turn_features(te,v,s)
    tp,vp,ep=make_sequence_pack(tr,trf,args.max_len),make_sequence_pack(va,vaf,args.max_len),make_sequence_pack(te,tef,args.max_len)
    m=RNNClassifier(tp.X.shape[-1],args.hidden_dim,"gru",False,.25)
    out=train_torch_model(name,m,tp,vp,ep,args.output_dir,args.epochs,args.batch_size,args.lr,args.device)
    return {k:out[k] for k in ["accuracy","precision","recall","f1","roc_auc","threshold","num_test_samples"] if k in out}

def get_types(rs): return {str(r.get("session_type","")) for r in rs if r.get("session_type") is not None}
def get_roles(rs): return {str(r.get("role","")) for r in rs if r.get("role") is not None}
def get_entities(rs):
    z=set()
    for r in rs:
        for e in r.get("referenced_entities",[]) if isinstance(r.get("referenced_entities",[]),list) else []:
            if e.get("type") in {"vendor","project","system"} and e.get("value") is not None:
                z.add((e.get("type"),str(e.get("value"))))
    return z

def holdout(d,getter,seed,target=.2):
    rng=np.random.default_rng(seed); mp=defaultdict(set)
    for sid,rs in d.items():
        for g in getter(rs):
            if g!="": mp[g].add(sid)
    gs=list(mp); rng.shuffle(gs)
    held=set(); test=set(); targetn=max(1,int(round(target*len(d))))
    for g in gs:
        held.add(g); test|=mp[g]
        if len(test)>=targetn: break
    train={sid for sid,rs in d.items() if sid not in test and set(getter(rs)).isdisjoint(held)}
    test={sid for sid in test if not set(getter(d[sid])).isdisjoint(held)}
    return train,test,held

def evaluate(name,d,trids,teids,args):
    trids,vids=train_val_ids(d,trids,args.seed)
    tr,va,te=flat(d,trids),flat(d,vids),flat(d,teids)
    print(f"\n{name}: train={len(trids)} val={len(vids)} test={len(teids)} sessions")
    return {"train_sessions":len(trids),"val_sessions":len(vids),"test_sessions":len(teids),
            "cumulative_lr_numeric":lr_eval(tr,va,te),"gru":gru_eval(tr,va,te,args,f"gru_{name}")}

def cross_profile(train_rows,test_rows,args,name):
    d=sessions(train_rows); trids,vids=train_val_ids(d,set(d),args.seed)
    tr,va=flat(d,trids),flat(d,vids)
    return {"train_sessions":len(trids),"val_sessions":len(vids),"test_sessions":len(sessions(test_rows)),
            "cumulative_lr_numeric":lr_eval(tr,va,test_rows),"gru":gru_eval(tr,va,test_rows,args,f"gru_{name}")}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--balanced",default="data/profiles/balanced/prompt_sessions_labeled.json")
    p.add_argument("--realistic",default="data/profiles/realistic_mixture//prompt_sessions_labeled.json")
    p.add_argument("--risk_enriched",default="data/profiles/risk_enriched/prompt_sessions_labeled.json")
    p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch_size",type=int,default=128); p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--hidden_dim",type=int,default=96); p.add_argument("--max_len",type=int,default=14)
    p.add_argument("--tfidf_features",type=int,default=1000)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir",default="data/hard_generalisation")
    a=p.parse_args(); set_seed(a.seed); ensure_dir(a.output_dir)
    b=enrich_missing_numeric_features(load_json(a.balanced)); b=sorted(b,key=lambda r:(int(r[SESSION_COL]),int(r[TURN_COL]))); bd=sessions(b)
    res={"numeric_features":NUMERIC_FEATURES,"seed":a.seed,"conditions":{}}
    tr,te,g=holdout(bd,get_types,a.seed+1); res["conditions"]["unseen_session_type"]=evaluate("unseen_session_type",bd,tr,te,a); res["conditions"]["unseen_session_type"]["heldout_groups"]=sorted(g)
    tr,te,g=holdout(bd,get_roles,a.seed+2); res["conditions"]["unseen_role"]=evaluate("unseen_role",bd,tr,te,a); res["conditions"]["unseen_role"]["heldout_groups"]=sorted(g)
    tr,te,g=holdout(bd,get_entities,a.seed+3); res["conditions"]["unseen_entities"]=evaluate("unseen_entities",bd,tr,te,a); res["conditions"]["unseen_entities"]["heldout_groups"]=[{"type":x[0],"value":x[1]} for x in sorted(g)]
    r=enrich_missing_numeric_features(load_json(a.realistic)); r=sorted(r,key=lambda x:(int(x[SESSION_COL]),int(x[TURN_COL])))
    q=enrich_missing_numeric_features(load_json(a.risk_enriched)); q=sorted(q,key=lambda x:(int(x[SESSION_COL]),int(x[TURN_COL])))
    res["conditions"]["balanced_to_realistic"]=cross_profile(b,r,a,"balanced_to_realistic")
    res["conditions"]["balanced_to_risk_enriched"]=cross_profile(b,q,a,"balanced_to_risk_enriched")
    out=os.path.join(a.output_dir,"hard_generalisation_results.json")
    with open(out,"w",encoding="utf-8") as f: json.dump(res,f,indent=2)
    print(f"\nSaved: {out}")
    for n,x in res["conditions"].items():
        print(n,"LR",x["cumulative_lr_numeric"]["f1"],x["cumulative_lr_numeric"]["roc_auc"],"GRU",x["gru"]["f1"],x["gru"]["roc_auc"])

if __name__=="__main__":
    main()
