import json, random, tarfile, tempfile
from io import StringIO
from pathlib import Path
import pandas as pd
import requests

SEED=42
N_VIST=2520
N_POEMSUM=1680
OUT_CSV='train_set.csv'
OUT_JSONL='train_set.jsonl'
VIST_URL='https://visionandlanguage.net/VIST/json_files/story-in-sequence/SIS-with-labels.tar.gz'
HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/151.0 Safari/537.36','Referer':'https://visionandlanguage.net/VIST/dataset.html','Accept':'*/*'}
POEMSUM_BASE='https://raw.githubusercontent.com/Ridwan230/PoemSum/main/Dataset/'
POEMSUM_FILES=['poemsum_train.csv','poemsum_valid.csv','poemsum_test.csv']

def clean(x): return '' if x is None else str(x).strip()
def norm(x): return ''.join(c.lower() if c.isalnum() else '_' for c in str(x)).strip('_')
def find_value(row,names):
    d={norm(k):k for k in row}
    for n in names:
        if norm(n) in d: return row[d[norm(n)]]
    for nk,orig in d.items():
        for n in names:
            k=norm(n)
            if k in nk or nk in k: return row[orig]
    return None

def download_vist(path):
    try:
        with requests.get(VIST_URL,headers=HEADERS,stream=True,timeout=120) as r:
            r.raise_for_status()
            with path.open('wb') as f:
                for chunk in r.iter_content(1024*1024):
                    if chunk: f.write(chunk)
            return
    except Exception as e:
        print('requests failed:',e,'- trying curl')
    import shutil, subprocess
    curl=shutil.which('curl')
    if not curl: raise RuntimeError('curl is required for VIST download.')
    subprocess.run([curl,'-L','--fail','--retry','4','-A',HEADERS['User-Agent'],'-e',HEADERS['Referer'],'-o',str(path),VIST_URL],check=True)

def extract_records(obj):
    if isinstance(obj,dict):
        ann=obj.get('annotations')
        if isinstance(ann,list):
            out=[]
            for group in ann:
                if isinstance(group,list):
                    items=[x for x in group if isinstance(x,dict)]
                    if items: out.append(items)
                elif isinstance(group,dict): out.append(group)
            return out
        for key in ('data','stories','story','records'):
            v=obj.get(key)
            if isinstance(v,list): return [x for x in v if isinstance(x,dict)]
    if isinstance(obj,list):
        out=[]
        for group in obj:
            if isinstance(group,list):
                items=[x for x in group if isinstance(x,dict)]
                if items: out.append(items)
            elif isinstance(group,dict): out.append(group)
        return out
    return []

def load_vist(n):
    with tempfile.TemporaryDirectory(prefix='train_vist_') as td:
        td=Path(td); archive=td/'vist.tar.gz'; download_vist(archive)
        ext=td/'vist'; ext.mkdir()
        with tarfile.open(archive,'r:gz') as tar: tar.extractall(ext,filter='data')
        out=[]; seen=set()
        for path in ext.rglob('*.json'):
            try: obj=json.loads(path.read_text(encoding='utf-8'))
            except Exception: continue
            for group in extract_records(obj):
                if not isinstance(group,list): continue
                rows=list(group)
                def order(r):
                    try: return int(find_value(r,['image_order','sentence_order','order']))
                    except Exception: return 999
                rows.sort(key=order)
                s=[]
                for row in rows:
                    t=clean(find_value(row,['storytext','story_text','sentence','caption','text']))
                    if t: s.append(t)
                content=' '.join(s[:5]).strip()
                if not content or content in seen: continue
                seen.add(content)
                sid=clean(find_value(rows[0],['story_id','sequence_id','album_id','storylet_id']))
                out.append({'id':sid or f'vist_{len(out):06d}','name_of_work':f'VIST Story {sid or len(out)+1}','content':content,'source':'vist'})
                if len(out)>=n: return out
        return out

def find_col(df,names):
    exact={str(c).strip().lower():c for c in df.columns}
    for n in names:
        if n.lower() in exact: return exact[n.lower()]
    for c in df.columns:
        x=str(c).strip().lower()
        for n in names:
            if n.lower() in x: return c
    return None

def load_poemsum(n):
    out=[]; seen=set()
    for fn in POEMSUM_FILES:
        print('Downloading:',fn)
        r=requests.get(POEMSUM_BASE+fn,timeout=120); r.raise_for_status()
        df=pd.read_csv(StringIO(r.text))
        text_col=find_col(df,['poem','poem_text','text','content','poetry','document'])
        title_col=find_col(df,['title','poem_title','name_of_work','name'])
        if text_col is None: raise RuntimeError(f'No poem text column in {fn}: {list(df.columns)}')
        for i,row in df.iterrows():
            if pd.isna(row[text_col]): continue
            content=str(row[text_col]).strip()
            if not content or content in seen: continue
            seen.add(content)
            title=str(row[title_col]).strip() if title_col is not None and pd.notna(row[title_col]) else f'Poem {len(out)+1}'
            out.append({'id':f'poem_sum:{fn}:{i}','name_of_work':title,'content':content,'source':'poem_sum'})
            if len(out)>=n: return out
    return out

def main():
    print(f'VIST: {N_VIST:,} | PoemSum: {N_POEMSUM:,} | Total: {N_VIST+N_POEMSUM:,}')
    vist=load_vist(N_VIST); poems=load_poemsum(N_POEMSUM)
    if len(vist)!=N_VIST: raise RuntimeError(f'Expected {N_VIST} VIST stories, got {len(vist)}')
    if len(poems)!=N_POEMSUM: raise RuntimeError(f'Expected {N_POEMSUM} PoemSum poems, got {len(poems)}')
    data=vist+poems; random.Random(SEED).shuffle(data)
    rows=[{'index':i,'id':x['id'],'name_of_work':x['name_of_work'],'content':x['content'],'source':x['source']} for i,x in enumerate(data)]
    pd.DataFrame(rows).to_csv(OUT_CSV,index=False)
    with open(OUT_JSONL,'w',encoding='utf-8') as f:
        for x in rows: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print('Saved',OUT_CSV,'and',OUT_JSONL)
    print(pd.DataFrame(rows)['source'].value_counts())

if __name__=='__main__': main()
