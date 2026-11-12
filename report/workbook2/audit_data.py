from pathlib import Path
import json,zipfile,hashlib,csv,collections,argparse
import numpy as np
parser=argparse.ArgumentParser(description='Audit raw and frozen LongBench exports without loading model weights.')
parser.add_argument('--raw-archive',type=Path,required=True)
parser.add_argument('--full',type=Path,required=True)
parser.add_argument('--dev',type=Path,required=True)
ARGS=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;REPO=ROOT
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:[json.loads(l) for l in Path(p).open() if l.strip()]
full=load(ARGS.full);dev=load(ARGS.dev)
assert len(full)==3731 and len(dev)==800
raw=[]
with zipfile.ZipFile(ARGS.raw_archive) as z:
 for task in sorted({r['task'] for r in full}):
  names=[n for n in z.namelist() if n.split('/')[-1]==task+'.jsonl'];assert len(names)==1
  for l in z.read(names[0]).decode().splitlines():
   if l.strip():r=json.loads(l);r['task']=task;raw.append(r)
assert len(raw)==3750
rawbyid={r['_id']:r for r in raw};fullbyid={r['record_id']:r for r in full}
assert set(fullbyid)<=set(rawbyid)
ids={r['record_id'] for r in dev};repoids=set(json.loads((REPO/'Data_pipeline/processed_longbench/dev_ids.json').read_text())['record_ids']);assert ids==repoids
calids=set(json.loads((ROOT/'members/vedant/results/sem1/adaptive/split_manifest.json').read_text())['record_ids'])
cal=[r for r in dev if r['record_id'] in calids];held=[r for r in dev if r['record_id'] not in calids]
hashcontext=lambda r:hashlib.sha256(r['context'].strip().encode()).hexdigest()
def shared_docs(a,b):
 ha={hashcontext(r) for r in a};hb={hashcontext(r) for r in b};over=ha&hb
 return {'unique_contexts_shared':len(over),'left_records_with_shared_context':sum(hashcontext(r) in over for r in a),'right_records_with_shared_context':sum(hashcontext(r) in over for r in b)}
stat={}
for task in sorted({r['task'] for r in full}):
 f=[r for r in full if r['task']==task];v=[r for r in dev if r['task']==task]
 stat[task]={'raw':sum(r['task']==task for r in raw),'clean':len(f),'dev':len(v),'truncated_full':sum(r['truncated'] for r in f),'truncated_dev':sum(r['truncated'] for r in v),'median_tokens':float(np.median([r['prompt_tokens_after'] for r in f]))}
# Genuine same-ID raw/cleaned/prepared excerpts, with short context quotations.
samples=[]
for task in ['qasper','trec','lcc']:
 r=next(r for r in held if r['task']==task);original=rawbyid[r['record_id']]
 samples.append({'id':r['record_id'],'task':task,'raw_keys':list(original),'raw_input':original.get('input',''),'raw_context_excerpt':' '.join(original['context'].split()[:14]),'raw_answers':original['answers'],'clean_input':r['input'],'clean_answers':r['answers'],'transformed_keys':list(r),'tokens_before':r['prompt_tokens_before'],'tokens_after':r['prompt_tokens_after'],'truncated':r['truncated'],'prompt_prefix':' '.join(r['prompt'].split()[:18]),'split':r['split']})
# One actual normalization example when source and prepared strings differ.
changed=[]
for r in full:
 orig=rawbyid[r['record_id']]
 for field in ['context','input']:
  if orig.get(field,'')!=r[field]:
   changed.append({'id':r['record_id'],'task':r['task'],'field':field,'raw_prefix':repr(orig.get(field,'')[:100]),'clean_prefix':repr(r[field][:100]),'raw_suffix':repr(orig.get(field,'')[-70:]),'clean_suffix':repr(r[field][-70:])});break
 if changed:break
tri=list(csv.DictReader((REPO/'members/samruddhi/results_out/per_record.csv').open()))
tri_tasks=sorted({r['task'] for r in tri});triids={r['record_id'] for r in tri if r['press']=='none'}
result={'raw_records':len(raw),'clean_records':len(full),'dev_records':len(dev),'full_columns':len(full[0]),'raw_columns':len(raw[0])-1,'dev_sha256':sha(ARGS.dev),'full_sha256':sha(ARGS.full),'raw_zip_sha256':sha(ARGS.raw_archive),'duplicate_ids_raw':len(raw)-len(rawbyid),'duplicate_clean_pairs':len(full)-len({(r['task'],hashcontext(r),r['input']) for r in full}),'removed_ids':sorted(set(rawbyid)-set(fullbyid)),'task_stats':stat,'dev_tokens_mean':float(np.mean([r['prompt_tokens_after'] for r in dev])),'dev_tokens_median':float(np.median([r['prompt_tokens_after'] for r in dev])),'full_tokens_max':max(r['prompt_tokens_after'] for r in full),'raw_required_context_missing':sum(not r.get('context','').strip() for r in raw),'adaptive_overlap':shared_docs(cal,held),'dev_vs_remaining_overlap':shared_docs(dev,[r for r in full if r['record_id'] not in ids]),'samples':samples,'normalization_example':changed,'tri_tasks':tri_tasks,'tri_unique_records':len(triids),'tri_actual_max_prompt_tokens':max(int(r['prompt_tokens']) for r in tri),'metadata_schema':list(full[0]),'dev_truncated':sum(r['truncated'] for r in dev),'full_truncated':sum(r['truncated'] for r in full)}
(OUT/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['samples','removed_ids','task_stats','normalization_example']},indent=2))
