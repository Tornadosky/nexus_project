"""Summarize the remote metadata scan and explicitly reject unjustified run reuse."""
from __future__ import annotations
import argparse, collections, csv, io, json
from pathlib import Path
from common import ROOT, write_json, atomic

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metadata',required=True)
    ap.add_argument('--out',default=str(ROOT/'audit')); a=ap.parse_args()
    data=[json.loads(s) for s in Path(a.metadata).read_text().splitlines()]
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    hashes=collections.Counter(r.get('sha256') for r in data)
    summary={'files':len(data),'bytes':sum(r['bytes'] for r in data),
      'decode_errors':[r['relative_path'] for r in data if r.get('error')],
      'unique_sha256':len(hashes),'duplicate_pairs':sum(n==2 for n in hashes.values()),
      'weight_files':sum(bool(r.get('has_weights')) for r in data),
      'weight_files_with_recorded_normalization':sum(bool(r.get('has_weights') and r.get('normalization_present')) for r in data),
      'metrics_only':[r['relative_path'] for r in data if not r.get('has_weights')],
      'unstable_during_read':[r['relative_path'] for r in data if not r.get('stable_during_read')],
      'exact_core_budget_candidates':[r['relative_path'] for r in data if r.get('actual_steps') in (117964800,32768000)],
      'rgb_name_candidates':[r['relative_path'] for r in data if any(w in r['relative_path'].lower() for w in ('rgb','pixel','camera'))],
      'approved_primary_run_reuse':0,
      'reuse_reason':'No matched core budget/snapshot cohorts; no RGB weights located; legacy LLM specifications and source/environment provenance differ. Budget alone is not a reuse proof.'}
    write_json(out/'checkpoint_summary.json',summary)
    buf=io.StringIO(); fields=['relative_path','bytes','sha256','env_name','seed','requested_steps','actual_steps','has_weights','normalization_present','commit_hash','reuse_class']
    w=csv.DictWriter(buf,fieldnames=fields); w.writeheader()
    for r in data:
        x={k:r.get(k) for k in fields}; x['reuse_class']='legacy_evaluation_candidate' if r.get('has_weights') and r.get('normalization_present') else 'needs_normalizer_provenance' if r.get('has_weights') else 'metrics_only_not_a_checkpoint'
        w.writerow(x)
    atomic(out/'checkpoint_index.csv',buf.getvalue().encode())
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
