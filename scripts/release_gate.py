"""Research-evidence checklist; intentional FAIL until real evidence is supplied."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--record',default='configs/research_evidence.json');a=p.parse_args();c=json.loads(Path(a.record).read_text());required=['original_development_dataset_identified','original_split_and_gallery_probe_available','original_checkpoints_available','original_per_sample_predictions_available','original_five_seed_runs_available','matched_modern_baseline_executed','official_protocol_evaluator_verified','explanation_source_outputs_available','original_latency_hardware_and_raw_timings_available']
missing=[k for k in required if c.get(k) is not True];print(json.dumps({'research_reproduction_ready':not missing,'unresolved':missing,'note':'Checklist only; true declarations must be supported by independently inspectable artifacts.'},indent=2));raise SystemExit(2 if missing else 0)
