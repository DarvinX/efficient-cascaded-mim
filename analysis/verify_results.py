"""Replay deterministic validation metrics from saved features on CPU only."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import pairwise_distances
from metrics import frechet_from_features
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def prdc(real,generated,k=5):
 rr=pairwise_distances(real,n_jobs=1);gg=pairwise_distances(generated,n_jobs=1);rg=pairwise_distances(real,generated,n_jobs=1);radius_real=np.partition(rr,k,axis=1)[:,k];radius_gen=np.partition(gg,k,axis=1)[:,k];inside=rg<radius_real[:,None]
 return {'precision':float(inside.any(0).mean()),'recall':float((rg<radius_gen[None,:]).any(1).mean()),'density':float(inside.sum(0).mean()/k),'coverage':float((rg.min(1)<radius_real).mean())}
if __name__=='__main__':
 d=json.loads((ROOT/'analysis/inputs/dense.json').read_text())['runs'][0];g=json.loads((ROOT/'analysis/inputs/grown.json').read_text());results={};identity=None
 for folder,expected in [('dense-17',d),('grown-17',g)]:
  base=ROOT/'evidence'/folder;primary=expected.get('primary_metrics',expected)
  with np.load(base/'inception-features-val-g1.0.npz') as z:real,generated,files=z['real'],z['generated'],z['files']
  assert real.shape==generated.shape==(2048,2048);fingerprint=hashlib.sha256(real.tobytes()+files.tobytes()).hexdigest()
  if identity is None:identity=fingerprint
  assert identity==fingerprint
  fid=frechet_from_features(real,generated);assert abs(fid-primary['clean_fid'])<1e-5
  diagnostic=prdc(real,generated);assert all(abs(v-expected['PRDC']['5'][k])<1e-12 for k,v in diagnostic.items())
  with np.load(base/'clip-embeddings-val-g1.0.npz') as z:images,texts,ids=z['images'],z['texts'],z['row_ids']
  cosine=float(np.einsum('ij,ij->i',images,texts).mean(dtype=np.float64));assert abs(cosine-primary['clip_cosine_mean'])<1e-6
  rows=json.loads((base/'samples-val-g1.0/prompts.json').read_text());assert [r['row_id'] for r in rows]==ids.tolist();captions=np.array([r['caption'] for r in rows],dtype=object);top=np.argsort(-(images@texts.T),axis=1,kind='stable')[:,:10];matches=captions[top]==captions[:,None];retrieval={f'R@{k}':float(matches[:,:k].any(1).mean()) for k in [1,5,10]};assert all(abs(v-expected['caption_retrieval'][k])<1e-12 for k,v in retrieval.items())
  quality=json.loads((base/'quality-val-g1.0.json').read_text());sample_shas={r['file']:r['sha256'] for r in quality['manifest']};ref_shas={r['file']:r['sha256'] for r in quality['reference_manifest']}
  for category,expected_shas in [('samples-val-g1.0',sample_shas),('references-val',ref_shas)]:
   for p in (base/category).glob('*.png'):assert sha(p)==expected_shas[p.name]
  results[folder]={'FID':fid,'CLIP':cosine,'PRDC_k5':diagnostic,'caption_retrieval':retrieval,'displayed_PNGs':'SHA256 match native evaluation','status':'pass'};print(folder,'deterministic metric replay passed',flush=True)
 out={'protocol':'curated-release-cpu-metric-replay-v1','status':'pass','results':results,'scope':'Saved validation features and displayed PNGs only. KID remains the native recorded subsample estimate; original subset draws were not serialized. No GPU, test evaluation or checkpoint selection.'};(ROOT/'results/verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
