"""Compact B/C animation previews from preserved native PNGs; no model inference."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists();a.output.mkdir(parents=True)
 history=a.source_run/'visual_history';bank=json.loads((history/'seed_bank.json').read_text())['samples'];timeline=json.loads((history/'timeline.json').read_text())['frames'];result={'seed_bank_sha256':sha(history/'seed_bank.json'),'script_sha256':sha(Path(__file__)),'movies':[],'scope':'Every sampled frame preserves all8 original prompts/feature/token seeds.32 uniform native keyframes including first/last. PNG masters untouched. GIFs are lossy display derivatives; playback does not represent training wall time.'}
 font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';large=ImageFont.truetype(font,16);small=ImageFont.truetype(font,11)
 for phase,title,prior in [('b','Joint Stage B','Neutral image-feature prior; text-conditioned B probe'),('c','Feature diffusion C','Text-only generated C features; fixed token/noise seeds')]:
  all_frames=sorted([x for x in timeline if x['phase']==phase],key=lambda x:x['step']);indices=[round(i*(len(all_frames)-1)/31) for i in range(32)];frames=[];sources=[]
  for index in indices:
   item=all_frames[index];folder=history/item['directory'];metadata=json.loads((folder/'metadata.json').read_text());assert metadata['phase']==phase and metadata['step']==item['step'];assert len(metadata['prompts'])==8
   frame=Image.new('RGB',(512,328),'#ffffff');draw=ImageDraw.Draw(frame);draw.text((10,8),f"{title} | update {item['step']:,}",font=large,fill='#202124');draw.text((10,33),'Same eight original prompts and feature/token seeds',font=small,fill='#575b60');draw.text((10,51),prior,font=small,fill='#575b60');paths={}
   for j,(expected,actual) in enumerate(zip(bank,metadata['prompts'])):
    assert all(expected[k]==actual[k] for k in ['row_id','caption','seed','feature_seed','token_seed']);image=folder/actual['file'];paths[str(image.relative_to(a.source_run))]=sha(image)
    with Image.open(image) as im:assert im.size==(256,256);frame.paste(im.convert('RGB').resize((128,128),Image.Resampling.NEAREST),(j%4*128,72+j//4*128))
   frames.append(frame);sources.append({'step':item['step'],'PNG_sha256':paths,'metadata_sha256':sha(folder/'metadata.json')})
  atlas=Image.new('RGB',(512*4,328*8),'white')
  for i,frame in enumerate(frames):atlas.paste(frame,((i%4)*512,(i//4)*328))
  palette=atlas.quantize(colors=96);q=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames];durations=[180]*len(frames);durations[0]=620;durations[-1]=2000;dest=a.output/(phase+'.gif');q[0].save(dest,save_all=True,append_images=q[1:],duration=durations,loop=0,optimize=True,disposal=2)
  with Image.open(dest) as im:
   assert im.n_frames==len(frames);duration=0
   for i in range(im.n_frames):im.seek(i);im.load();duration+=im.info['duration']
   im.seek(im.n_frames-1);im.convert('RGB').save(a.output/(phase+'-last.png'))
  assert duration==sum(durations);assert all(sha(a.source_run/path)==digest for row in sources for path,digest in row['PNG_sha256'].items())
  result['movies'].append({'phase':phase,'frames':len(frames),'first_step':sources[0]['step'],'last_step':sources[-1]['step'],'duration_ms':duration,'bytes':dest.stat().st_size,'sha256':sha(dest),'sources':sources,'decode':'pass'})
 result['visual_QA']='pending';(a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='movies'},indent=2));print([(r['phase'],r['bytes']) for r in result['movies']])
if __name__=='__main__':main()
