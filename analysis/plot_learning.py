"""Render the recorded stage trajectories; no fitting, inference or selection."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR','/tmp/mtp-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
if __name__=='__main__':
 data=json.loads((ROOT/'results/learning-history.json').read_text());plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#aaaaaa','svg.fonttype':'none'})
 fig,axes=plt.subplots(2,3,figsize=(13.5,6.6),gridspec_kw={'height_ratios':[2,1]})
 for j,p,title,color in [(0,'b','(a) Joint B: validation CE','#28618c'),(1,'c','(b) C: fixed-noise MSE','#725b84'),(2,'finetune','(c) B fine-tuning: validation CE','#b66138')]:
  d=data[p];xs,ys=zip(*d['validation']);selected=d['selected_step'];value=dict(d['validation'])[selected];axes[0,j].plot(xs,ys,color=color,lw=1.4);axes[0,j].scatter(selected,value,marker='*',s=80,color=color,zorder=3);axes[0,j].set_title(title,loc='left',pad=12);axes[0,j].text(.96,.95,f'Selected {selected:,}\n{value:.4f}',transform=axes[0,j].transAxes,ha='right',va='top',color=color,fontsize=9)
  ranks=d['rank_steps'];axes[1,j].step([r[0] for r in ranks],[r[1] for r in ranks],where='post',color=color,lw=1.6);axes[1,j].set_ylim(0,12);axes[1,j].set_yticks([2,4,6,8,10]);axes[1,j].set_ylabel('Maximum rank');axes[1,j].set_xlabel('Optimizer updates')
  for row in [0,1]:
   ax=axes[row,j];ax.axvline(d['native_final_step'],color='#777777',ls='--',lw=.8);ax.set_xlim(0,d['native_final_step']*1.015);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v/1000:g}k'));ax.grid(alpha=.22)
 fig.suptitle('Recorded learning dynamics and adaptive rank',x=.075,y=.975,ha='left',fontsize=17,weight='bold');fig.text(.075,.89,'Training seed17; fixed validation corruption; observed phase ends without a verified terminal plateau.',fontsize=9,color='#575b60');fig.text(.075,.035,'Stars: selected checkpoints. Dashed lines: recorded phase ends. Objectives and feature targets differ between stages; loss scales are interpreted within a stage.',fontsize=8.3,color='#575b60');fig.subplots_adjust(left=.075,right=.98,top=.78,bottom=.13,hspace=.39,wspace=.34)
 for fmt in ['png','svg']:fig.savefig(ROOT/f'figures/learning-and-rank.{fmt}',dpi=240,metadata={'Date':None} if fmt=='svg' else None)
 plt.close(fig)
