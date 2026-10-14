#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();rows=list(csv.DictReader((a.directory/'summary.csv').open()))
    for suite in ('foreground-cover','no-foreground','stacked'):
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        specs=[('median_seconds','Algorithm time (seconds)',1),('materialized_result_boundary_vertices','Materialized Boolean-result boundary vertices',1),('layer1_triangles','Layer 1 exported triangles',1),('median_seconds','All-layer algorithm time (seconds)','all')]
        for ax,(field,label,k) in zip(axes.flat,specs):
            for method,order in [('visible-peel','near-first'),('visible-peel','random'),('visible-peel','reverse'),('envelope-peel','input'),('arrangement','input')]:
                rs=sorted([r for r in rows if r['suite']==suite and r['layers']==str(k) and r['method']==method and r['order']==order],key=lambda r:int(r['size']))
                if not rs:continue
                ax.plot([int(r['size']) for r in rs],[float(r[field]) for r in rs],marker='o',label=method+'/'+order)
                if field=='median_seconds':ax.fill_between([int(r['size']) for r in rs],[float(r['min_seconds']) for r in rs],[float(r['max_seconds']) for r in rs],alpha=.12)
            ax.set(xlabel='g' if suite!='stacked' else 'source count',ylabel=label,title=f'K={k}');ax.set_xscale('log',base=2);ax.grid(alpha=.25)
            if field=='median_seconds':ax.set_yscale('log')
        axes[0,0].legend(fontsize=7);fig.suptitle(suite+' — successful measurements only; see status.jsonl for caps/timeouts')
        for ext in ('png','svg','pdf'):fig.savefig(a.directory/f'{suite}.{ext}',dpi=160)
        plt.close(fig)
if __name__=='__main__':main()
