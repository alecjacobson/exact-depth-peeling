#!/usr/bin/env python3
"""Full labelled-set oracle agreement, all orders, prefixes, and export invariants."""
import argparse
import json
import subprocess
import tempfile
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,default=ROOT/'build/geometric_peel');p.add_argument('--report',type=Path,default=ROOT/'results/validation.json');p.add_argument('--quick',action='store_true');a=p.parse_args();exe=str(a.exe.resolve())
    manifest=json.loads((ROOT/'tests/fixtures/manifest.json').read_text());cases=[];runs=0
    def invoke(args,accept=True):
        nonlocal runs
        runs+=1;r=subprocess.run([exe]+list(map(str,args)),capture_output=True,text=True,timeout=90)
        if accept and r.returncode:raise RuntimeError(' '.join(map(str,args))+'\n'+r.stderr)
        if not accept:assert r.returncode and 'positive-area depth tie' in r.stderr,r.stderr
        return r
    with tempfile.TemporaryDirectory(prefix='exact-peel-validation-') as tmp:
        tmp=Path(tmp)
        for fixture in manifest['fixtures']:
            name=fixture['name'];path=ROOT/'tests/fixtures'/f'{name}.json'
            if fixture['expected'].startswith('reject'):
                for method in ('arrangement','envelope-peel','visible-peel'):invoke(['--scene',path,'--method',method,'--out',tmp/'reject'],False)
                cases.append({'fixture':name,'status':'expected_rejection'});continue
            oracle=tmp/'oracle';invoke(['--scene',path,'--method','arrangement','--layers','all','--verify','--out',oracle]);oj=json.loads((oracle/'exact.json').read_text())
            variants=[('envelope-peel','input',True)]+[('visible-peel',o,True) for o in ('input','random','near-first','reverse')]+[('visible-peel',o,False) for o in ('input','random','near-first')]
            for method,order,accel in variants:
                dest=tmp/'candidate';base=['--scene',path,'--method',method,'--order',order,'--out',dest,'--verify']+([] if accel else ['--no-accelerate'])
                invoke(base+['--layers','all']);invoke(['--compare',dest/'exact.json',oracle/'exact.json'])
                all_json=json.loads((dest/'exact.json').read_text())
                for k in ([1] if a.quick else [1,2,4]):
                    invoke(base+['--layers',k]);prefix=dict(all_json,layers=all_json['layers'][:k]);(tmp/'prefix.json').write_text(json.dumps(prefix));invoke(['--compare',dest/'exact.json',tmp/'prefix.json'])
                if name=='cover_2' and method=='visible-peel' and order=='near-first' and accel:
                    mj=json.loads((dest/'metrics.json').read_text());first=mj['passes'][0];assert first['output']['boundary_vertices']==3 and first['exported_triangles']==1
                    assert first['polygon_intersections']==0 and first['complete_occlusion_rejections']==8
            if name=='holes':assert json.loads((oracle/'metrics.json').read_text())['passes'][0]['output']['holes']>=2
            if name=='components':assert json.loads((oracle/'metrics.json').read_text())['passes'][0]['output']['components']>3
            if name=='cyclic':
                for front,back in ((0,1),(1,2),(2,0)):
                    cells=[c for c in oj['oracle_cells'] if front in c['depth_order'] and back in c['depth_order']]
                    assert cells and all(c['depth_order'].index(front)<c['depth_order'].index(back) for c in cells)
                    # Certify strict separation on the closure of the entire overlap.
                    src={s['source_id']:list(map(F,s['score'])) for s in oj['sources']}
                    d=[src[back][i]-src[front][i] for i in range(3)]
                    for cell in cells:
                        for v in cell['triangle']:
                            x,y=map(F,v);assert d[0]*x+d[1]*y+d[2]>0
                assert not any(len(c['depth_order'])==3 for c in oj['oracle_cells'])
            cases.append({'fixture':name,'status':'passed','all_layers':len(oj['layers'])});print(f'PASS {name}',flush=True)
    report={'status':'passed','subprocess_checks':runs,'fixtures':cases,'requested_prefixes':[1] if a.quick else [1,2,4],'orders':['input','random','near-first','reverse'],'unaccelerated_orders':['input','random','near-first']}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print(f'{len(cases)} fixtures, {runs} checks passed')
if __name__=='__main__':main()
