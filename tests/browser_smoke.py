"""Browser UI/worker integration tests. Explicit synthetic runtime fixture, NOT real E5/Laya.

The production site has no fixture backend and no inference fallback to these models.
Run: python tests/browser_smoke.py --browser /usr/bin/chromium
"""
from __future__ import annotations
import argparse
import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unicodedata
from playwright.sync_api import sync_playwright, expect
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from bundle import write_json, describe_file, normalize

FAKE_TOKENIZERS = """// SYNTHETIC TEST DOUBLE. Not Tokenizers.js.
export class Tokenizer {
 constructor(){}
 encode(text){return {ids:[0,...Array.from(text,c=>c.codePointAt(0)%10000),2]};}
}
"""
FAKE_ORT = '''// SYNTHETIC TEST DOUBLE. Not ONNX Runtime or E5.
export const env={wasm:{}};
export class Tensor {
 constructor(type,data,dims){this.type=type;this.data=data;this.dims=dims}
 async getData(){return this.data} dispose(){}
}
export const InferenceSession={async create(bytes,opts){
 if(JSON.stringify(opts.executionProviders)!=='["wasm"]') throw new Error('CPU EXPECTED');
 return {inputNames:['input_ids','attention_mask','token_type_ids'],outputNames:['embedding'],
 async run(feeds){await new Promise(r=>setTimeout(r,15));
 const angle=Array.from(feeds.input_ids.data,Number).reduce((a,b)=>a+b,0)*0.001;
 const data=new Float32Array(384);data[0]=Math.cos(angle);data[1]=Math.sin(angle);
 return {embedding:new Tensor('float32',data,[1,384])};},async release(){}}
}};
'''
class Quiet(SimpleHTTPRequestHandler):
    extensions_map={**SimpleHTTPRequestHandler.extensions_map,'.mjs':'text/javascript'}
    def log_message(self,*args):pass

def serve(directory):
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(directory)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server,f'http://127.0.0.1:{server.server_port}'

def fixture(directory):
    site=directory/'repo';shutil.copytree(ROOT/'docs',site)
    (site/'vendor/tokenizers').mkdir(parents=True)
    (site/'vendor/onnxruntime').mkdir(parents=True)
    (site/'vendor/tokenizers/tokenizers.mjs').write_text(FAKE_TOKENIZERS)
    (site/'vendor/onnxruntime/ort.wasm.min.mjs').write_text(FAKE_ORT)
    write_json(site/'vendor-lock.json',{'format':'text-iq-runtime-v2','status':'ready','note':'TEST FIXTURE ONLY'})
    model=site/'model';(model/'encoder').mkdir();(model/'tokenizer').mkdir()
    (model/'encoder/part-000.bin').write_bytes(b'NOT A REAL ONNX MODEL. UI TEST ONLY.')
    write_json(model/'tokenizer/tokenizer.json',{'test_only':True})
    write_json(model/'tokenizer/tokenizer_config.json',{'tokenizer_class':'XLMRobertaTokenizer'})
    cases=[]
    for text in ['An example.','Another text.','日本語の例。','Cafe\u0301','']:
        ids=[0]+[ord(c)%10000 for c in 'query: '+normalize(text)]+[2]
        angle=sum(ids)*.001;v=[math.cos(angle),math.sin(angle)]+[0]*382
        cases.append({'text':text,'input_ids':ids,'embedding':v,'raw_score':50+5*v[0]+2*v[1]})
    write_json(model/'parity.json',{'cases':cases})
    m={'format':'text-iq-pages-v1','schema_version':1,'status':'ready','id':'f'*64,
       'features':{'adapter':'e5-masked-mean-l2-v1','prefix':'query: ','normalization':'nfc-python-whitespace','dimension':384,'max_length':512},
       'ridge':{'coef':[5,2]+[0]*382,'intercept':50,'clip':[0,100],'iq_offset':70,'iq_scale':.6},
       'encoder':{'precision':'fp32','inputs':['input_ids','attention_mask','token_type_ids'],'output':'embedding',
                  'bytes':(model/'encoder/part-000.bin').stat().st_size,'chunks':[describe_file(model/'encoder/part-000.bin',model)]},
       'tokenizer':{'files':[describe_file(f,model) for f in sorted((model/'tokenizer').iterdir())]},
       'parity':{**describe_file(model/'parity.json',model),'python_onnx_passed':True,'embedding_atol':.0002,'score_atol':.02},
       'training':{'warning':'SYNTHETIC BROWSER INTEGRATION TEST ONLY; NOT LAYA/E5.'}}
    write_json(model/'manifest.json',m)
    return site

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--browser',default=shutil.which('chromium'));ap.add_argument('--output',type=Path,default=ROOT/'browser-results');a=ap.parse_args()
    a.output.mkdir(exist_ok=True,parents=True)
    passed=[]
    with tempfile.TemporaryDirectory() as temp, sync_playwright() as p:
        temp=Path(temp);shutil.copytree(ROOT/'docs',temp/'blank');fixture(temp)
        server,base=serve(temp)
        try:
            browser=p.chromium.launch(executable_path=a.browser,headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1440,'height':1200},device_scale_factor=1)
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(base+'/blank/');expect(page.locator('#setup')).to_be_visible()
            expect(page.locator('#score')).to_be_disabled();expect(page.locator('#iq')).to_have_text('—')
            page.screenshot(path=str(a.output/'unconfigured-desktop.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(a.output/'unconfigured-mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            page.locator('a[href="guide.html"]').first.click();expect(page.locator('h1')).to_contain_text('学習済みモデル')
            passed.append('unconfigured page: no fabricated scores, mobile layout, help navigation')
            page.set_viewport_size({'width':1280,'height':1000})
            page.goto(base+'/repo/');expect(page.locator('#load')).to_be_enabled()
            page.click('#load');expect(page.locator('#model-status')).to_contain_text('数値照合済み',timeout=30000)
            passed.append('actual Worker protocol, asset hash checks, runtime initialization and parity with explicit test doubles')
            page.click('#score');expect(page.locator('#iq')).not_to_have_text('—',timeout=10000)
            expected=float(page.locator('#iq').inner_text());assert 70<=expected<=130
            page.click('#explain');expect(page.locator('.heat-token').first).to_be_visible(timeout=10000)
            page.locator('.heat-token').first.focus();expect(page.locator('#heat-detail')).to_contain_text('差分')
            with page.expect_download() as download:page.click('#export')
            report=json.loads(Path(download.value.path()).read_text())
            assert report['model_id']=='f'*64 and report['method'].startswith('one-span')
            passed.append('score, occlusion heatmap, keyboard-accessible details and JSON export')
            page.fill('#text','Café 🧑🏽‍💻は例。');expect(page.locator('#iq')).to_have_text('—');page.click('#score')
            expect(page.locator('#explain')).to_be_enabled();page.select_option('#granularity','grapheme');page.click('#explain')
            expect(page.locator('.heat-token').filter(has_text='🧑🏽‍💻')).to_have_count(1,timeout=10000)
            passed.append('grapheme explanation preserves emoji sequence')
            payload='<img src=x onerror=globalThis.pwned=1> 原因は未確定。'
            page.fill('#text',payload);page.click('#score');expect(page.locator('#explain')).to_be_enabled();page.select_option('#granularity','word');page.click('#explain')
            expect(page.locator('#heatmap')).to_have_text(payload,timeout=10000)
            assert page.locator('#heatmap img').count()==0 and page.evaluate('globalThis.pwned') is None
            passed.append('HTML-like input is rendered only as text (no XSS)')
            page.fill('#text','a'*513);page.click('#score');expect(page.locator('#error')).to_contain_text('上限',timeout=10000);expect(page.locator('#iq')).to_have_text('—')
            passed.append('overlong tokenizer input fails without truncation')
            page.fill('#text','あいうえお'*10);page.click('#score');expect(page.locator('#explain')).to_be_enabled();page.select_option('#granularity','grapheme');page.click('#explain');page.click('#cancel')
            expect(page.locator('#model-status')).to_contain_text('中止');expect(page.locator('#load')).to_be_enabled();expect(page.locator('#score')).to_be_disabled()
            passed.append('cancellation terminates worker and requires reload')
            # A corrupt fixture must fail before a usable model can be declared ready.
            broken=temp/'repo/model/encoder/part-000.bin';broken.write_bytes(b'X'*broken.stat().st_size)
            context=browser.new_context();q=context.new_page();q.goto(base+'/repo/');q.click('#load')
            expect(q.locator('#error')).to_contain_text('チェックサム',timeout=10000);expect(q.locator('#score')).to_be_disabled();context.close()
            passed.append('corrupt model file fails with no fake inference fallback')
            assert not errors,errors
            browser.close()
        finally:server.shutdown();server.server_close()
    write_json(a.output/'ui-tests.json',{'passed':len(passed),'checks':passed,'real_laya_e5_onnx_executed':False,'note':'Inference libraries were explicitly replaced in an isolated fixture site. Production files were not modified.'})
    print(json.dumps({'passed':len(passed),'checks':passed},ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
