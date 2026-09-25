"""Offline DOM checks only. No server navigation, real Worker, tokenizer or ONNX.

Use when a managed browser prohibits URL navigation. This does not modify browser
policy. HTML/CSS/JS source is inserted into about:blank, with imports joined locally
and network/Worker/download replaced by explicit test doubles. Full HTTP/Worker
integration is separately provided by browser_smoke.py, and is NOT claimed here.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
from playwright.sync_api import sync_playwright, expect
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('http_browser_fixture', ROOT / 'tests/browser_smoke.py')
fixture_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture_module)

MOCK = r'''
// Explicit DOM-test doubles: not a real model, Worker or download.
const TEST_MANIFEST = __MANIFEST__;
globalThis.fetch = async (url) => ({ok: true, json: async () => String(url).includes('vendor-lock.json') ? {status:'ready'} : TEST_MANIFEST});
const values = text => {
 const clean = assertInput(text, true);
 const tokens = Array.from(clean).length + 8;
 if(tokens > 512) throw new Error('入力が上限512トークンを超えました。');
 const v = new Array(384).fill(0), angle = Array.from(clean).reduce((s,c)=>s+c.codePointAt(0),0)*.001;
 v[0]=Math.cos(angle);v[1]=Math.sin(angle);
 return {text:clean, tokens, score:scoreVector(v, TEST_MANIFEST.ridge), elapsed_ms:12};
};
globalThis.Worker = class ExplicitDOMTestDouble {
 stopped = false;
 terminate(){this.stopped=true}
 postMessage(data){
  const send=(type,payload)=> {if(!this.stopped)this.onmessage?.({data:{id:data.id,type,...payload}})};
  setTimeout(async()=>{try{
   if(data.type==='init') send('ready',{backend:'wasm',model_id:TEST_MANIFEST.id,verification:{passed:true,cases:8}});
   else if(data.type==='score')send('score',values(data.text));
   else if(data.type==='explain'){
    const p=occlusionPlan(data.text,data.mode), b=values(p.text), spans=[];
    for(let i=0;i<p.spans.length;i++){
     await new Promise(r=>setTimeout(r,3)); if(this.stopped)return;
     const x=values(p.variants[i]); spans.push({...p.spans[i],delta:attribution(b.score,x.score,.6),without_iq:x.score.iq,without_raw_iq:x.score.raw_iq,variant:p.variants[i],empty_variant:!p.variants[i]});
    }
    send('explanation',{...b,spans,mode:data.mode,method:'one-span DOM-test fixture only'});
   }
  }catch(e){send('error',{message:e.message})}},25);
 }
};
globalThis.__downloaded = null;
const blobMap = new Map();
URL.createObjectURL = blob => {blobMap.set('blob:DOM-TEST',blob);return 'blob:DOM-TEST'};
URL.revokeObjectURL = url => blobMap.delete(url);
HTMLAnchorElement.prototype.click = function(){if(this.download){blobMap.get(this.href)?.text().then(t=>globalThis.__downloaded=JSON.parse(t))}};
'''

def show(page, manifest):
    html = (ROOT / 'docs/index.html').read_text()
    html = re.sub(r'<meta http-equiv="Content-Security-Policy"[^>]*>', '', html)
    html = re.sub(r'<link[^>]*>', '', html)
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.S)
    page.set_content(html)
    page.add_style_tag(content=(ROOT / 'docs/styles.css').read_text())
    core = (ROOT / 'docs/src/core.mjs').read_text().replace('export ', '')
    app = (ROOT / 'docs/src/app.mjs').read_text()
    app = re.sub(r'^import .*?;\n', '', app, count=1)
    app = app.replace('import.meta.url', json.dumps('https://unit-test.invalid/repo/src/app.mjs'))
    code = '(function(){\n' + core + '\n' + MOCK.replace('__MANIFEST__', json.dumps(manifest)) + '\n' + app + '\n})();'
    page.add_script_tag(content=code)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--browser', default=shutil.which('chromium'))
    ap.add_argument('--output', type=Path, default=ROOT/'browser-results')
    args=ap.parse_args(); args.output.mkdir(exist_ok=True, parents=True)
    passed=[]
    with tempfile.TemporaryDirectory() as temp, sync_playwright() as p:
        folder=fixture_module.fixture(Path(temp))
        manifest=json.loads((folder/'model/manifest.json').read_text())
        browser=p.chromium.launch(executable_path=args.browser,headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1200},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        show(page, {'status':'missing'})
        expect(page.locator('#setup')).to_be_visible()
        expect(page.locator('#score')).to_be_disabled(); expect(page.locator('#iq')).to_have_text('—')
        page.screenshot(path=str(args.output/'unconfigured-desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(args.output/'unconfigured-mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        expect(page.locator('a[href="guide.html"]').first).to_be_visible()
        passed.append('unconfigured UI has no fabricated score; desktop/mobile layout and guide link')
        page.close();page=browser.new_page(viewport={'width':1280,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
        show(page,manifest);expect(page.locator('#load')).to_be_enabled();page.click('#load')
        expect(page.locator('#model-status')).to_contain_text('数値照合済み')
        page.click('#score'); expect(page.locator('#iq')).not_to_have_text('—')
        page.click('#explain');expect(page.locator('.heat-token').first).to_be_visible()
        page.locator('.heat-token').first.focus();expect(page.locator('#heat-detail')).to_contain_text('差分')
        page.click('#export');page.wait_for_function('globalThis.__downloaded !== null')
        report=page.evaluate('globalThis.__downloaded');assert report['model_id']=='f'*64
        passed.append('DOM event flow: load/score/heatmap/keyboard details/JSON serialization (test-double Worker)')
        page.fill('#text','Café 🧑🏽‍💻は例。');expect(page.locator('#iq')).to_have_text('—');page.click('#score')
        expect(page.locator('#explain')).to_be_enabled();page.select_option('#granularity','grapheme');page.click('#explain')
        expect(page.locator('.heat-token').filter(has_text='🧑🏽‍💻')).to_have_count(1)
        passed.append('grapheme heatmap preserves complete emoji sequence')
        payload='<img src=x onerror=globalThis.pwned=1> 原因は未確定。'
        page.fill('#text',payload);page.click('#score');expect(page.locator('#explain')).to_be_enabled();page.select_option('#granularity','word');page.click('#explain')
        expect(page.locator('#heatmap')).to_have_text(payload)
        assert page.locator('#heatmap img').count()==0 and page.evaluate('globalThis.pwned') is None
        passed.append('user markup rendered as text, without inserting image or running handlers')
        page.fill('#text','a'*513);page.click('#score');expect(page.locator('#error')).to_contain_text('上限');expect(page.locator('#iq')).to_have_text('—')
        passed.append('error display leaves no stale score')
        page.fill('#text','あいうえお'*10);page.click('#score');expect(page.locator('#explain')).to_be_enabled();page.select_option('#granularity','grapheme');page.click('#explain');page.click('#cancel')
        expect(page.locator('#model-status')).to_contain_text('中止');expect(page.locator('#load')).to_be_enabled();expect(page.locator('#score')).to_be_disabled()
        passed.append('cancel resets model controls and ignores stale test-double responses')
        assert not errors,errors
        browser.close()
    report={'passed':len(passed),'checks':passed,'real_http_worker_onnx_executed':False,
     'note':'DOM-only: source joined locally in about:blank; network/Worker/download are explicit doubles. Browser policy was not changed. Real HTTP/Worker test is separate.'}
    (args.output/'dom-tests.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
