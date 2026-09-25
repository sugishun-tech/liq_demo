"""Offline tests. Synthetic runtime bytes here are never shipped as real libraries."""
from pathlib import Path
import importlib.metadata
import json
import shutil
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import vendor, cpu_only
from bundle import BundleError, write_json, read_json, describe_file
from check import validate_site

@pytest.fixture
def site(tmp_path):
    p=tmp_path/'docs';shutil.copytree(ROOT/'docs',p)
    return p

def fake_download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if '.wasm' in path.name and '.mjs' not in path.name:
        path.write_bytes(b'\x00asm\x01\x00\x00\x00'+b'UNIT TEST ONLY'*5)
    else:path.write_bytes(b'// UNIT TEST FIXTURE, NOT A REAL LIBRARY.\nexport const test=1;\n')

@pytest.fixture
def offline_download(monkeypatch,tmp_path):
    monkeypatch.setattr(vendor,'ROOT',tmp_path)
    monkeypatch.setattr(vendor,'download',fake_download)

def test_vendor_populates_only_cpu_files(site,offline_download):
    vendor.prepare_vendor(site)
    assert vendor.validate_vendor(site)
    assert {p.relative_to(site).as_posix() for p in (site/'vendor').rglob('*') if p.is_file()}==vendor.EXPECTED
    assert not any('webgpu' in p or 'jsep' in p for p in vendor.EXPECTED)
    assert read_json(site/'runtime-config.json')==vendor.SETTINGS

def test_vendor_second_run_no_download(site,offline_download,monkeypatch):
    vendor.prepare_vendor(site)
    def fail(*a): raise AssertionError('Should use bundled files')
    monkeypatch.setattr(vendor,'download',fail)
    vendor.prepare_vendor(site,offline=True)

@pytest.mark.parametrize('mode',['bytes','extra','missing','symlink'])
def test_vendor_detects_altered_files(site,offline_download,mode):
    vendor.prepare_vendor(site)
    p=site/'vendor/tokenizers/tokenizers.mjs'
    if mode=='bytes':p.write_text('MODIFIED '+p.read_text())
    elif mode=='extra':(site/'vendor/extra.js').write_text('not expected')
    elif mode=='missing':p.unlink()
    else:
        dest=site/'saved.mjs';shutil.copy2(p,dest);p.unlink();p.symlink_to(dest)
    with pytest.raises(BundleError):vendor.validate_vendor(site)

def test_vendor_offline_missing_stops(site):
    with pytest.raises(BundleError,match='初回'):vendor.prepare_vendor(site,offline=True)

def test_download_failure_never_marks_ready(site,monkeypatch,tmp_path):
    monkeypatch.setattr(vendor,'ROOT',tmp_path)
    def fail(*a):raise OSError('simulated network failure')
    monkeypatch.setattr(vendor,'download',fail)
    with pytest.raises(OSError):vendor.prepare_vendor(site)
    assert not (site/'vendor').exists()
    assert read_json(site/'vendor-lock.json')['status']=='missing'

def test_no_gpu_or_framework_archive_fetch():
    assert all(not any(x in u for x in ['webgpu','jsep','transformers','tgz']) for _,u in vendor.SPECS)
    assert len(vendor.SPECS)==6

def test_invalid_wasm_and_html_are_rejected(tmp_path):
    p=tmp_path/'wrong.wasm';p.write_bytes(b'not wasm'*20)
    with pytest.raises(BundleError):vendor.check_payload(p)
    p=tmp_path/'wrong.mjs';p.write_text('<!DOCTYPE html><html>'+('error'*20))
    with pytest.raises(BundleError):vendor.check_payload(p)

def test_root_or_docs_static_copy_without_build(site,tmp_path):
    assert validate_site(site,True)>0
    root=tmp_path/'public-repo';shutil.copytree(site,root)
    assert validate_site(root,True)>0
    assert (root/'.nojekyll').exists() and (root/'index.html').exists()
    assert not (ROOT/'.github').exists()
    assert not (ROOT/'tools/build.py').exists()

def test_check_rejects_training_database(site):
    (site/'training.sqlite3').write_bytes(b'private')
    with pytest.raises(BundleError,match='学習'):validate_site(site,True)

def test_check_rejects_symlink_even_inside_site(site):
    (site/'oops').symlink_to(site/'styles.css')
    with pytest.raises(BundleError,match='symlink'):validate_site(site,True)

def test_check_requires_runtimes_when_model_ready(site,offline_download):
    # Incomplete model is not accepted, regardless of the vendor status.
    write_json(site/'model/manifest.json',{'format':'text-iq-pages-v1','status':'ready'})
    with pytest.raises(BundleError):validate_site(site)

def test_cpu_guard_accepts_cpu_and_rejects_gpu(monkeypatch):
    cpu=SimpleNamespace(version=SimpleNamespace(cuda=None,hip=None),__version__='2.10.0+cpu')
    monkeypatch.setitem(sys.modules,'torch',cpu)
    monkeypatch.setattr(importlib.metadata,'distributions',lambda:[])
    assert cpu_only.require_cpu()=='2.10.0+cpu'
    cpu.version.cuda='12.8'
    with pytest.raises(BundleError):cpu_only.require_cpu()

@pytest.mark.parametrize('name',['nvidia-cublas-cu12','onnxruntime-gpu','triton','cupy-cuda12x','tensorrt'])
def test_cpu_guard_rejects_unwanted_packages(monkeypatch,name):
    cpu=SimpleNamespace(version=SimpleNamespace(cuda=None,hip=None),__version__='2.10.0+cpu')
    monkeypatch.setitem(sys.modules,'torch',cpu)
    monkeypatch.setattr(importlib.metadata,'distributions',lambda:[SimpleNamespace(metadata={'Name':name})])
    with pytest.raises(BundleError):cpu_only.require_cpu()

@pytest.mark.parametrize('mount',['','repo/'])
def test_plain_http_without_build_at_root_and_project_path(site,tmp_path,mount):
    import functools
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from serve import Handler
    serve_root=tmp_path/'web'
    destination=serve_root/mount
    shutil.copytree(site,destination)
    class Quiet(Handler):
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(serve_root)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        base=f'http://127.0.0.1:{server.server_port}/{mount}'
        for file,media in [('index.html','text/html'),('src/app.mjs','text/javascript'),('src/worker.mjs','text/javascript'),('model/manifest.json','application/json'),('runtime-config.json','application/json'),('.nojekyll',None)]:
            with urllib.request.urlopen(base+file,timeout=3) as response:
                assert response.read()==(destination/file).read_bytes()
                if media: assert response.headers.get_content_type()==media
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3)
