from __future__ import annotations
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from bundle import (BundleError, digest, fingerprint, load_student, normalize, publish_directory,
                    split_file, write_json, public_metadata, target_is_replaceable)
from check import validate_site, main as check_main
from export_web import make_wrapper, resolve_encoder, check_versions
from vendor import check_payload

@pytest.fixture
def student(tmp_path):
    root=tmp_path/'student'; root.mkdir()
    np.savez_compressed(root/'ridge.npz',coef=np.arange(384,dtype=float)/384,intercept=np.array(50.))
    m={'format':'liq-linear-v3','dimension':384,'weights':'ridge.npz','weights_sha256':digest(root/'ridge.npz'),
       'embedding_spec':{'adapter':'e5-masked-mean-l2-v1','model':{'kind':'hub','repo':'intfloat/multilingual-e5-small','revision':'a'*40,'subfolder':''},
          'prefix':'query: ','max_length':512,'dimension':384,'normalize':True,'precision':'float32-cpu',
          'versions':{'torch':'2.10.0+cpu','transformers':'4.57.6','numpy':'2.3.5'}},
       'clip':[0.,100.],'iq_proxy_transform':{'offset':70.,'scale':.6},
       'teacher_spec':{'adapter':'UNIT TEST FIXTURE ONLY'}, 'counts':{'samples':256},
       'dataset':{'path':'/private/user/data','text':'PRIVATE TRAINING ROW'}, 'metrics':{}}
    write_json(root/'model.json',m)
    return root,m

def test_read_student(student):
    root,m=student;got,w,b=load_student(root)
    assert got==m and w.shape==(384,) and b==50.

def test_missing_student(tmp_path):
    with pytest.raises(BundleError,match='学習済み'): load_student(tmp_path/'missing')

@pytest.mark.parametrize('change',[
    lambda m:m.update(format='old'),lambda m:m.update(dimension=768),
    lambda m:m.update(weights='../other.npz'),lambda m:m.update(weights_sha256='b'*64),
    lambda m:m.update(clip=[0,200]),lambda m:m.update(iq_proxy_transform={'offset':0,'scale':1}),
    lambda m:m['embedding_spec'].update(prefix='passage: '),
    lambda m:m['embedding_spec'].update(normalize=False),
    lambda m:m['embedding_spec'].update(precision='int8'),
    lambda m:m['embedding_spec'].update(max_length=2048),
    lambda m:m['embedding_spec']['model'].update(revision='main'),
    lambda m:m['embedding_spec']['model'].update(repo='other/model'),
    lambda m:m.update(teacher_spec={'adapter':'TEST-DOUBLE-NOT-LAYA'}),
])
def test_reject_bad_source(student,change):
    root,m=student;change(m);write_json(root/'model.json',m)
    with pytest.raises((BundleError,ValueError)):load_student(root)

@pytest.mark.parametrize('coef,bias',[(np.zeros(383),np.array(0.)),(np.full(384,np.nan),np.array(0.)),(np.zeros(384),np.array([0.,1.])),(np.zeros(384),np.array(np.inf))])
def test_reject_invalid_numeric_weights(student,coef,bias):
    root,m=student;np.savez(root/'ridge.npz',coef=coef,intercept=bias);m['weights_sha256']=digest(root/'ridge.npz');write_json(root/'model.json',m)
    with pytest.raises(BundleError):load_student(root)

def test_python_js_normalization_agrees():
    values=['  日本語\tです\u0085。 ','e\u0301 \u001c x','\ufefftest\ufeff','\u3000','カ\u3099','🧑🏽‍💻\u2028hello']
    code="import {normalizeText} from './docs/src/core.mjs'; console.log(JSON.stringify(JSON.parse(process.argv[1]).map(normalizeText)));"
    result=subprocess.run(['node','--input-type=module','-e',code,json.dumps(values)],cwd=ROOT,check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)==[normalize(x) for x in values]

def test_roundtrip_linear_prediction_matches_sklearn(tmp_path):
    from sklearn.linear_model import Ridge
    rng=np.random.default_rng(28)
    X=rng.normal(size=(160,384));X/=np.linalg.norm(X,axis=1,keepdims=True)
    y=50+X@rng.normal(size=384)
    reg=Ridge(alpha=0.1).fit(X,y)
    payload={'ridge':{'coef':reg.coef_.tolist(),'intercept':float(reg.intercept_),'clip':[0,100],'iq_offset':70,'iq_scale':.6},'vectors':X[:8].tolist()}
    file=tmp_path/'numeric.json';write_json(file,payload)
    code="import fs from 'node:fs';import {scoreVector} from './docs/src/core.mjs';const p=JSON.parse(fs.readFileSync(process.argv[1]));console.log(JSON.stringify(p.vectors.map(v=>scoreVector(v,p.ridge).raw)));"
    r=subprocess.run(['node','--input-type=module','-e',code,str(file)],cwd=ROOT,check=True,capture_output=True,text=True)
    np.testing.assert_allclose(json.loads(r.stdout),reg.predict(X[:8]),rtol=0,atol=1e-10)

def test_chunk_roundtrip(tmp_path):
    source=tmp_path/'big.onnx';source.write_bytes(bytes(range(256))*9)
    out=tmp_path/'bundle';chunks=split_file(source,out,chunk_bytes=128)
    assert b''.join((out/c['path']).read_bytes() for c in chunks)==source.read_bytes()
    assert all(digest(out/c['path'])==c['sha256'] for c in chunks)
    assert max(c['bytes'] for c in chunks)<=128

@pytest.mark.parametrize('chunk',[-1,0,49*1024**2])
def test_bad_chunk_size(tmp_path,chunk):
    src=tmp_path/'graph';src.write_bytes(b'abc')
    with pytest.raises(BundleError):split_file(src,tmp_path/'out',chunk)

def test_atomic_replace_requires_intent(tmp_path):
    out=tmp_path/'model';out.mkdir();(out/'existing').write_text('keep')
    stage=tmp_path/'stage';stage.mkdir();(stage/'new').write_text('new')
    with pytest.raises(BundleError):publish_directory(stage,out)
    assert (out/'existing').read_text()=='keep'
    publish_directory(stage,out,replace=True)
    assert (out/'new').exists() and not stage.exists()

def test_placeholder_replacement(tmp_path):
    out=tmp_path/'model';out.mkdir();write_json(out/'manifest.json',{'status':'missing'})
    stage=tmp_path/'stage';stage.mkdir();(stage/'new').write_text('new')
    publish_directory(stage,out)
    assert (out/'new').exists()

def test_source_metadata_not_training_text(student):
    _,m=student;public=public_metadata(m)
    assert 'PRIVATE TRAINING ROW' not in json.dumps(public) and '/private' not in json.dumps(public)
    m['embedding_spec']['model'].update(kind='local',repo='/home/user/model')
    assert '/home/user' not in json.dumps(public_metadata(m))

def test_local_checkpoint_requires_matching_hash(tmp_path):
    p=tmp_path/'encoder';p.mkdir();write_json(p/'config.json',{'hidden_size':384})
    hashes={'config.json':digest(p/'config.json')}
    spec={'model':{'kind':'local','repo':str(p),'revision':fingerprint(hashes)}}
    assert resolve_encoder(spec,True,None)==p
    (p/'config.json').write_text('changed')
    with pytest.raises(BundleError):resolve_encoder(spec,True,None)

def test_unknown_versions_fail_before_download():
    with pytest.raises(BundleError):check_versions({'versions':{'torch':'0.0.0'}})

def test_real_torch_pooling_wrapper():
    import torch
    class FakeBackbone(torch.nn.Module):
        # Tiny mathematical fixture, not a pretrained model.
        def forward(self,input_ids,attention_mask,token_type_ids):
            h=torch.stack([input_ids.float(),input_ids.float()+1],dim=-1)
            return SimpleNamespace(last_hidden_state=h)
    wrapper=make_wrapper(FakeBackbone())
    ids=torch.tensor([[1,3,100],[4,99,99]])
    mask=torch.tensor([[1,1,0],[1,0,0]])
    got=wrapper(ids,mask,torch.zeros_like(ids)).numpy()
    expected=np.array([[2.,3.],[4.,5.]])
    expected/=np.linalg.norm(expected,axis=1,keepdims=True)
    np.testing.assert_allclose(got,expected,atol=1e-7)

def test_missing_model_not_ready():
    assert check_main([])==2
    assert check_main(['--allow-missing-model'])==0

def test_checks_do_not_rewrite_site():
    site=ROOT/'docs'
    before={str(p): (p.stat().st_mtime_ns,p.stat().st_size) for p in site.rglob('*') if p.is_file()}
    assert check_main(['--allow-missing-model'])==0
    after={str(p): (p.stat().st_mtime_ns,p.stat().st_size) for p in site.rglob('*') if p.is_file()}
    assert before==after

def test_onnxruntime_license_uses_exact_upstream_tag():
    from vendor import SPECS
    urls = dict(SPECS)
    assert urls['onnxruntime/LICENSE'] == 'https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/LICENSE'
    assert not urls['onnxruntime/LICENSE'].startswith('https://cdn.jsdelivr.net/npm/onnxruntime-web@')


def test_all_vendor_sources_are_absolute_https_urls():
    from vendor import SPECS
    assert all(url.startswith('https://') for _, url in SPECS)
