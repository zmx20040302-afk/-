"""Regenerate the GitHub Pages entry and worker file list from the native release."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "short_circuit/inverse_design"
UI = MODELS / "S237"
prefix = "short_circuit/inverse_design/"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Native entry changed, update the browser adapter: {old!r}")
    return text.replace(old, new)


def build():
    source = (UI / "index.html").read_text(encoding="utf-8")
    page = replace_once(source, 'href="/studio.css"', f'href="{prefix}S237/studio.css"')
    page = replace_once(page, "src='/studio.js'", f"src='{prefix}S237/studio.js'")
    page = replace_once(page, 'href="/docs/S308/classification.html"', f'href="{prefix}S308/classification.html"')
    page = replace_once(page, 'href="/docs/S321/report.html"', f'href="{prefix}S321/report.html"')
    page = page.replace("<script>", '<script src="browser/bridge.js"></script><script>', 1)
    page, changes = re.subn(r'^async function api\(path,body\).*$', 'const api = browserAPI;', page, flags=re.M)
    if changes != 1:
        raise ValueError("Native API entry changed")
    page = replace_once(page, "(async()=>{try{cfg=", "readProtectedBatch=async()=>false; (async()=>{try{cfg=")
    page = replace_once(page, "请通过本机启动器打开。", "浏览器计算引擎加载失败，请刷新后重试。")
    (ROOT / ".nojekyll").touch()
    # WASM has no Python thread/process workers. Only replace the two scheduling
    # hooks in the worker's copies; equations, classifiers and bounds stay shared.
    generated = HERE / "generated"
    generated.mkdir(exist_ok=True)
    repair = (UI / "cluster_repair.py").read_text(encoding="utf-8")
    repair = replace_once(repair, "from multiprocessing import parent_process", "")
    repair = replace_once(repair, "return None if parent_process() is not None else ThreadPoolExecutor(max_workers=4)", "return None")
    (generated / "cluster_repair.py").write_text(repair, encoding="utf-8")
    continuation = (UI / "cluster_continuation.py").read_text(encoding="utf-8")
    continuation = replace_once(continuation, "from concurrent.futures import ThreadPoolExecutor", "from browser_executor import SerialExecutor as ThreadPoolExecutor")
    (generated / "cluster_continuation.py").write_text(continuation, encoding="utf-8")
    forward = ["S217/model.py", "S217/kernels.npz", "S217/start.csv"]
    inverse = ["S231/targets.json","S232/common.py","S281/geometry.py"]
    for path in sorted((*UI.glob("*.py"), *UI.glob("*.npz"), *(MODELS/"S284").glob("*.py"), *(MODELS/"S284").glob("*.npz"))):
        if path.name in ("workbench.py","plane_batch.py"):
            continue
        name = path.relative_to(MODELS).as_posix()
        inverse.append(dict(file=name,source="generated/"+path.name) if (generated/path.name).exists() else name)
    inverse.append(dict(file="S237/browser_executor.py",source="executor.py"))
    example = ["S233/S005525_20261123_base.csv","S233/S005525_ref.npy","S234/S005525_20261123_visit38.csv"]
    version = hashlib.sha256()
    for path in (UI/"index.html", UI/"studio.js", UI/"studio.css", HERE/"engine.py",
                 HERE/"bridge.js", HERE/"worker.js"):
        version.update(path.read_bytes())
    for item in (*forward, *inverse, *example):
        path = MODELS/item if isinstance(item, str) else HERE/item["source"]
        if not path.is_file():
            raise ValueError(f"Missing web asset {path}")
        version.update(path.read_bytes())
    (HERE/"assets.json").write_text(json.dumps(dict(forward=forward,inverse=inverse,example=example),indent=2)+"\n",encoding="utf-8")
    revision = version.hexdigest()[:12]
    for asset in ("browser/bridge.js", prefix+"S237/studio.css", prefix+"S237/studio.js"):
        page = replace_once(page, asset, asset+"?v="+revision)
    (ROOT/"index.html").write_text(page,encoding="utf-8")


if __name__ == "__main__":
    build()
