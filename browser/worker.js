'use strict';

const runtimeURL = 'https://cdn.jsdelivr.net/pyodide/v0.29.3/full/';
let runtime;
let manifest;
const installed = new Set();
const base = new URL('../short_circuit/inverse_design/', self.location.href);

async function checked(url) {
  const location = new URL(url, self.location.href);
  location.search = self.location.search;
  const res = await fetch(location);
  if (!res.ok) throw Error('下载失败 ' + res.status + ' · ' + new URL(url, self.location.href).pathname);
  return res;
}

async function loadGroup(py, name) {
  if (installed.has(name)) return;
  await Promise.all(manifest[name].map(async item => {
    const file = typeof item === 'string' ? item : item.file;
    const location = typeof item === 'string' ? new URL(item, base) : new URL(item.source, self.location.href);
    const content = new Uint8Array(await (await checked(location)).arrayBuffer());
    const dest = '/pmut/' + file;
    py.FS.mkdirTree(dest.slice(0, dest.lastIndexOf('/')));
    py.FS.writeFile(dest, content);
  }));
  py.runPython('import importlib; importlib.invalidate_caches()');
  installed.add(name);
}

async function init() {
  postMessage({progress:'正在加载浏览器计算引擎与正向模型…'});
  importScripts(runtimeURL + 'pyodide.js');
  const py = await loadPyodide({indexURL:runtimeURL});
  const files = checked('assets.json').then(r => r.json());
  await py.loadPackage('numpy');
  manifest = await files;
  await loadGroup(py, 'forward');
  py.runPython(await (await checked('engine.py')).text());
  return py;
}

async function run({id, path, body}) {
  try {
    runtime ??= init().catch(error => { runtime = undefined; throw error; });
    const py = await runtime;
    if (path === '/family-design' && !installed.has('inverse')) {
      postMessage({progress:'正在加载逆向求解器和排布模型…'});
      await loadGroup(py, 'inverse');
    }
    if (path === '/example') await loadGroup(py, 'example');
    const dispatch = py.globals.get('dispatch');
    try {
      postMessage({id, result:JSON.parse(dispatch(path, JSON.stringify(body)))});
    } finally { dispatch.destroy(); }
  } catch (error) {
    postMessage({id,error:String(error.message || error)});
  }
}

// Serialize jobs so dictionary initialization and request-local solver state
// never compete, including during the first network downloads.
let previous = Promise.resolve();
self.onmessage = ({data}) => { previous = previous.then(() => run(data)); };
