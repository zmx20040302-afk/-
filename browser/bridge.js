'use strict';

const workerURL = new URL('./worker.js', document.currentScript.src);
workerURL.search = new URL(document.currentScript.src).search;
const pmutWorker = new Worker(workerURL);
const pmutRequests = new Map();
let pmutSequence = 0;
let pmutFailure;

pmutWorker.onmessage = ({data}) => {
  if (data.progress) {
    const status = document.getElementById('status');
    if (status) status.textContent = data.progress;
    return;
  }
  const request = pmutRequests.get(data.id);
  if (!request) return;
  pmutRequests.delete(data.id);
  if (data.error) request.reject(new Error(data.error));
  else request.resolve(data.result);
};
pmutWorker.onerror = (event) => {
  pmutFailure = new Error(event.message || '浏览器计算引擎启动失败，请刷新重试。');
  for (const request of pmutRequests.values()) request.reject(pmutFailure);
  pmutRequests.clear();
};

function browserAPI(path, body) {
  if (pmutFailure) return Promise.reject(pmutFailure);
  return new Promise((resolve, reject) => {
    const id = ++pmutSequence;
    pmutRequests.set(id, {resolve, reject});
    pmutWorker.postMessage({id, path, body: body || {}});
  });
}
