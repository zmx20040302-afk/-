'use strict';
let goal=null, storedCandidate=null, baseline=null, cacheField=null, currentSummary=null;
let displayFocus='peak', displaySlice=null, comparisonStart=null;
const loc=[-.5,-.25,0,.25,.5];
const scopeLabels={plane:'整面平均幅值',center:'几何中心',roi:'中心 25 点平均幅值',point:'所选测点'};
const metricNames=['12.50 MHz 整面误差','23 频整面误差','5.60 MHz 整面平均幅值偏差','5.60 MHz 中心幅值误差','12.50 MHz 中心幅值误差','5.60 MHz 25 点 RMSE','12.50 MHz 25 点 RMSE','离散峰位'];
function roiSample(field,f,x,y){
  const ix=(x-cfg.axis[0])/.12,iy=(y-cfg.axis[0])/.12,c=Math.floor(ix),r=Math.floor(iy);
  let re=0,im=0;
  for(let j=0;j<2;j++)for(let i=0;i<2;i++){
    const w=(i?ix-c:1-ix+c)*(j?iy-r:1-iy+r),k=2*(2500*f+(r+j)*50+c+i);
    re+=w*field[k];im+=w*field[k+1];
  }
  return [re,im,Math.hypot(re,im)];
}
function summary(field){
  const area=cfg.frequency.map((_,f)=>loc.flatMap(y=>loc.map(x=>roiSample(field,f,x,y)[2])));
  const plane=cfg.frequency.map((_,f)=>{
    let sum=0;
    for(let i=f*5000;i<(f+1)*5000;i+=2)sum+=Math.hypot(field[i],field[i+1]);
    return sum/2500;
  });
  return {area,plane,mean:area.map(row=>row.reduce((a,b)=>a+b)/25)};
}
function getSpectrum(field=P,region=currentSummary){
  const scope=$('spectrum').value;
  if(scope==='plane')return region.plane;
  if(scope==='center')return region.area.map(row=>row[12]);
  if(scope==='roi')return region.mean;
  return cfg.frequency.map((_,i)=>value(field,i,...pt)[0]);
}
function scopeLabel(){
  return $('spectrum').value==='point'?'测点 ('+cfg.axis[pt[1]]+', '+cfg.axis[pt[0]]+') mm':scopeLabels[$('spectrum').value];
}
function exportSpectrum(){
  fresh();
  if($('spectrum').value==='point')return csv('point_response.csv',makeRows(true,false,true));
  const vals=getSpectrum();csv($('spectrum').value+'_response.csv','MHz,Pa\n'+cfg.frequency.map((f,i)=>[f,vals[i]].join(',')).join('\n'));
}
function metrics(field,ref,region){
  let sum=0,energy=0,highErr=0,highEnergy=0,peakDiff=0,max=-1,peak=0;
  const main=cfg.frequency.indexOf(5.6),high=cfg.frequency.indexOf(12.5);
  for(let f=0;f<cfg.frequency.length;f++){
    let plane=0,diff=0;
    for(let j=0;j<2500;j++){
      const i=2*(f*2500+j),p=Math.hypot(field[i],field[i+1]),t=ref.amp[i/2],e=p-t;
      sum+=e*e;energy+=t*t;plane+=p;diff+=e;
      if(f===high){highErr+=e*e;highEnergy+=t*t}
    }
    if(plane>max){max=plane;peak=cfg.frequency[f]}
    if(f===main)peakDiff=Math.abs(diff/2500);
  }
  const e=[main,high].map(f=>region.area[f].map((p,j)=>p-ref.area[f][j]));
  return [100*Math.sqrt(highErr/highEnergy),100*Math.sqrt(sum/energy),peakDiff,...e.map(v=>Math.abs(v[12])),...e.map(v=>Math.sqrt(v.reduce((s,z)=>s+z*z,0)/25)),peak];
}
function setGoal(field,encoded,label){
  if(field.length!==115000 || !field.some(v=>v!==0))throw Error('请选择有声压输出的目标图');
  staleFamilyResults();
  const amp=new Float64Array(field.length/2);
  for(let i=0;i<amp.length;i++)amp[i]=Math.hypot(field[i*2],field[i*2+1]);
  const region=summary(field);
  goal={amp,field,region,area:region.area,encoded};baseline=null;cacheField=null;comparisonStart=null;
  $('refLabel').textContent=label;$('runNote').textContent='';
}
function encodeField(field){
  const bytes=new Uint8Array(field.buffer,field.byteOffset,field.byteLength);
  let s='';for(let i=0;i<bytes.length;i+=8192)s+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(s);
}
let currentMetrics=null;
const bandRows=[
  ['peak_Pa','峰幅',' Pa'],['peak_MHz','峰频',' MHz'],['left_MHz','−6 dB 左带边',' MHz'],
  ['right_MHz','−6 dB 右带边',' MHz'],['relative_bandwidth_pct','相对带宽','%']
];
const planeRows=[
  ['peak_plane_pct','目标峰频整面幅值误差','%'],['left_plane_pct','目标左带边整面幅值误差','%'],
  ['right_plane_pct','目标右带边整面幅值误差','%']
];
function fmt(value,unit,digits=3){
  return Number.isFinite(value)?value.toFixed(digits)+unit:'—';
}
function bandFeatures(values,frequency=cfg.frequency){
  const amplitude=Math.max(...values);
  const result={peak_Pa:amplitude,threshold_Pa:amplitude*10**(-6/20),peak_MHz:null,left_MHz:null,right_MHz:null,bandwidth_MHz:null,relative_bandwidth_pct:null};
  if(amplitude===0)return result;
  const peak=values.indexOf(amplitude);
  result.peak_MHz=frequency[peak];
  for(const step of [-1,1]){
    for(let i=peak+step;i>=0&&i<values.length;i+=step){
      if(values[i]>result.threshold_Pa)continue;
      const j=i-step;
      result[step<0?'left_MHz':'right_MHz']=frequency[i]+(frequency[j]-frequency[i])*(result.threshold_Pa-values[i])/(values[j]-values[i]);
      break;
    }
  }
  if(result.left_MHz!==null&&result.right_MHz!==null){
    result.bandwidth_MHz=result.right_MHz-result.left_MHz;
    result.relative_bandwidth_pct=200*result.bandwidth_MHz/(result.right_MHz+result.left_MHz);
  }
  return result;
}
function sliceAt(frequency){
  const hi=cfg.frequency.findIndex(f=>f>=frequency);
  if(hi<=0||cfg.frequency[hi]===frequency)return {lo:Math.max(hi,0),hi:Math.max(hi,0),weight:0,frequency};
  const lo=hi-1;
  return {lo,hi,weight:(frequency-cfg.frequency[lo])/(cfg.frequency[hi]-cfg.frequency[lo]),frequency};
}
function blend(slice,first,second){return first+(second-first)*slice.weight;}
function sliceValue(field,slice,index,phase=false){
  const a=(slice.lo*2500+index)*2;
  if(phase)return Math.atan2(field[a+1],field[a])*180/Math.PI;
  const low=Math.hypot(field[a],field[a+1]);
  if(slice.lo===slice.hi)return low;
  const b=(slice.hi*2500+index)*2;
  return blend(slice,low,Math.hypot(field[b],field[b+1]));
}
function selectSlice(band){
  const options={peak:band.peak_MHz,left:band.left_MHz,right:band.right_MHz};
  for(const [key,id] of Object.entries({peak:'viewPeak',left:'viewLeft',right:'viewRight'})){
    $(id).disabled=options[key]===null;
    $(id).setAttribute('aria-pressed',String(displayFocus===key&&options[key]!==null));
  }
  if(displayFocus!=='sample'&&options[displayFocus]===null)displayFocus='sample';
  const frequency=displayFocus==='sample'?cfg.frequency[k]:options[displayFocus];
  const slice=sliceAt(frequency);
  if(slice.lo===slice.hi && displayFocus!=='sample'){k=slice.lo;$('freq').selectedIndex=k;}
  $('kind').disabled=slice.lo!==slice.hi;
  if($('kind').disabled)$('kind').selectedIndex=0;
  return slice;
}
function bandSummary(band){
  if(band.peak_MHz===null)return '零声压，峰频与带边未定义。';
  const edge=(f)=>f===null?'采样范围内未闭合':fmt(f,' MHz');
  return '峰幅 '+fmt(band.peak_Pa,' Pa',2)+' · 峰频 '+fmt(band.peak_MHz,' MHz')+
    '\n−6 dB 阈值 '+fmt(band.threshold_Pa,' Pa',2)+' · 左 '+edge(band.left_MHz)+' · 右 '+edge(band.right_MHz)+
    (band.bandwidth_MHz===null?'':'\n带宽 '+fmt(band.bandwidth_MHz,' MHz')+' · 相对带宽 '+fmt(band.relative_bandwidth_pct,'%',2));
}
function drawSpectrum(values,slice){
  const g=$('curve').getContext('2d'),freq=cfg.frequency;
  const band=bandFeatures(values),top=band.peak_Pa||1;
  const x=f=>72+(f-freq[0])/(freq.at(-1)-freq[0])*405;
  const y=a=>260-a/top*210;
  function line(x1,y1,x2,y2,color){
    g.strokeStyle=color;g.beginPath();g.moveTo(x1,y1);g.lineTo(x2,y2);g.stroke();
  }
  function dot(f,a,color,r){
    g.fillStyle=color;g.beginPath();g.arc(x(f),y(a),r,0,2*Math.PI);g.fill();
  }
  g.clearRect(0,0,500,310);
  g.font='14px sans-serif';g.fillStyle='#26494d';g.fillText('Pa',8,20);g.lineWidth=1;
  for(let i=0;i<=4;i++){
    const v=top*i/4;
    g.fillText(v.toPrecision(3),2,y(v)+4);line(72,y(v),477,y(v),'#dbe4e3');
  }
  for(const f of [1,4,7,10,12.5])g.fillText(String(f),x(f)-5,280);
  g.fillText('MHz',449,304);
  if(band.peak_MHz!==null){
    g.setLineDash([5,4]);line(72,y(band.threshold_Pa),477,y(band.threshold_Pa),'#ad6735');g.setLineDash([]);
    g.fillStyle='#975426';g.fillText('−6 dB',75,y(band.threshold_Pa)-6);
  }
  g.strokeStyle='#217a73';g.lineWidth=2.3;g.beginPath();
  values.forEach((v,i)=>i?g.lineTo(x(freq[i]),y(v)):g.moveTo(x(freq[i]),y(v)));
  g.stroke();
  if(band.peak_MHz!==null){
    dot(band.peak_MHz,band.peak_Pa,'#155c57',5);
    g.textAlign='center';g.fillText('峰 '+fmt(band.peak_MHz,' MHz'),Math.max(116,Math.min(449,x(band.peak_MHz))),33);g.textAlign='start';
    for(const side of [band.left_MHz,band.right_MHz])if(side!==null)dot(side,band.threshold_Pa,'#ad6735',4);
  }
  g.strokeStyle='#153d67';g.lineWidth=1.5;g.beginPath();
  g.arc(x(slice.frequency),y(blend(slice,values[slice.lo],values[slice.hi])),7,0,2*Math.PI);g.stroke();
}
let comparisonKey=null;
function renderBandCompare(){
  const key=[goal.field,comparisonStart.field,P,$('spectrum').value,...pt];
  if(comparisonKey&&key.every((item,i)=>item===comparisonKey[i])&&$('compare').firstChild)return;
  comparisonKey=key;
  const features=[bandFeatures(getSpectrum(goal.field,goal.region)),bandFeatures(getSpectrum(comparisonStart.field,comparisonStart.region)),bandFeatures(getSpectrum())];
  const rows=[...bandRows,['threshold_Pa','−6 dB 阈值',' Pa'],['bandwidth_MHz','带宽',' MHz']];
  const mapRows=['peak_MHz','left_MHz','right_MHz'].map((key,i)=>{
    const frequency=features[0][key],slice=frequency===null?null:sliceAt(frequency);
    let norm2=0,startError=0,currentError=0;
    if(slice)for(let index=0;index<2500;index++){
      const ref=sliceValue(goal.field,slice,index);
      norm2+=ref*ref;
      startError+=(sliceValue(comparisonStart.field,slice,index)-ref)**2;
      currentError+=(sliceValue(P,slice,index)-ref)**2;
    }
    return '<tr><th>'+planeRows[i][1]+'</th><td>'+fmt(frequency,' MHz')+'</td><td>'+
      fmt(norm2?100*Math.sqrt(startError/norm2):null,'%')+'</td><td>'+fmt(norm2?100*Math.sqrt(currentError/norm2):null,'%')+'</td></tr>';
  });
  $('compare').innerHTML='<table><caption>'+scopeLabel()+' · 目标与开关图的响应</caption><thead><tr><th>指标</th><th>目标</th><th>本轮初图</th><th>当前图</th></tr></thead><tbody>'+
    rows.map(([key,label,unit])=>'<tr><th>'+label+'</th>'+features.map(s=>'<td>'+fmt(s[key],unit)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'+
    '<table><caption>在目标三个工作频率比较整面幅值</caption><thead><tr><th>指标</th><th>目标频率</th><th>初图误差</th><th>当前误差</th></tr></thead><tbody>'+mapRows.join('')+'</tbody></table>';
}
function renderStudy(){
  const ok=!!P && ready===rev;
  $('protect').disabled=!ok||!goal||designing;$('archived').disabled=!storedCandidate||designing;
  if(!ok){for(const id of ['geo','avg','minR'])$(id).textContent='—';$('roi').replaceChildren();$('compare').replaceChildren();$('band').textContent='';for(const id of ['viewPeak','viewLeft','viewRight'])$(id).disabled=true;displaySlice=null;return}
  if(cacheField!==P){currentSummary=summary(P);currentMetrics=goal?metrics(P,goal,currentSummary):null;cacheField=P}
  const features=bandFeatures(getSpectrum());
  displaySlice=selectSlice(features);
  const slice=displaySlice;
  const area=currentSummary.area[slice.lo].map((v,i)=>blend(slice,v,currentSummary.area[slice.hi][i]));
  const lo=Math.min(...area),hi=Math.max(...area);
  $('band').textContent=bandSummary(features);
  $('geo').textContent=area[12].toFixed(2)+' Pa';$('avg').textContent=(area.reduce((a,b)=>a+b,0)/25).toFixed(2)+' Pa';$('minR').textContent=lo.toFixed(2)+' Pa';
  $('roi').innerHTML='<caption>'+fmt(slice.frequency,' MHz')+(slice.lo===slice.hi?'':' · 频间幅值插值')+' · 幅值（Pa）</caption><tr><th>y / x mm</th>'+loc.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+[4,3,2,1,0].map(y=>'<tr><th>'+loc[y]+'</th>'+loc.map((_,x)=>{let v=area[y*5+x],q=(v-lo)/(hi-lo||1);return '<td style="background:hsl(169 34% '+(95-q*24)+'%)">'+v.toFixed(1)+'</td>'}).join('')+'</tr>').join('');
  if(!goal){$('compare').replaceChildren();return}
  if(!baseline)baseline=currentMetrics.slice();
  if(!comparisonStart)comparisonStart={field:P,region:currentSummary};
  if($('planeMode').value==='peak_band'){renderBandCompare();return;}
  $('compare').innerHTML='<table><caption>相对已保存目标 · 基线与当前图</caption><tr><th>指标</th><th>基线</th><th>当前图</th><th>变化</th></tr>'+metricNames.map((s,i)=>{
    const u=i<2?'%':i===7?' MHz':' Pa',delta=currentMetrics[i]-baseline[i];
    return '<tr><th>'+s+'</th><td>'+baseline[i].toFixed(3)+u+'</td><td>'+currentMetrics[i].toFixed(3)+u+'</td><td>'+(delta>0?'+':'')+delta.toFixed(3)+(i<2?' 个百分点':u)+'</td></tr>'
  }).join('')+'</table>';
}
function fullReference(text){
  const rows=lines(text);if(/^MHz,/i.test(rows[0]||''))rows.shift();
  if(rows.length!==57500)throw Error('整面目标需 23 频 × 2500 行，列为 MHz,x_mm,y_mm,Pa,phase_deg');
  const result=new Float64Array(115000),seen=new Uint8Array(57500);
  rows.forEach((line,n)=>{
    const parts=line.split(',').map(v=>v.trim()),[f,x,y,a,phase]=parts.map(Number);
    const k=cfg.frequency.indexOf(f),j=cfg.axis.indexOf(y),i=cfg.axis.indexOf(x);
    if(parts.length!==5||parts.slice(0,4).includes('')||!Number.isFinite(a)||a<0||!Number.isFinite(phase)||(a>0&&!parts[4])||Math.min(k,i,j)<0)throw Error('全频复声场第 '+(n+1)+' 行的坐标、幅值或相位无效');
    const pos=(50*k+j)*50+i;if(seen[pos])throw Error('重复的频率与坐标，第 '+(n+1)+' 行');seen[pos]=1;
    result[2*pos]=a*Math.cos(phase*Math.PI/180);result[2*pos+1]=a*Math.sin(phase*Math.PI/180);
  });return result;
}
$('importRef').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  try{if(designing||busy||!cfg)throw Error('请等待当前计算完成');
    const version=familyGeneration,text=await file.text();if(designing||busy||version!==familyGeneration)throw Error('当前任务已改变，请重新导入目标');
    const field=fullReference(text);setGoal(field,encodeField(field),'已导入目标 · '+file.name);storedCandidate=null;draw();
  }catch(err){note(err.message,true)}finally{e.target.value=''}
};
for(const option of $('spectrum').options)$('inverseScope').add(new Option(option.text,option.value));
function syncScope(event){
  $('spectrum').value=$('inverseScope').value=event.target.value;
  resetScore();
  if(cfg)draw();
}
$('spectrum').onchange=$('inverseScope').onchange=syncScope;
function modeChanged(){
  comparisonKey=null;
  const peak=$('planeMode').value==='peak_band';
  $('scopeLabel').hidden=!peak;
  $('protect').textContent=peak?'固定通电数，匹配峰值与带边':'固定通电数，优化 12.50 MHz';
  $('modeHelp').textContent=peak?
    '按所选响应的主峰及左右 −6 dB 交点搜索，并比较目标三个频点的整面幅值。相对初图通过要求八项误差均未变差且至少一项改善，或各项误差均在数值精度内。':
    '高频保护沿用历史验收，比较 12.50 MHz 整面误差、初图峰位和六项上限。';
}
$('planeMode').onchange=()=>{modeChanged();resetScore();if(cfg)draw()};
modeChanged();
for(const [key,id] of Object.entries({peak:'viewPeak',left:'viewLeft',right:'viewRight'})){
  $(id).onclick=()=>{if(P&&ready===rev){displayFocus=key;draw()}};
}
action('saveR',()=>{fresh();const rows=['MHz,x_mm,y_mm,Pa,phase_deg'];cfg.frequency.forEach((f,i)=>loc.forEach(y=>loc.forEach(x=>{const [re,im,v]=roiSample(P,i,x,y);rows.push([f,x,y,v,v?Math.atan2(im,re)*180/Math.PI:''].join(','))})));csv('center25.csv',rows.join('\n'))});
action('capture',()=>{fresh();setGoal(P,encodeField(P),'目标为当前图已冻结的 23 频复声场。请修改开关或导入起始矩阵，然后优化。');storedCandidate=null;draw()});
function replaceMask(mask){M=mask.map(r=>r.slice());invalidate()}
action('example',async()=>{if(busy||pending)throw Error('请等待当前计算完成');designing=busy=true;try{const ex=await api('/example');setGoal(unpack(ex.target),ex.target,'目标 S005525 · 原库声场 · 65 个阵元通电');storedCandidate=ex.candidate;replaceMask(ex.mask)}finally{designing=busy=false;renderStudy()}});
action('archived',()=>{if(busy||pending)throw Error('请等待当前计算完成');replaceMask(storedCandidate);$('runNote').textContent='已载入历史 12.50 MHz 方案，按当前模式重新比较。'});
const familyNames = {random:'随机分散',cluster:'紧凑成团',edge:'边缘优先',stripe:'条纹',patch:'不规则多斑块'};
const familyModes = {peak_band:'峰值与 −6 dB 通带',protected:'整面保护 / 12.50 MHz',weighted:'测点或频响 / 逐频归一化',absolute:'测点或频响 / 逐项等权'};
const familyStates = {verified:'声学达标',partial:'形态符合 · 声学未达标',unavailable:'本次未找到该类方案',error:'请求失败'};
let familyGeneration=0, familyRun=null;

function finishFamilyApply(version,ok){
  if(!familyRun||familyRun.stale||familyRun.appliedRev!==version)return;
  $('familyProgress').textContent=familyRun.appliedLabel+(ok?' · 当前开关图的声场、振速和比较指标已更新。':' · 当前图复算失败，可点“立即重算”重试。');
}

function staleFamilyResults(){
  familyGeneration++;
  if(!familyRun||familyRun.stale)return;
  familyRun.stale=true;
  $('familyProgress').textContent='初图、目标或选项已改变。下方保留上次结果供查看和导出，请重新反求后再应用。';
  $('familyCards').querySelectorAll('.familyApply').forEach(btn=>btn.disabled=true);
}

for(const id of ['familyFull','familyPoint']){
  const menu=$(id);
  menu.add(new Option('全部五类','all'));
  for(const [key,name] of Object.entries(familyNames))menu.add(new Option(name,key));
  menu.onchange=()=>{
    $('familyFull').value=$('familyPoint').value=menu.value;
    resetScore();
  };
}
$('fixed').onchange=$('weight').onchange=()=>resetScore();

function familyNode(tag,text,style){
  const el=document.createElement(tag);if(text!==undefined)el.textContent=text;
  if(style)el.className=style;return el;
}

function maskCanvas(mask,label){
  const canvas=document.createElement('canvas');canvas.width=canvas.height=150;
  canvas.className='familyMap';canvas.setAttribute('role','img');canvas.setAttribute('aria-label',label+'，第1行在上，深绿通电');
  const ctx=canvas.getContext('2d');
  ctx.fillStyle='#e7eeea';ctx.fillRect(0,0,150,150);ctx.fillStyle='#25655e';
  for(let i=0;i<50;i++)for(let j=0;j<50;j++)if(mask[i][j])ctx.fillRect(j*3,i*3,3,3);
  return canvas;
}

function familyMetricRows(mode,m){
  const number=(v,unit)=>Number.isFinite(v)?v.toFixed(3)+unit:'—';
  let rows=[['通电阵元',Number.isInteger(m.active)?m.active+' / 2500':'—']];
  if(mode==='peak_band'){
    const response=m.response||{};
    rows.push(['候选峰值',fmt(response.peak_Pa,' Pa')+' · '+fmt(response.peak_MHz,' MHz')],
      ['候选 −6 dB 左 / 右带边',fmt(response.left_MHz,' MHz')+' / '+fmt(response.right_MHz,' MHz')],
      ['相对带宽',fmt(response.relative_bandwidth_pct,'%')]);
    const checks=Object.values(m.checks||{});
    rows.push(['误差保持或改善',checks.filter(x=>x===true).length+' / 8']);
    if(checks.includes(null))rows.push(['未能比较',checks.filter(x=>x===null).length+' 项']);
  }else if(mode==='protected'){
    rows.push(['12.50 MHz 整面误差',number(m.before,'%')+' → '+number(m.after,'%')]);
    if(Number.isFinite(m.initial_peak_MHz)&&Number.isFinite(m.peak_MHz))
      rows.push(['离散峰位',m.initial_peak_MHz+' → '+m.peak_MHz+' MHz']);
    if(Array.isArray(m.limits_ok))rows.push(['保护项达标',m.limits_ok.filter(Boolean).length+' / '+m.limits_ok.length]);
  }else{
    rows.push(['目标达标项',(Number.isInteger(m.passed)?m.passed:'—')+' / '+(Number.isInteger(m.items)?m.items:'—')],
      ['RMSE',number(m.initial_rmse,' Pa')+' → '+number(m.rmse,' Pa')],
      ['最大绝对误差',number(m.maximum,' Pa')]);
    if(Number.isFinite(m.relative))rows.push(['相对误差',number(m.relative,'%')]);
  }
  if(Number.isInteger(m.changed))rows.push(['相对初图改变',m.changed+' 格']);
  return rows;
}
function resultLabel(mode,result){
  if(mode==='peak_band'&&result.status==='verified')
    return result.metrics?.exact_response?'八项指标数值匹配':'相对初图通过';
  return familyStates[result.status];
}
function addBandChecks(details,m){
  const table=familyNode('table',undefined,'checkTable');
  table.append(familyNode('caption','各项误差相对初图的验收'));
  const head=table.createTHead().insertRow();
  for(const title of ['误差','初图','候选','判定'])head.append(familyNode('th',title));
  const body=table.createTBody();
  for(const [key,label,unit] of [...bandRows,...planeRows]){
    const check=m.checks?.[key],u=key==='relative_bandwidth_pct'?' 百分点':unit;
    const row=body.insertRow();
    for(const text of [label,fmt(m.errors_before?.[key],u),fmt(m.errors_after?.[key],u),
      check===true?'保持或改善':check===false?'超限':'不可比'])row.insertCell().textContent=text;
  }
  details.append(table);
}

function paintFamilyCard(run,family,result){
  const card=run.cards.get(family);
  card.replaceChildren(familyNode('h3',familyNames[family]),familyNode('p',resultLabel(run.mode,result),'familyState '+result.status));
  if(result.reason)card.append(familyNode('p',result.reason,'familyReason'));
  if(!result.mask)return;
  card.append(maskCanvas(result.mask,familyNames[family]));
  const rows=familyMetricRows(run.mode,result.metrics);
  const dl=familyNode('dl',undefined,'familyMetrics');
  for(const [term,value] of rows)dl.append(familyNode('dt',term),familyNode('dd',value));
  card.append(dl);
  const details=familyNode('details');
  details.append(familyNode('summary',run.mode==='peak_band'?'逐项验收与排布':'放大查看开关图'),maskCanvas(result.mask,familyNames[family]));
  const m=result.metrics;
  if(run.mode==='peak_band')addBandChecks(details,m);
  if(run.mode==='protected'&&[m.limits_before,m.limits_after,m.limits_ok].every(a=>Array.isArray(a)&&a.length===6)){
    const limits=familyNode('dl',undefined,'familyMetrics');
    for(let i=0;i<6;i++){
      const unit=i===0?'%':' Pa';
      limits.append(familyNode('dt',metricNames[i+1]),
        familyNode('dd',m.limits_before[i].toFixed(3)+unit+' → '+m.limits_after[i].toFixed(3)+unit+' · '+(m.limits_ok[i]?'保持':'超限')));
    }
    details.append(limits);
  }
  if(Number.isInteger(m.trials)&&Number.isFinite(m.elapsed))details.append(familyNode('p',
    '评价 '+m.trials+' 次 · 用时 '+m.elapsed.toFixed(2)+' 秒','info'));
  card.append(details);
  const actions=familyNode('div',undefined,'familyActions');
  const apply=familyNode('button','应用并复算','familyApply');
  apply.type='button';apply.disabled=run.running||run.stale;
  apply.onclick=()=>{
    if(familyRun!==run||run.stale||designing)return;
    if(busy||pending){note('当前开关图正在重算，请稍后再应用',true);return}
    M=result.mask.map(row=>row.slice());
    if(goal&&run.baseline)baseline=run.baseline.slice();
    invalidate(true);
    run.appliedRev=rev;run.appliedLabel='已应用'+familyNames[family]+'（'+resultLabel(run.mode,result)+'）';
    for(const button of $('familyCards').querySelectorAll('.familyApply'))button.textContent='应用并复算';
    apply.textContent='已应用到当前图';
    $('familyProgress').textContent='当前图已切换至'+familyNames[family]+'方案（'+familyStates[result.status]+'），上方正在复算整面声场。其他结果仍以本轮共同初图为参照。';
  };
  const save=familyNode('button','导出 CSV');save.type='button';
  save.onclick=()=>csv('PMUT_'+run.mode+'_'+family+'_'+result.status+'.csv',result.mask.map(row=>row.join(',')).join('\n'));
  actions.append(apply,save);card.append(actions);
}

function validateFamilyResult(family,result){
  if(!result||result.family!==family||!['verified','partial','unavailable'].includes(result.status))
    throw Error('服务返回的排布类别或验收状态无效');
  if(result.status==='unavailable')return {...result,mask:null};
  if(!Array.isArray(result.mask)||result.mask.length!==50||result.mask.some(
    row=>!Array.isArray(row)||row.length!==50||row.some(x=>x!==0&&x!==1)))throw Error('结果矩阵需为 50×50 个 0 或 1');
  const active=result.mask.flat().reduce((n,v)=>n+v,0);
  if(result.metrics?.active!=null&&result.metrics.active!==active)throw Error('返回阵元数与结果矩阵不一致');
  return {...result,metrics:{...result.metrics,active}};
}

async function readProtectedBatch(mask,target,onResult,options={}){
  const response=await fetch(options.mode==='peak_band'?'/peak-families':'/protected-families',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({mask,target,...options})});
  if(response.status===404)return false;
  if(!response.ok){
    const message=await response.json();
    throw Error(message.error||response.statusText);
  }
  const reader=response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer='';
  function receive(line){
    const result=JSON.parse(line);
    if(result.error)throw Error(result.error);
    onResult(result);
  }
  try{
    for(;;){
      const {done,value}=await reader.read();
      if(done)break;
      buffer+=value;
      const lines=buffer.split('\n');
      buffer=lines.pop();
      for(const line of lines)if(line.trim())receive(line);
    }
    if(buffer.trim())receive(buffer);
  }finally{reader.releaseLock()}
  return true;
}

async function runFamilyDesign(mode){
  fresh();
  if(busy||pending||designing)throw Error('请等待当前计算完成');
  const protectedMode=mode==='protected'||mode==='peak_band';
  if(protectedMode&&!goal)throw Error('请先导入或设置整面目标');
  const goals=protectedMode?[]:targets($('target').value);
  if(!protectedMode&&!goals.length)throw Error('请填写或导入测点目标');
  const selected=$(protectedMode?'familyFull':'familyPoint').value;
  const families=selected==='all'?Object.keys(familyNames):[selected];
  const mask=M.map(row=>row.slice()), active=mask.flat().reduce((n,v)=>n+v,0);
  const body={mode,mask,goals,fixed:protectedMode||$('fixed').checked,target:protectedMode?goal.encoded:null};
  const options=mode==='peak_band'?{mode,scope:$('spectrum').value,point:pt.slice()}:{mode};
  Object.assign(body,options);
  if(protectedMode){comparisonStart={field:P,region:currentSummary};baseline=currentMetrics.slice()}
  const run={generation:++familyGeneration,mode,stale:false,running:true,cards:new Map(),baseline:currentMetrics?.slice()};
  familyRun=run;designing=busy=true;pen=null;
  const locked=['importM','importRef','importT','target','fixed','weight','familyPoint','familyFull','planeMode','spectrum','inverseScope','solve','restore','clear','calc','capture','example','fillF','fillC'];
  const status=$(protectedMode?'runNote':'score');
  const results=[];
  try{
    locked.forEach(id=>$(id).disabled=true);renderStudy();
    $('runNote').textContent=$('score').textContent='';
    $('familyResults').hidden=false;
    document.querySelectorAll('.resultsLink').forEach(link=>link.hidden=false);
    $('familyContext').textContent=familyModes[mode]+(mode==='peak_band'?' · '+scopeLabel():'')+' · 共同初图通电 '+active+' / 2500 · '+(body.fixed?'固定通电数':'允许调整通电数')+
      (protectedMode?'':' · '+goals.length+' 项目标')+'。深绿格为通电位置。';
    $('familyCards').replaceChildren();
    for(const family of families){
      const card=familyNode('article',undefined,'familyCard');card.dataset.family=family;
      card.append(familyNode('h3',familyNames[family]),familyNode('p','等待计算','queue'));
      run.cards.set(family,card);$('familyCards').append(card);
    }
    function display(family,result){
      if(run.generation!==familyGeneration)return;
      results.push(result);
      paintFamilyCard(run,family,result);
    }
    let batched=false;
    if(protectedMode && selected==='all' && typeof TextDecoderStream!=='undefined'){
      const remaining=new Set(families);
      let failure='服务响应提前结束，请重试尚未返回的类别。';
      try{
        status.textContent=$('familyProgress').textContent='正在计算五类整面排布…';
        for(const card of run.cards.values())card.querySelector('.queue').textContent='正在搜索并复算…';
        batched=await readProtectedBatch(mask,goal.encoded,result=>{
          const family=result.family;
          if(!remaining.delete(family))throw Error('服务重复返回类别或类别未知。');
          try{
            if(result.status==='error')throw Error(result.reason||'类别搜索失败。');
            display(family,validateFamilyResult(family,result));
          }catch(err){display(family,{status:'error',reason:err.message,mask:null})}
          if(!run.stale)status.textContent=$('familyProgress').textContent='已返回 '+results.length+' / '+families.length+' 类';
        },options);
      }catch(err){batched=true;failure=err.message}
      if(batched)for(const family of remaining)display(family,{status:'error',reason:failure,mask:null});
    }
    if(!batched)for(const [i,family] of families.entries()){
      if(run.generation!==familyGeneration)break;
      const progress='正在计算 '+(i+1)+' / '+families.length+' · '+familyNames[family];
      status.textContent=$('familyProgress').textContent=progress;note(progress);
      run.cards.get(family).querySelector('.queue').textContent='正在搜索并复算…';
      let result;
      try{result=validateFamilyResult(family,await api('/family-design',{...body,family}))}
      catch(err){result={status:'error',reason:err.message,mask:null}}
      if(run.generation!==familyGeneration)break;
      display(family,result);
    }
    if(!run.stale){
      const counts=new Map();
      for(const result of results){
        const label=resultLabel(mode,result);
        counts.set(label,(counts.get(label)||0)+1);
      }
      const summary=Array.from(counts,([label,n])=>label+' '+n+' 类').join(' · ');
      status.textContent=$('familyProgress').textContent='完成 '+results.length+' / '+families.length+' 类。'+summary;
      note('搜索结束，查看结果后可应用或导出。');
    }
  }finally{
    run.running=false;designing=busy=false;locked.forEach(id=>$(id).disabled=false);renderStudy();
    $('familyCards').querySelectorAll('.familyApply').forEach(btn=>btn.disabled=run.stale);
    if(pending)predict();
  }
}

action('solve',()=>runFamilyDesign($('weight').selectedIndex?'weighted':'absolute'));
action('protect',()=>runFamilyDesign($('planeMode').value));
