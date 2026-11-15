import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="ADAS Pedestrian Intent", layout="wide", page_icon="🚸")

st.markdown("""
<style>
.stApp{background:#0b1220;color:#e6edf7}
header[data-testid="stHeader"]{background:transparent}
.stApp p,.stApp label,.stApp li,.stApp span{color:#e6edf7}
.block-container{padding-top:1.2rem;max-width:1300px}
.hero{padding:22px 26px;border-radius:16px;background:linear-gradient(120deg,#0f2747,#13395f 55%,#0b1220);border:1px solid #1f3b5f;margin-bottom:14px}
.hero h1{margin:0;font-size:30px;letter-spacing:.5px;color:#fff}
.hero p{margin:6px 0 0 0;color:#9fb6d3;font-size:15px}
.pipe{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:10px 0 14px 0}
.chip{padding:9px 14px;border-radius:10px;background:#12233d;border:1px solid #27466e;color:#d6e6ff;font-weight:600;font-size:14px}
.arrow{color:#4da3ff;font-size:18px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}
.card{padding:14px 16px;border-radius:12px;background:#10203a;border:1px solid #1f3b5f}
.card b{display:block;font-size:26px;color:#4da3ff}
.card span{color:#9fb6d3;font-size:13px}
.alertbox{padding:18px;border-radius:12px;font-weight:800;font-size:20px;text-align:center;letter-spacing:.5px}
.alert{background:linear-gradient(90deg,#8b0000,#e11d1d);color:#fff;box-shadow:0 0 24px #e11d1d88}
.safe{background:#0f3d2a;color:#7dffb3;border:1px solid #1d7a52}
.big{font-size:54px;font-weight:800;color:#fff;line-height:1}
.next{margin-top:18px;padding:14px 18px;border-radius:12px;background:#10203a;border:1px dashed #2f5d92;color:#9fb6d3;text-align:center}
</style>
<div class="hero">
  <h1>🚸 AI Pedestrian Crossing Intent Prediction</h1>
  <p>Early warning for ADAS: understands body pose and motion to predict that a pedestrian will step onto the road, about one second before it happens.</p>
</div>
<div class="pipe">
  <div class="chip">📷 Camera</div><div class="arrow">➜</div>
  <div class="chip">🦴 Pose estimation (17 keypoints)</div><div class="arrow">➜</div>
  <div class="chip">⏱ 30-frame window + motion</div><div class="arrow">➜</div>
  <div class="chip">🧠 LSTM</div><div class="arrow">➜</div>
  <div class="chip">🚗 Braking decision</div>
</div>
<div class="cards">
  <div class="card"><b>17</b><span>body keypoints per person</span></div>
  <div class="card"><b>30</b><span>frames per prediction window</span></div>
  <div class="card"><b>~1 s</b><span>early warning before crossing</span></div>
  <div class="card"><b>36</b><span>features per frame (pose + motion)</span></div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all():
    import tensorflow as tf
    model = tf.keras.models.load_model("lstm_jaad.keras")
    nz = np.load("norm_jaad.npz")
    data = np.load("demo_data.npz")
    meta = json.load(open("clips.json"))
    mu, sd, X = nz["mu"], nz["sd"], data["x"]
    probs = [float(model.predict(((X[k] - mu) / sd)[None].astype("float32"), verbose=0)[0, 0])
             for k in range(len(meta))]
    return meta, probs


meta, probs = load_all()

LIVE_HTML = r"""
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box}
body{margin:0;background:#0b1220;color:#e6edf7;font-family:'Segoe UI',Arial,sans-serif}
.grid{display:grid;grid-template-columns:3fr 2fr;gap:16px;padding:4px}
#stage{position:relative;background:#000;border-radius:14px;overflow:hidden;aspect-ratio:4/3;border:3px solid #1f3b5f}
#stage.alarm{border-color:#ff2d2d;animation:flash .5s infinite alternate}
@keyframes flash{from{box-shadow:0 0 0 #ff2d2d}to{box-shadow:0 0 40px #ff2d2d}}
video,#c{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;transform:scaleX(-1)}
#log{position:absolute;left:10px;top:10px;font-family:Consolas,monospace;font-size:13px;color:#7dffb3;background:#000a;padding:8px 10px;border-radius:8px;max-width:90%;white-space:pre-line}
#overlay{position:absolute;inset:0;display:none;align-items:flex-start;justify-content:center;pointer-events:none}
#stage.alarm #overlay{display:flex}
#overlay div{margin-top:14px;background:#d90000;color:#fff;font-weight:800;padding:10px 18px;border-radius:10px;font-size:20px}
#hud{position:absolute;right:10px;bottom:10px;font-family:Consolas,monospace;font-size:12px;color:#bfe3ff;background:#000a;padding:6px 10px;border-radius:8px}
.panel{background:#10203a;border:1px solid #1f3b5f;border-radius:14px;padding:14px;margin-bottom:12px}
.big{font-size:52px;font-weight:800;line-height:1;color:#fff}
.sub{color:#9fb6d3;font-size:12px;margin-top:4px;letter-spacing:.3px}
.bar{position:relative;height:14px;background:#0b1730;border-radius:8px;margin:12px 0;overflow:hidden}
#fill{height:100%;width:0;background:linear-gradient(90deg,#1ec878,#f5c542,#ff3b3b);transition:width .15s}
#thrline{position:absolute;top:-2px;bottom:-2px;width:3px;background:#fff}
.tag{padding:11px;border-radius:10px;text-align:center;font-weight:800;font-size:15px;background:#12233d;color:#9fb6d3;margin-top:8px}
.ok{background:#0f3d2a;color:#7dffb3}.al{background:linear-gradient(90deg,#8b0000,#e11d1d);color:#fff}.warn{background:#4a3a0a;color:#ffd666}
button{background:#1d6fe0;color:#fff;border:0;border-radius:10px;padding:10px 16px;font-size:14px;font-weight:700;cursor:pointer;margin-right:6px}
button.sec{background:#27466e}
button:disabled{opacity:.5;cursor:default}
input[type=range]{width:100%}
label.t{display:flex;align-items:center;gap:8px;font-size:13px;margin-top:8px;color:#d6e6ff}
#graph{width:100%;height:120px;background:#0b1730;border-radius:10px;display:block}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{padding:4px 6px;border-bottom:1px solid #1f3b5f;text-align:left;color:#cfe1ff}
#chk{font-size:11px;color:#6f8bb0;margin-top:6px}
.row{display:flex;justify-content:space-between;gap:8px;font-size:13px;margin-top:6px;color:#d6e6ff}
</style></head><body>
<div class="grid">
  <div>
    <div id="stage">
      <video id="v" playsinline muted></video>
      <canvas id="c"></canvas>
      <div id="log">System offline. Press START.</div>
      <div id="overlay"><div>⚠ PEDESTRIAN CROSSING INTENT DETECTED</div></div>
      <div id="hud">FPS -- | latency -- ms</div>
    </div>
    <div style="margin:12px 0">
      <button id="start">▶ START SYSTEM</button>
      <button id="stop" class="sec" disabled>■ STOP</button>
      <button id="test" class="sec">🔔 Test alarm</button>
    </div>
    <div class="panel">
      <div class="sub">LIVE CROSSING PROBABILITY (last 15 s)</div>
      <canvas id="graph" width="900" height="120"></canvas>
    </div>
    <div class="panel">
      <div class="sub">ALERT EVENT LOG</div>
      <table id="evt"><tr><th>Time</th><th>Probability</th><th>Threshold</th><th>Decision</th></tr></table>
      <div style="margin-top:8px"><button id="csv" class="sec">⬇ Download log (CSV)</button><button id="clr" class="sec">Clear</button></div>
    </div>
  </div>
  <div>
    <div class="panel">
      <div class="sub">CROSSING PROBABILITY</div>
      <div class="big" id="pv">--</div>
      <div class="bar"><div id="fill"></div><div id="thrline" style="left:50%"></div></div>
      <div class="tag" id="banner">SYSTEM OFFLINE</div>
      <div class="tag" id="dir">MOVEMENT: --</div>
      <div class="sub" style="margin-top:14px">ALERT SENSITIVITY (threshold <b id="tv">0.50</b>)</div>
      <input type="range" id="thr" min="0.1" max="0.9" step="0.05" value="0.5">
      <label class="t"><input type="checkbox" id="night"> 🌙 Low-light enhancement (brightness + contrast before pose)</label>
    </div>
    <div class="panel">
      <div class="sub">🚧 DANGER ZONE (road edge)</div>
      <label class="t"><input type="checkbox" id="zoneOn" checked> Enable danger zone (alert is more sensitive near the road edge)</label>
      <div class="sub" style="margin-top:8px">Road edge position on screen: <b id="zv">75</b>%</div>
      <input type="range" id="zone" min="10" max="90" step="1" value="75">
      <div class="row"><span>Pedestrian zone status</span><b id="zst">--</b></div>
    </div>
    <div class="panel">
      <div class="sub">🚗 VEHICLE SIMULATION (speed and distance are set by you)</div>
      <div class="sub" style="margin-top:8px">Vehicle speed: <b id="sv">40</b> km/h</div>
      <input type="range" id="spd" min="10" max="80" step="5" value="40">
      <div class="sub" style="margin-top:8px">Distance to pedestrian: <b id="dv">25</b> m</div>
      <input type="range" id="dst" min="5" max="60" step="1" value="25">
      <div class="row"><span>Stopping distance</span><b id="stp">--</b></div>
      <div class="row"><span>Extra distance from 1 s early warning</span><b id="gain">--</b></div>
      <div class="tag" id="brake">CRUISING</div>
      <div class="sub" style="margin-top:6px">Assumes 1.0 s reaction time and 7 m/s² braking on dry road.</div>
    </div>
    <div class="sub">Tip: stand 2 to 3 m from the camera, full body in view, and walk sideways across the frame.</div>
    <div id="chk"></div>
  </div>
</div>
<script id="w" type="application/json">__WEIGHTS__</script>
<script type="module">
import {PoseLandmarker, FilesetResolver} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const W = JSON.parse(document.getElementById('w').textContent);
const $ = id => document.getElementById(id);
const sig = x => 1/(1+Math.exp(-x));

// ---------- LSTM(64) -> Dense(32, relu) -> Dense(1, sigmoid): same weights as the trained Keras model ----------
function lstmPredict(seq){
  const H=64; const h=new Float32Array(H), c=new Float32Array(H);
  for(const x of seq){
    const z=new Float32Array(4*H); for(let j=0;j<4*H;j++) z[j]=W.b[j];
    for(let i=0;i<36;i++){const xi=x[i], row=W.Wk[i]; for(let j=0;j<4*H;j++) z[j]+=xi*row[j];}
    for(let i=0;i<H;i++){const hi=h[i], row=W.Wr[i]; for(let j=0;j<4*H;j++) z[j]+=hi*row[j];}
    for(let j=0;j<H;j++){
      const ig=sig(z[j]), fg=sig(z[H+j]), gg=Math.tanh(z[2*H+j]), og=sig(z[3*H+j]);
      c[j]=fg*c[j]+ig*gg; h[j]=og*Math.tanh(c[j]);
    }
  }
  const d=new Float32Array(32);
  for(let k=0;k<32;k++){let s=W.D1b[k]; for(let i=0;i<H;i++) s+=h[i]*W.D1k[i][k]; d[k]=Math.max(0,s);}
  let o=W.D2b[0]; for(let k=0;k<32;k++) o+=d[k]*W.D2k[k][0];
  return sig(o);
}
const chk = lstmPredict(W.test_x);
$('chk').textContent = 'Model check: browser ' + chk.toFixed(4) + ' vs trained model ' + W.test_p.toFixed(4) + (Math.abs(chk-W.test_p)<1e-3 ? '  ✓ identical' : '  ✗ MISMATCH');

// ---------- settings ----------
let thr=0.5, zoneOn=true, zonePct=75, speed=40, dist=25;
function stopDist(v){ const ms=v/3.6; return ms*1.0 + ms*ms/(2*7); }
function refreshVehicle(){
  $('sv').textContent=speed; $('dv').textContent=dist;
  $('stp').textContent=stopDist(speed).toFixed(1)+' m';
  $('gain').textContent=(speed/3.6).toFixed(1)+' m';
}
$('thr').oninput = e => { thr=parseFloat(e.target.value); $('tv').textContent=thr.toFixed(2); };
$('zoneOn').onchange = e => { zoneOn=e.target.checked; };
$('zone').oninput = e => { zonePct=parseInt(e.target.value); $('zv').textContent=zonePct; };
$('spd').oninput = e => { speed=parseInt(e.target.value); refreshVehicle(); };
$('dst').oninput = e => { dist=parseInt(e.target.value); refreshVehicle(); };
refreshVehicle();

const logLines=[]; function log(t){ logLines.push(t); while(logLines.length>5) logLines.shift(); $('log').textContent=logLines.join('\n'); }
function tag(id,text,cls){ const b=$(id); b.textContent=text; b.className='tag '+(cls||''); }

// ---------- siren ----------
let actx=null, osc=null, sirenTimer=null;
function ensureAudio(){ if(!actx){ actx=new (window.AudioContext||window.webkitAudioContext)(); } if(actx.state==='suspended') actx.resume(); }
function sirenStart(){
  ensureAudio(); if(osc) return;
  osc=actx.createOscillator(); const g=actx.createGain();
  osc.type='sawtooth'; g.gain.value=0.22; osc.connect(g); g.connect(actx.destination); osc.start();
  const sweep=()=>{const t=actx.currentTime; osc.frequency.cancelScheduledValues(t); osc.frequency.setValueAtTime(700,t); osc.frequency.linearRampToValueAtTime(1150,t+0.4); osc.frequency.linearRampToValueAtTime(700,t+0.8);};
  sweep(); sirenTimer=setInterval(sweep,800);
}
function sirenStop(){ if(sirenTimer){clearInterval(sirenTimer); sirenTimer=null;} if(osc){try{osc.stop();}catch(e){} osc=null;} }
$('test').onclick = () => { sirenStart(); setTimeout(sirenStop,1600); };

// ---------- event log ----------
const events=[];
function addEvent(p, th, decision){
  const e={time:new Date().toLocaleTimeString(), p:p.toFixed(2), th:th.toFixed(2), speed, dist, decision};
  events.push(e);
  const tr=document.createElement('tr');
  tr.innerHTML='<td>'+e.time+'</td><td>'+e.p+'</td><td>'+e.th+'</td><td>'+e.decision+'</td>';
  const t=$('evt'); t.insertBefore(tr, t.rows[1]||null);
  while(t.rows.length>7) t.deleteRow(t.rows.length-1);
}
$('csv').onclick = () => {
  const lines=['time,probability,threshold,speed_kmh,distance_m,decision'].concat(events.map(e=>[e.time,e.p,e.th,e.speed,e.dist,e.decision].join(',')));
  const blob=new Blob([lines.join('\n')],{type:'text/csv'}); const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='alert_log.csv'; document.body.appendChild(a); a.click(); a.remove();
};
$('clr').onclick = () => { events.length=0; const t=$('evt'); while(t.rows.length>1) t.deleteRow(1); };

// ---------- graph ----------
const gr=$('graph'), gctx=gr.getContext('2d'); const series=[];
function drawGraph(tn, effThr){
  const w=gr.width, h=gr.height; gctx.clearRect(0,0,w,h);
  gctx.strokeStyle='#1f3b5f'; gctx.lineWidth=1;
  for(const y of [0.25,0.5,0.75]){ gctx.beginPath(); gctx.moveTo(0,h*(1-y)); gctx.lineTo(w,h*(1-y)); gctx.stroke(); }
  gctx.setLineDash([6,4]); gctx.strokeStyle='#ffffff'; gctx.beginPath(); gctx.moveTo(0,h*(1-effThr)); gctx.lineTo(w,h*(1-effThr)); gctx.stroke(); gctx.setLineDash([]);
  if(series.length<2) return;
  gctx.lineWidth=2.5;
  for(let i=1;i<series.length;i++){
    const a=series[i-1], b=series[i];
    gctx.strokeStyle='hsl('+(120*(1-b.p))+',90%,55%)';
    gctx.beginPath(); gctx.moveTo(w-(tn-a.t)/15*w, h*(1-a.p)); gctx.lineTo(w-(tn-b.t)/15*w, h*(1-b.p)); gctx.stroke();
  }
}

// ---------- pose -> model features ----------
const MAP=[0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28];   // MediaPipe landmark -> COCO-17 order
const EDGES=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];
let landmarker=null, running=false, lastVT=-1, hist=[], ema=null, streak=0, alarmOn=false, lastPred=0, stream=null, lastFrameT=0, fpsEma=0, latEma=0;
const video=$('v'), canvas=$('c'), ctx=canvas.getContext('2d');
const pre=document.createElement('canvas'), pctx=pre.getContext('2d');

function norm(x){ return x.map((v,i)=>(v-W.mu[i])/W.sd[i]); }
function sample(tg){
  let a=null,b=null;
  for(let k=hist.length-1;k>=0;k--){ if(hist[k].t<=tg){a=hist[k]; b=hist[k+1]||null; break;} }
  if(!a) return null;
  if(!b) return (tg-a.t<0.15)?a:null;
  if(b.t-a.t>0.3) return null;
  const u=(tg-a.t)/(b.t-a.t);
  return {f:a.f.map((v,i)=>v+(b.f[i]-v)*u), cx:a.cx+(b.cx-a.cx)*u, cy:a.cy+(b.cy-a.cy)*u};
}
function buildSeq(tn){
  const seq=[]; let pcx=null,pcy=null;
  for(let i=0;i<30;i++){
    const s=sample(tn-(29-i)/30); if(!s) return null;
    const dx=pcx===null?0:s.cx-pcx, dy=pcy===null?0:s.cy-pcy;
    seq.push(norm([...s.f,dx,dy])); pcx=s.cx; pcy=s.cy;
  }
  return seq;
}

function setAlarm(on){
  if(on===alarmOn) return; alarmOn=on;
  $('stage').classList.toggle('alarm', on);
  if(on) sirenStart(); else sirenStop();
}
function resetUI(msg){
  hist=[]; ema=null; streak=0; setAlarm(false); series.length=0;
  $('pv').textContent='--'; $('fill').style.width='0%';
  tag('banner',msg,''); tag('dir','MOVEMENT: --',''); $('zst').textContent='--';
  tag('brake','CRUISING','');
}
function drawZone(vw,vh){
  if(!zoneOn) return;
  const lx=vw*(1-zonePct/100), zw=vw*0.18;
  ctx.fillStyle='rgba(255,60,60,0.14)'; ctx.fillRect(lx-zw,0,2*zw,vh);
  ctx.strokeStyle='#ff4d4d'; ctx.lineWidth=3; ctx.setLineDash([14,10]);
  ctx.beginPath(); ctx.moveTo(lx,0); ctx.lineTo(lx,vh); ctx.stroke(); ctx.setLineDash([]);
}

function handle(res, t, ms){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const vw=video.videoWidth, vh=video.videoHeight;
  drawZone(vw,vh);
  const effThrBase = thr;
  const lm = res.landmarks && res.landmarks[0];
  if(!lm){ resetUI('NO PEDESTRIAN DETECTED'); drawGraph(t,thr); return; }
  const vis=i=>(lm[i].visibility===undefined?1:lm[i].visibility);
  if(![11,12,23,24].every(i=>vis(i)>0.5)){ resetUI('LOW VISIBILITY: step back so your full body is in view'); drawGraph(t,thr); return; }
  const P=MAP.map(i=>[lm[i].x*vw, lm[i].y*vh]);
  const xs=P.map(p=>p[0]), ys=P.map(p=>p[1]);
  const minx=Math.min(...xs), maxx=Math.max(...xs), miny=Math.min(...ys), maxy=Math.max(...ys);
  const w0=maxx-minx, h0=maxy-miny;
  if(h0<vh*0.25){ resetUI('PEDESTRIAN TOO FAR: move closer'); drawGraph(t,thr); return; }
  // approximate full-body box, then the same 10% padding crop used in training
  const bx1=minx-0.05*w0, bx2=maxx+0.05*w0, by1=miny-0.08*h0, by2=maxy+0.03*h0;
  const bw=bx2-bx1, bh=by2-by1;
  const cx1=bx1-0.1*bw, cy1=by1-0.1*bh, cw=1.2*bw, ch=1.2*bh;
  const f=[]; for(const p of P){ f.push((p[0]-cx1)/cw, (p[1]-cy1)/ch); }
  const cxn=((bx1+bx2)/2)/vw;
  hist.push({t, f, cx:cxn, cy:((by1+by2)/2)/vh});
  while(hist.length && hist[0].t < t-2.5) hist.shift();

  // danger zone status (data coordinates; the screen is mirrored)
  const lineN=1-zonePct/100;
  const inZone = zoneOn && Math.abs(cxn-lineN) < 0.18;
  $('zst').textContent = !zoneOn ? 'zone off' : (inZone ? 'INSIDE ZONE' : 'outside zone');
  $('zst').style.color = inZone ? '#ff8080' : '#7dffb3';
  const effThr = Math.max(0.05, effThrBase - (inZone ? 0.15 : 0));
  $('thrline').style.left=(effThr*100)+'%';

  // movement direction over the last ~0.5 s
  let dirMsg='● STANDING', dirCls='';
  const old=hist.find(h=>h.t>=t-0.5);
  if(old && t-old.t>0.25){
    const vx=(cxn-old.cx)/(t-old.t);          // frame-widths per second (data coordinates)
    if(Math.abs(vx)>0.06){
      const screenRight = vx<0;               // mirrored display
      const toward = (lineN-cxn)*vx>0;
      dirMsg=(screenRight?'➡ ':'⬅ ')+(toward?'MOVING TOWARD ROAD EDGE':'MOVING AWAY FROM ROAD');
      dirCls=toward?'warn':'ok';
    }
  }
  tag('dir','MOVEMENT: '+dirMsg,dirCls);

  // skeleton coloured by risk
  const risk = ema===null?0:ema;
  const col='hsl('+(120*(1-risk))+',95%,55%)';
  ctx.lineWidth=4; ctx.strokeStyle=col;
  for(const [a,b] of EDGES){ ctx.beginPath(); ctx.moveTo(P[a][0],P[a][1]); ctx.lineTo(P[b][0],P[b][1]); ctx.stroke(); }
  ctx.fillStyle='#ffffff'; for(const p of P){ ctx.beginPath(); ctx.arc(p[0],p[1],5,0,7); ctx.fill(); }
  ctx.lineWidth=2; ctx.strokeStyle=alarmOn?'#ff2d2d':'#35b6ff'; ctx.strokeRect(bx1,by1,bw,bh);

  if(t-lastPred<0.1) { drawGraph(t,effThr); return; } lastPred=t;
  if(hist[0].t > t-1.0){ tag('banner','COLLECTING 1-SECOND WINDOW...',''); drawGraph(t,effThr); return; }
  const seq=buildSeq(t);
  if(!seq){ tag('banner','TRACKING UNSTABLE: hold still in view',''); drawGraph(t,effThr); return; }
  const p=lstmPredict(seq);
  ema = ema===null ? p : 0.6*ema+0.4*p;
  series.push({t, p:ema}); while(series.length && series[0].t<t-15) series.shift();
  $('pv').textContent=ema.toFixed(2); $('fill').style.width=(ema*100)+'%';
  if(ema>=effThr) streak++; else if(ema<effThr-0.05) streak=0;

  const wasOn=alarmOn;
  if(streak>=3){
    setAlarm(true); tag('banner','⚠ ALERT: PEDESTRIAN LIKELY TO CROSS','al');
    const need=stopDist(speed);
    if(need>dist){ tag('brake','🛑 BRAKE NOW: stopping distance '+need.toFixed(0)+' m > '+dist+' m','al'); }
    else { tag('brake','⚠ PREPARE TO BRAKE: '+(dist-need).toFixed(0)+' m margin','warn'); }
    if(!wasOn) addEvent(ema, effThr, need>dist?'BRAKE NOW':'PREPARE TO BRAKE');
  } else {
    setAlarm(false); tag('banner','TRACKING PEDESTRIAN: no crossing intent','ok'); tag('brake','CRUISING','ok');
  }
  drawGraph(t,effThr);

  // performance counters (FPS and latency of the full pipeline)
  const lat=performance.now()-ms; latEma=latEma?0.9*latEma+0.1*lat:lat;
  const dt=t-lastFrameT; lastFrameT=t; if(dt>0&&dt<1){ fpsEma=fpsEma?0.9*fpsEma+0.1*(1/dt):1/dt; }
  $('hud').textContent='FPS '+fpsEma.toFixed(0)+' | latency '+latEma.toFixed(0)+' ms';
}

function loop(){
  if(!running) return;
  if(video.readyState>=2 && video.currentTime!==lastVT){
    lastVT=video.currentTime;
    const ms=performance.now();
    try{
      let src=video;
      if($('night').checked){
        pre.width=video.videoWidth; pre.height=video.videoHeight;
        pctx.filter='brightness(1.9) contrast(1.25)'; pctx.drawImage(video,0,0); src=pre;
      }
      const res=landmarker.detectForVideo(src, ms);
      handle(res, ms/1000, ms);
    }catch(e){ log('error: '+e.message); }
  }
  requestAnimationFrame(loop);
}

$('start').onclick = async () => {
  $('start').disabled=true; logLines.length=0; ensureAudio(); $('log').style.display='block';
  try{
    log('Initializing camera...');
    stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'}, audio:false});
    video.srcObject=stream; await video.play();
    canvas.width=video.videoWidth; canvas.height=video.videoHeight;
    log('Camera online ('+video.videoWidth+'x'+video.videoHeight+')');
    if(!landmarker){
      log('Loading pose model...');
      const fs = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
      const url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
      try{
        landmarker = await PoseLandmarker.createFromOptions(fs,{baseOptions:{modelAssetPath:url,delegate:"GPU"},runningMode:"VIDEO",numPoses:1});
      }catch(e){
        landmarker = await PoseLandmarker.createFromOptions(fs,{baseOptions:{modelAssetPath:url,delegate:"CPU"},runningMode:"VIDEO",numPoses:1});
      }
    }
    log('Pose model ready'); log('LSTM intent model loaded'); log('SYSTEM READY');
    running=true; resetUI('SYSTEM READY: step into view'); $('stop').disabled=false;
    setTimeout(()=>{ if(running) $('log').style.display='none'; },2500);
    loop();
  }catch(e){
    log('Could not start: '+e.message);
    log('Allow the camera (icon in the address bar) and try again, or use the Video Demo tab.');
    $('start').disabled=false;
  }
};
$('stop').onclick = () => {
  running=false; setAlarm(false);
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
  ctx.clearRect(0,0,canvas.width,canvas.height);
  $('log').style.display='block'; logLines.length=0; log('System offline. Press START.');
  resetUI('SYSTEM OFFLINE'); drawGraph(0,thr); $('hud').textContent='FPS -- | latency -- ms';
  $('start').disabled=false; $('stop').disabled=true;
};
</script></body></html>
"""

tab_live, tab_vid = st.tabs(["📷 Live Camera", "🎞 Video Demo"])

with tab_live:
    html = LIVE_HTML.replace("__WEIGHTS__", open("model_weights.json").read())
    components.html(html, height=1420, scrolling=True)

with tab_vid:
    thr = st.slider("Alert sensitivity (threshold)", 0.1, 0.9, 0.5, 0.05, key="thr_video")
    names = [m["title"] for m in meta]
    pick = st.radio("Select a scenario", names, horizontal=True)
    i = names.index(pick)
    left, right = st.columns([3, 2])
    with left:
        st.video(meta[i]["file"])
        st.caption("Green skeleton = pose keypoints, blue box = tracked pedestrian. Sample clip from the JAAD test set.")
    with right:
        p = probs[i]
        st.markdown(f"<div style='color:#9fb6d3;font-size:13px'>CROSSING PROBABILITY</div><div class='big'>{p:.2f}</div>",
                    unsafe_allow_html=True)
        st.progress(min(max(p, 0.0), 1.0))
        if p >= thr:
            st.markdown("<div class='alertbox alert'>⚠ ALERT: PEDESTRIAN LIKELY TO CROSS</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alertbox safe'>✔ NO CROSSING INTENT</div>", unsafe_allow_html=True)
        if st.checkbox("Reveal what actually happened"):
            st.info("The pedestrian DID cross." if meta[i]["label"] == 1 else "The pedestrian did NOT cross.")

st.markdown("<div class='next'>NEXT PHASE &nbsp;➜&nbsp; PIE dataset &nbsp;·&nbsp; Indian road data &nbsp;·&nbsp; CLAHE low-light handling &nbsp;·&nbsp; Raspberry Pi edge deployment</div>",
            unsafe_allow_html=True)
