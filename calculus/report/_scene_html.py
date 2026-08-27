"""The HTML/WebGL shell for render_3d.py, kept separate so the renderer stays
readable and so the template can be edited without touching the geometry code.

Self-contained on purpose: no CDN, no external font, no fetch. One file that
opens anywhere, including inside an artifact viewer with a strict CSP.
"""

TEMPLATE = r"""<title>__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  /* SINGLE THEME BY COMMITMENT, not omission. Every radiology workstation is
     dark, and a stone is read as a bright object against a dark ground -- a
     light variant of this scene would invert the one convention its readers
     rely on. So the palette is painted explicitly rather than inherited, and
     the page holds on either host ground. The neutral is biased toward the
     kidney blue so it reads as chosen, not as default grey. */
  :root{
    --bg:#0b0f16; --panel:rgba(18,24,34,.82); --line:rgba(255,255,255,.10);
    --ink:#e8eef7; --dim:#93a3b8; --accent:#7cc0ff;
    --sans:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    /* Plex Mono for every measured quantity: it was cut for instrument and
       data contexts and its figures are genuinely tabular, which the HUD needs
       because the numbers change as the scene spins. */
    --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 700px at 50% -10%,#1a2740,#0b0f16 70%);
       color:var(--ink);overflow:hidden;
       font:400 14px/1.5 var(--sans)}
  canvas{display:block;width:100vw;height:100vh;cursor:grab}
  canvas:active{cursor:grabbing}
  .hud{position:fixed;pointer-events:none;user-select:none}
  #head{top:0;left:0;right:0;padding:18px 22px;
        background:linear-gradient(#0b0f16dd,#0b0f1600)}
  #head h1{margin:0;font-size:17px;font-weight:600;letter-spacing:-.15px;
           text-wrap:balance}
  #head .sub{color:var(--dim);font-size:12px;margin-top:4px;
             font-family:var(--mono);font-variant-numeric:tabular-nums}
  #stats{position:fixed;top:76px;left:22px;display:flex;gap:8px;flex-wrap:wrap;
         max-width:44vw}
  .chip{background:var(--panel);border:1px solid var(--line);border-radius:9px;
        padding:7px 11px;backdrop-filter:blur(8px)}
  .chip b{font-size:16px;font-family:var(--mono);font-weight:500;
          font-variant-numeric:tabular-nums;letter-spacing:-.2px}
  .chip span{color:var(--dim);font-size:10px;display:block;
             text-transform:uppercase;letter-spacing:.6px}
  #legend{position:fixed;right:22px;top:76px;width:196px;background:var(--panel);
          border:1px solid var(--line);border-radius:11px;padding:13px;
          backdrop-filter:blur(8px)}
  #legend h2{margin:0 0 9px;font-size:10px;color:var(--dim);
             text-transform:uppercase;letter-spacing:.7px}
  .row{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px}
  .sw{width:11px;height:11px;border-radius:3px;flex:none}
  .ramp{height:9px;border-radius:5px;margin:8px 0 3px;
        background:linear-gradient(90deg,#ffd479,#ff9d4d,#ff5f52,#c62f6d)}
  .rampx{display:flex;justify-content:space-between;color:var(--dim);
         font-size:10px;font-family:var(--mono);font-variant-numeric:tabular-nums}
  #foot{bottom:0;left:0;right:0;padding:13px 22px;color:var(--dim);font-size:11px;
        background:linear-gradient(#0b0f1600,#0b0f16e6)}
  #tip{position:fixed;pointer-events:none;background:#0f1621f5;
       border:1px solid var(--line);border-radius:9px;padding:8px 11px;
       font-size:12px;display:none;box-shadow:0 10px 28px #0009;z-index:9}
  #tip b{color:var(--accent);font-family:var(--mono);font-weight:500}
  #tip{font-variant-numeric:tabular-nums}
  #ctl{position:fixed;left:22px;bottom:44px;display:flex;gap:7px;z-index:8}
  button{background:var(--panel);color:var(--ink);border:1px solid var(--line);
         border-radius:8px;padding:6px 11px;font:inherit;font-size:12px;
         cursor:pointer;backdrop-filter:blur(8px)}
  button:hover{border-color:var(--accent);color:var(--accent)}
  button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  button[aria-pressed=false]{opacity:.45}
  @media (max-width:720px){
    #legend{display:none} #stats{max-width:92vw}
  }
</style>
<canvas id="c"></canvas>
<div class="hud" id="head">
  <h1>__HEADING__</h1>
  <div class="sub">__SUBTITLE__</div>
</div>
<div id="stats">__STATS__</div>
<div id="legend">
  <h2>Anatomy</h2>
  <div class="row"><span class="sw" style="background:#5a9ef2"></span>Kidney</div>
  <div class="row"><span class="sw" style="background:#fac947"></span>Bladder</div>
  <div class="row"><span class="sw" style="background:#4a5566"></span>Bone &mdash; orientation</div>
  <h2 style="margin-top:14px">Calculus density</h2>
  <div class="ramp"></div>
  <div class="rampx"><span>200 HU</span><span>1500+</span></div>
  <div class="row" style="margin-top:11px;color:var(--dim);font-size:11px;
       line-height:1.45">Sphere diameter is the measured size. Hover a stone
       for its numbers.</div>
</div>
<div id="ctl">
  <button id="bSpin" aria-pressed="true" type="button">Spin</button>
  <button id="bBone" aria-pressed="true" type="button">Bone</button>
  <button id="bOrgan" aria-pressed="true" type="button">Organs</button>
  <button id="bReset" type="button">Reset view</button>
</div>
<div class="hud" id="foot">__FOOT__</div>
<div id="tip"></div>
<script>
const DATA = __DATA__;

const M4={
  persp:(f,a,n,fa)=>{const t=1/Math.tan(f/2);return[t/a,0,0,0, 0,t,0,0,
    0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0];},
  mul:(a,b)=>{const o=new Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){
    let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;},
  ident:()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
  trans:(x,y,z)=>[1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1],
  rotY:a=>{const c=Math.cos(a),s=Math.sin(a);return[c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1];},
  rotX:a=>{const c=Math.cos(a),s=Math.sin(a);return[1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1];},
};
const normalMat=m=>[m[0],m[4],m[8], m[1],m[5],m[9], m[2],m[6],m[10]];

const cv=document.getElementById('c');
const gl=cv.getContext('webgl',{antialias:true,alpha:false});
if(!gl){document.body.innerHTML=
  '<p style="padding:2rem;font:15px system-ui;color:#e8eef7">'+
  'This view needs WebGL, which this browser has disabled.</p>';}

const VS=`
attribute vec3 aPos; attribute vec3 aNrm;
uniform mat4 uMVP, uMV; uniform mat3 uN;
varying vec3 vN, vE;
void main(){ vN=normalize(uN*aNrm);
  vec4 p=uMV*vec4(aPos,1.0); vE=normalize(-p.xyz);
  gl_Position=uMVP*vec4(aPos,1.0); }`;
const FS=`
precision highp float;
uniform vec3 uCol; uniform float uAlpha, uRim;
varying vec3 vN, vE;
void main(){
  vec3 n=normalize(vN), e=normalize(vE);
  vec3 L=normalize(vec3(0.45,0.75,0.90));
  float d=max(dot(n,L),0.0);
  float fres=pow(1.0-max(dot(n,e),0.0), 2.4);
  vec3 c=uCol*(0.34+0.66*d)+vec3(0.62,0.80,1.0)*fres*uRim;
  float a=clamp(uAlpha+fres*uRim*0.85, 0.0, 1.0);
  gl_FragColor=vec4(c,a); }`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
  if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(o);return o;}
const prog=gl.createProgram();
gl.attachShader(prog,sh(gl.VERTEX_SHADER,VS));
gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(prog);gl.useProgram(prog);
const A={pos:gl.getAttribLocation(prog,'aPos'),nrm:gl.getAttribLocation(prog,'aNrm')};
const U={};['uMVP','uMV','uN','uCol','uAlpha','uRim'].forEach(k=>U[k]=gl.getUniformLocation(prog,k));
const bigIdx=!!gl.getExtension('OES_element_index_uint');

function dec(b,T){const s=atob(b),u=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new T(u.buffer);}

function normalsFor(v,f){
  const n=new Float32Array(v.length);
  for(let i=0;i<f.length;i+=3){
    const a=f[i]*3,b=f[i+1]*3,c=f[i+2]*3;
    const e1x=v[b]-v[a],e1y=v[b+1]-v[a+1],e1z=v[b+2]-v[a+2];
    const e2x=v[c]-v[a],e2y=v[c+1]-v[a+1],e2z=v[c+2]-v[a+2];
    const nx=e1y*e2z-e1z*e2y, ny=e1z*e2x-e1x*e2z, nz=e1x*e2y-e1y*e2x;
    n[a]+=nx;n[a+1]+=ny;n[a+2]+=nz; n[b]+=nx;n[b+1]+=ny;n[b+2]+=nz;
    n[c]+=nx;n[c+1]+=ny;n[c+2]+=nz;
  }
  for(let i=0;i<n.length;i+=3){const L=Math.hypot(n[i],n[i+1],n[i+2])||1;
    n[i]/=L;n[i+1]/=L;n[i+2]/=L;}
  return n;
}
function mesh(verts,faces,col,alpha,rim,group,meta){
  const idx = (bigIdx||verts.length/3<65535) ? faces : new Uint16Array(faces);
  const type = (idx instanceof Uint32Array) ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
  const n=normalsFor(verts,faces);
  const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);
  gl.bufferData(gl.ARRAY_BUFFER,verts,gl.STATIC_DRAW);
  const nb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,nb);
  gl.bufferData(gl.ARRAY_BUFFER,n,gl.STATIC_DRAW);
  const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,idx,gl.STATIC_DRAW);
  return {vb,nb,ib,type,count:idx.length,col,alpha,rim,group,meta};
}
function sphere(cx,cy,cz,r,seg){
  const V=[],F=[];
  for(let i=0;i<=seg;i++){const th=i*Math.PI/seg;
    for(let j=0;j<=seg*2;j++){const ph=j*Math.PI/seg;
      V.push(cx+r*Math.sin(th)*Math.cos(ph), cy+r*Math.cos(th),
             cz+r*Math.sin(th)*Math.sin(ph));}}
  const W=seg*2+1;
  for(let i=0;i<seg;i++)for(let j=0;j<seg*2;j++){
    const a=i*W+j,b=a+W;F.push(a,b,a+1,b,b+1,a+1);}
  return [new Float32Array(V),new Uint32Array(F)];
}
function huColour(hu){
  const t=Math.max(0,Math.min(1,(hu-200)/1300));
  const s=[[1,.83,.47],[1,.62,.30],[1,.37,.32],[.78,.18,.43]];
  const x=t*(s.length-1),i=Math.min(s.length-2,Math.floor(x)),k=x-i;
  return s[i].map((v,n)=>v+(s[i+1][n]-v)*k);
}

const meshes=[];
if(gl){
  for(const s of DATA.surfaces)
    meshes.push(mesh(dec(s.v,Float32Array),dec(s.f,Uint32Array),
                     s.col,s.alpha,s.rim,s.group,null));
  for(const st of DATA.stones){
    const [v,f]=sphere(st.c[0],st.c[1],st.c[2],Math.max(st.d/2,1.2),12);
    meshes.push(mesh(v,f,huColour(st.hu),1.0,0.35,'stone',st));
  }
}

// The ambient spin is the page's only animation. A reader who has asked the
// operating system for reduced motion gets a still scene they can still drag --
// the information is unchanged, only the autoplay goes.
const REDUCED = window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let rotY=0.55, rotX=-0.22, dist=DATA.radius*3.1, spin=!REDUCED, drag=null;
const show={organ:true,bone:true,stone:true};
cv.addEventListener('mousedown',e=>{drag=[e.clientX,e.clientY];spin=false;
  document.getElementById('bSpin').setAttribute('aria-pressed','false');});
window.addEventListener('mouseup',()=>drag=null);
window.addEventListener('mousemove',e=>{
  if(!drag)return;
  rotY+=(e.clientX-drag[0])*0.008;
  rotX=Math.max(-1.45,Math.min(1.45,rotX+(e.clientY-drag[1])*0.008));
  drag=[e.clientX,e.clientY];});
cv.addEventListener('wheel',e=>{e.preventDefault();
  dist=Math.max(DATA.radius*1.25,Math.min(DATA.radius*9,
       dist*Math.exp(e.deltaY*0.0011)));},{passive:false});
const tgl=(id,k)=>{const b=document.getElementById(id);
  b.onclick=()=>{show[k]=!show[k];b.setAttribute('aria-pressed',show[k]);};};
tgl('bBone','bone');tgl('bOrgan','organ');
document.getElementById('bSpin').onclick=e=>{spin=!spin;
  e.target.setAttribute('aria-pressed',spin);};
document.getElementById('bReset').onclick=()=>{rotY=0.55;rotX=-0.22;
  dist=DATA.radius*3.1;};

let VP=null;
function project(c){
  if(!VP)return null;
  const x=c[0]-DATA.centre[0],y=c[1]-DATA.centre[1],z=c[2]-DATA.centre[2];
  const w=VP[3]*x+VP[7]*y+VP[11]*z+VP[15];
  if(w<=0)return null;
  return [((VP[0]*x+VP[4]*y+VP[8]*z+VP[12])/w*0.5+0.5)*cv.clientWidth,
          (0.5-(VP[1]*x+VP[5]*y+VP[9]*z+VP[13])/w*0.5)*cv.clientHeight];
}
const tip=document.getElementById('tip');
cv.addEventListener('mousemove',e=>{
  let best=null,bd=28;
  for(const m of meshes){
    if(m.group!=='stone')continue;
    const p=project(m.meta.c); if(!p)continue;
    const d=Math.hypot(p[0]-e.clientX,p[1]-e.clientY);
    if(d<bd){bd=d;best=m.meta;}
  }
  if(best){
    const side=best.side?best.side[0].toUpperCase()+best.side.slice(1)+' ':'';
    tip.innerHTML='<b>'+best.d+' mm</b> &nbsp; '+best.hu+' HU<br>'+side+
      best.organ.replace(/_/g,' ')+(best.zone?' &middot; '+best.zone:'');
    tip.style.display='block';
    tip.style.left=(e.clientX+15)+'px';tip.style.top=(e.clientY+15)+'px';
  } else tip.style.display='none';
});

function draw(){
  const dpr=Math.min(window.devicePixelRatio||1,2);
  const w=cv.clientWidth,h=cv.clientHeight;
  if(cv.width!==w*dpr||cv.height!==h*dpr){cv.width=w*dpr;cv.height=h*dpr;}
  gl.viewport(0,0,cv.width,cv.height);
  gl.clearColor(0.043,0.059,0.086,1);
  gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  if(spin)rotY+=0.0032;

  const P=M4.persp(50*Math.PI/180,w/h,Math.max(dist*0.05,1),dist*4+DATA.radius*4);
  let MV=M4.trans(-DATA.centre[0],-DATA.centre[1],-DATA.centre[2]);
  MV=M4.mul(M4.rotY(rotY),MV);
  MV=M4.mul(M4.rotX(rotX),MV);
  MV=M4.mul(M4.trans(0,0,-dist),MV);
  const MVP=M4.mul(P,MV); VP=MVP;
  const nm=new Float32Array(normalMat(MV));

  const pass=(want,depthWrite)=>{
    gl.depthMask(depthWrite);
    for(const m of meshes){
      if(m.group!==want||!show[m.group])continue;
      gl.bindBuffer(gl.ARRAY_BUFFER,m.vb);
      gl.enableVertexAttribArray(A.pos);
      gl.vertexAttribPointer(A.pos,3,gl.FLOAT,false,0,0);
      gl.bindBuffer(gl.ARRAY_BUFFER,m.nb);
      gl.enableVertexAttribArray(A.nrm);
      gl.vertexAttribPointer(A.nrm,3,gl.FLOAT,false,0,0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,m.ib);
      gl.uniformMatrix4fv(U.uMVP,false,new Float32Array(MVP));
      gl.uniformMatrix4fv(U.uMV,false,new Float32Array(MV));
      gl.uniformMatrix3fv(U.uN,false,nm);
      gl.uniform3fv(U.uCol,new Float32Array(m.col));
      gl.uniform1f(U.uAlpha,m.alpha);
      gl.uniform1f(U.uRim,m.rim);
      gl.drawElements(gl.TRIANGLES,m.count,m.type,0);
    }
  };
  // Stones opaque FIRST so they own the depth buffer, then the shells with
  // depth-write off. That keeps a stone visible THROUGH the organ containing
  // it, which is the entire reason for drawing this in 3D.
  pass('stone',true);
  pass('bone',false);
  pass('organ',false);
  requestAnimationFrame(draw);
}
if(gl){
  if(REDUCED) document.getElementById('bSpin').setAttribute('aria-pressed','false');
  draw();
}
</script>
"""
