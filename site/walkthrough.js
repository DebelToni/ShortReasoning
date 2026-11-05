(() => {
  'use strict';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const teacher = [.02,.03,.04,.06,.10,.15,.25,.35];
  const initial = [.30,.24,.16,.10,.07,.05,.04,.04];
  const tokens = ['Let','me','check','again','Thus','B','done','EOS'];
  const clamp = (v) => Math.max(0,Math.min(1,v));
  const smooth = (v) => {const p=clamp(v);return p*p*(3-2*p);};
  const positions = [220,475,730];
  function network() {
    let lines='';
    for (const [x1,y1,x2,y2] of [[28,16,72,16],[28,47,72,47],[28,16,72,47],[72,16,28,47]]) lines+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
    return `<svg viewBox="0 0 100 64" aria-hidden="true">${lines}${[16,47].map(y=>[28,50,72].map(x=>`<circle cx="${x}" cy="${y}" r="3.7"/>`).join('')).join('')}</svg>`;
  }
  function model(x,y,label,purple,time,extra='') {
    return `<div class="flow-model ${purple?'compressor-model':'target-model'} ${extra}" style="left:${x}px;top:${y}px" data-time="${time}" data-duration=".45"><div class="network-glyph">${network()}</div><span>${label}</span></div>`;
  }
  function card(x,y,kind,time,duration=1.3) {
    const compact=kind==='compact', output=kind==='output';
    const widths=compact?[94,75,84]:output?[94,72,86]:[94,84,96,74,92,80,94];
    return `<div class="flow-card ${compact?'compact-paper':output?'output-paper':'reasoning-paper'}" style="left:${x}px;top:${y}px" data-time="${time}" data-duration="${duration}" data-kind="${kind}"><b>${compact?'Compact':output?'New reasoning':'Reasoning'}</b><div>${widths.map((w,j)=>`<i class="flow-ink ${!compact&&!output&&![0,3,6].includes(j)?'gray-ink':''}" style="width:${w}px" data-line="${j}"></i>`).join('')}</div></div>`;
  }
  function tool(x,y,time,ghost=false) {
    return `<div class="flow-tool${ghost?' moving-tool':''}" style="left:${x}px;top:${y}px" ${ghost?`data-ghost="${time}"`:`data-time="${time}" data-duration=".35"`}>Tool</div>`;
  }
  function label(number,text,y,time,color='blue') {
    return `<div class="flow-step-heading heading-${color}" style="top:${y}px" data-time="${time}" data-duration=".4"><span>${number}</span>${text}</div>`;
  }
  function probability(x,y,target,time) {
    return `<div class="flow-probability ${target?'teacher-probability':'student-probability'}" style="left:${x}px;top:${y}px" data-time="${time}" data-duration=".7"><b>${target?'pθ(next)':'qφ(next)'}</b><div>${(target?teacher:initial).map((v,i)=>`<i data-bin="${i}" style="height:${v*90}px"></i>`).join('')}</div></div>`;
  }
  function curve(values,width) {
    const points=values.map((v,i)=>[22+i*(width-44)/7,172-v*380]);
    let d=`M${points[0][0]} ${points[0][1]}`;
    for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i],m=(a[0]+b[0])/2;d+=` C${m} ${a[1]} ${m} ${b[1]} ${b[0]} ${b[1]}`;}
    return {line:d,area:`${d} L${points.at(-1)[0]} 172 L${points[0][0]} 172 Z`};
  }
  function mount(id,training) {
    const root=document.getElementById(id),height=training?1020:720,total=training?36:24;
    const playId=training?'distill-play':'method-play';
    const phaseNames=['History','Rewrite',training?'Distill':'Continue'];
    let nodes=`<div class="flow-title">ShortReasoning${training?' · self-distillation':''}</div>`;
    let wires='';
    const wire=(d,color,time,duration=.5,arrow=true,extra='')=>{wires+=`<path class="flow-wire ${extra}" d="${d}" stroke="var(--flow-${color})" ${arrow?`marker-end="url(#${id}-${color}-arrow)"`:''} data-time="${time}" data-duration="${duration}"/>`;};
    nodes+=label(1,'Build the original history',104,0);
    nodes+=model(60,194,training?'Student φ':'Target model',false,0);
    positions.forEach((x,i)=>{
      const t=.5+i*1.65;
      nodes+=card(x,162,'reasoning',t);nodes+=tool(x+166,203,t+1.15);
      wire(`M${i?x-27:158} 227 H${x-5}`,'blue',t-.1);
      wire(`M${x+138} 227 H${x+162}`,'green',t+1.05);
    });
    nodes+=label(2,'Rewrite',315,6,'amber');
    nodes+=model(60,377,'Compressor LLM',true,6.2);
    wire('M158 409 H215','purple',6.5);
    positions.forEach((x,i)=>{
      const t=6.8+i*2.15;
      wires+=`<path class="flow-funnel" d="M${x+33} 301 H${x+101} L${x+67} 362 Z" data-time="${t-.2}" data-duration=".6"/>`;
      nodes+=card(x,374,'compact',t,1.5);
      if(i<2)wire(`M${x+67} 458 C${x+67} 508 ${x+322} 508 ${x+322} 458`,'amber',t+1.4,.9,false,'flow-cache-link');
    });
    nodes+='<div class="flow-tools-note" data-time="12.8" data-duration=".5">Tool messages stay unchanged</div>';
    nodes+=label(3,training?'Frozen teacher on the compact history':'Continue from the compact history',514,14);
    nodes+=model(60,578,training?'Frozen teacher θ':'Target model',false,14.2);
    positions.forEach((x,i)=>{
      const t=14.7+i*1.05;
      nodes+=card(x,574,'compact',t,.55);nodes+=tool(x+166,585,t+.75);
      wire(`M${i?x-27:158} 611 H${x-5}`,'amber',t-.1,.35);
      wire(`M${x+138} 611 H${x+162}`,'green',t+.35,.35);
      nodes+=tool(x+166,203,t+.25,true);
    });
    wire('M962 611 H1053','blue',17.8,.5);
    if(training){
      nodes+=probability(1060,574,true,18.2);
      nodes+=label(4,'Train the original-context student',706,19.2);
      nodes+=model(60,787,'Trainable student φ',false,19.5,'trainable-model');
      positions.forEach((x,i)=>{
        const t=19.7+i*.55;
        nodes+=card(x,750,'reasoning',t,.55);nodes+=tool(x+166,791,t+.25);
        wire(`M${i?x-27:158} 815 H${x-5}`,'blue',t-.05,.25);
        wire(`M${x+138} 815 H${x+162}`,'green',t+.20,.25);
      });
      wire('M962 815 H1053','blue',21.5,.4);
      nodes+=probability(1060,778,false,21.9);
      nodes+='<div class="flow-kl-node" data-time="23" data-duration=".5">KL(pθ ∥ qφ)</div><div class="flow-gradient-note" data-time="24" data-duration=".5">Update φ only</div>';
      wire('M1198 611 H1230 V722 H1197','amber',22.5,.6);
      wire('M1198 815 H1230 V722 H1197','blue',22.7,.6);
      wire('M1142 747 C1142 950 20 950 20 819 H56','purple',24,1,true,'flow-gradient');
    }else{
      nodes+=card(1060,574,'output',18.3,1.3);
      nodes+='<div class="flow-fewer" data-time="19.0" data-duration=".5">Often fewer tokens</div><div class="flow-footnote" data-time="19.7" data-duration=".5">Same target model · No retraining · No new instruction to be brief</div>';
    }
    const markers=['blue','green','purple','amber'].map(c=>`<marker id="${id}-${c}-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="var(--flow-${c})"/></marker>`).join('');
    root.classList.add('gif-recreation');
    root.innerHTML=`<div class="flow-controls"><div class="flow-phases" role="group" aria-label="Animation phases">${phaseNames.map((v,i)=>`<button data-phase="${i}" aria-pressed="${i===0}">${i+1} ${v}</button>`).join('')}</div><div class="flow-actions"><button class="icon-button" id="${playId}" aria-label="Pause animation">Ⅱ</button><button class="icon-button flow-expand" aria-label="Expand animation">⤢</button></div></div><div class="flow-viewport"><div class="flow-sizer"><div class="flow-stage" style="height:${height}px" role="img" aria-label="${training?'The GIF structure extended with compact-context teacher, original-context student and KL updates':'Faithful recreation of the GIF: original reasoning and tools, individual rewrite funnels, reconstructed compact history and new reasoning'}"><svg class="flow-wires" viewBox="0 0 1280 ${height}" aria-hidden="true"><defs>${markers}</defs>${wires}</svg>${nodes}</div></div></div>${training?'<div class="flow-mass"><div class="mass-heading"><h3>Distill the compact behavior</h3><span class="toy-label">Illustrative next-token probabilities</span></div><div class="legend"><span><i class="teacher-key"></i>Frozen teacher · compact history</span><span><i class="student-key"></i>Student · original history</span></div><svg class="mass-plot" role="img" aria-label="The student probability mass aligns with the fixed teacher distribution"></svg><div class="loss-line"><span>KL(teacher ∥ student)</span><strong id="kl-value"></strong></div><label class="slider-label" for="match-slider">Training progress<input id="match-slider" type="range" min="0" max="100" value="0"></label></div>':''}`;
    const stage=root.querySelector('.flow-stage'),sizer=root.querySelector('.flow-sizer'),viewport=root.querySelector('.flow-viewport');
    const elements=[...stage.querySelectorAll('[data-time]')].map(el=>({el,t:Number(el.dataset.time),d:Number(el.dataset.duration),length:el.tagName==='path'?el.getTotalLength():0}));
    const ghosts=[...stage.querySelectorAll('[data-ghost]')];
    elements.filter(o=>o.length).forEach(o=>{o.el.style.strokeDasharray=String(o.length);});
    let time=0,alpha=0,playing=!reduced.matches,visible=false,raf=0,epoch=performance.now(),plotWidth=600;
    const play=root.querySelector(`#${playId}`),plot=root.querySelector('.mass-plot');
    function drawMass(value){
      if(!training)return;
      const q=initial.map((v,i)=>v*(1-value)+teacher[i]*value),p=curve(q,plotWidth);
      plot.querySelector('.student-line').setAttribute('d',p.line);plot.querySelector('.student-fill').setAttribute('d',p.area);
      const loss=teacher.reduce((sum,v,i)=>sum+v*Math.log(v/q[i]),0);
      root.querySelector('#kl-value').textContent=Math.max(0,loss).toFixed(3);root.querySelector('#match-slider').value=String(value*100);
      root.querySelectorAll('.student-probability [data-bin]').forEach((bar,i)=>bar.style.height=`${q[i]*90}px`);
    }
    function fit(){
      const available=viewport.clientWidth,renderWidth=root.classList.contains('is-expanded')?Math.max(950,available):available,scale=renderWidth/1280;
      sizer.style.width=`${renderWidth}px`;sizer.style.height=`${height*scale}px`;stage.style.transform=`scale(${scale})`;
      if(training){
        plotWidth=Math.max(240,plot.getBoundingClientRect().width);plot.setAttribute('viewBox',`0 0 ${plotWidth} 210`);const p=curve(teacher,plotWidth);
        plot.innerHTML=`<line x1="22" x2="${plotWidth-22}" y1="172" y2="172" class="mass-baseline"/><path class="teacher-fill" d="${p.area}"/><path class="student-fill"/><path class="teacher-line" d="${p.line}"/><path class="student-line"/>${tokens.map((v,i)=>`<text x="${22+i*(plotWidth-44)/7}" y="197" text-anchor="middle">${v}</text>`).join('')}`;drawMass(alpha);
      }
    }
    function render(seconds){
      time=seconds;const phase=seconds<6?0:seconds<14?1:2;root.dataset.stage=String(phase);root.dataset.time=seconds.toFixed(2);
      elements.forEach(({el,t,d,length})=>{
        const p=smooth((seconds-t)/d);el.style.opacity=String(p*(el.classList.contains('flow-cache-link') ? 0.5 : 1));el.style.visibility=p===0?'hidden':'visible';
        if(length)el.style.strokeDashoffset=String((1-p)*length);
        const lines=el.querySelectorAll('.flow-ink');lines.forEach((line,j)=>line.style.transform=`scaleX(${clamp(p*(lines.length+1)-j)})`);
        el.classList.toggle('is-writing',p>0&&p<1);
      });
      ghosts.forEach(el=>{const p=(seconds-Number(el.dataset.ghost))/.9;el.style.visibility=p>0&&p<1?'visible':'hidden';el.style.transform=`translateY(${clamp(p)*382}px)`;el.style.opacity=String(p<.15?clamp(p/.15):p>.85?clamp((1-p)/.15):1);});
      root.querySelectorAll('[data-phase]').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===phase)));
      alpha=clamp((seconds-24)/8);root.classList.toggle('is-training',training&&seconds>=24&&seconds<32);drawMass(alpha);
    }
    function queue(){if(raf||!playing||!visible||document.hidden)return;raf=requestAnimationFrame(now=>{raf=0;render(((now-epoch)/1000)%total);queue();});}
    function setPlaying(value){playing=value;root.classList.toggle('paused',!value);play.textContent=value?'Ⅱ':'▶';play.setAttribute('aria-label',`${value?'Pause':'Play'} animation`);if(value){epoch=performance.now()-time*1000;queue();}else{cancelAnimationFrame(raf);raf=0;}}
    play.addEventListener('click',()=>setPlaying(!playing));
    root.querySelectorAll('[data-phase]').forEach((b,i)=>b.addEventListener('click',()=>{setPlaying(false);render([5.8,13.5,training?28:21.5][i]);}));
    if(training)root.querySelector('#match-slider').addEventListener('input',e=>{setPlaying(false);render(24+Number(e.target.value)/100*8);});
    const dialog=document.createElement('dialog');dialog.className='animation-dialog';dialog.setAttribute('aria-label','Expanded animation');document.body.append(dialog);let placeholder;
    root.querySelector('.flow-expand').addEventListener('click',()=>{if(dialog.open){dialog.close();return;}placeholder=document.createElement('div');placeholder.style.height=`${root.offsetHeight}px`;root.replaceWith(placeholder);dialog.append(root);root.classList.add('is-expanded');dialog.showModal();root.querySelector('.flow-expand').textContent='×';root.querySelector('.flow-expand').setAttribute('aria-label','Close expanded animation');fit();});
    dialog.addEventListener('close',()=>{placeholder.replaceWith(root);root.classList.remove('is-expanded');root.querySelector('.flow-expand').textContent='⤢';root.querySelector('.flow-expand').setAttribute('aria-label','Expand animation');fit();});
    let resizeFrame=0;
    new ResizeObserver(()=>{if(!resizeFrame)resizeFrame=requestAnimationFrame(()=>{resizeFrame=0;fit();});}).observe(root.querySelector('.flow-controls'));
    new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(!visible){cancelAnimationFrame(raf);raf=0;}else queue();},{threshold:.1}).observe(root);
    document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(raf);raf=0;}else queue();});
    function motion(){play.disabled=reduced.matches;setPlaying(!reduced.matches);if(reduced.matches)render(training?32:21.5);}
    reduced.addEventListener('change',motion);fit();render(0);motion();root.dataset.storyReady='true';
  }
  mount('hero-story',false);mount('training-story',true);
})();
