(() => {
  'use strict';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const teacher = [.02, .03, .04, .06, .10, .15, .25, .35];
  const initial = [.30, .24, .16, .10, .07, .05, .04, .04];
  const tokens = ['Let', 'me', 'check', 'again', 'Thus', 'B', 'done', 'EOS'];
  const traces = (compact) => [0, 1, 2].map((i) => `<div class="story-card ${compact ? 'rewritten' : ''}"><b aria-label="${compact ? 'State' : 'Reasoning'} ${i+1}"><span class="trace-long">${compact ? 'State' : 'Reasoning'}</span><span class="trace-short">${compact ? 'S' : 'R'}</span> ${i + 1}</b><div class="trace-lines">${Array.from({length: compact ? 2 : 4}, (_,j) => `<i style="--line:${i + j / 4};width:${[94,72,85,55][j]}%"></i>`).join('')}</div></div>${i < 2 ? `<div class="story-tool"><span>⌘</span><b>Tool ${i + 1}</b></div>` : ''}`).join('');
  const actor = (name, symbol, color) => `<div class="story-actor"><span class="actor-shape ${color}">${symbol}</span><b>${name}</b></div>`;
  const curve = (values, width) => {
    const points = values.map((p, i) => [22 + i * (width - 44) / 7, 172 - p * 380]);
    let d = `M${points[0][0]} ${points[0][1]}`;
    for (let i=1; i<points.length; i++) {
      const a=points[i-1], b=points[i], mid=(a[0]+b[0])/2;
      d += ` C${mid} ${a[1]} ${mid} ${b[1]} ${b[0]} ${b[1]}`;
    }
    return {line:d,area:`${d} L${points.at(-1)[0]} 172 L${points[0][0]} 172 Z`};
  };
  function mount(id, training) {
    const root = document.getElementById(id);
    const stages = ['Rollout', 'Compact', training ? 'Train' : 'Continue'];
    const playId = training ? 'distill-play' : 'method-play';
    root.innerHTML = `<div class="story-toolbar"><div class="story-steps" role="group" aria-label="${training ? 'Training' : 'Method'} animation stages">${stages.map((name,i)=>`<button data-stage="${i}" aria-pressed="${i===0}"><span>${i+1}</span>${name}</button>`).join('')}</div><button class="icon-button" id="${playId}" aria-label="Pause ${training ? 'training' : 'method'} animation">Ⅱ</button></div>
      <div class="story-lane-title">${training ? 'Student rollout · original history H' : 'Build the original history'}</div>
      <div class="story-lane original-lane">${actor(training ? 'Student φ' : 'Model','π','blue')}<span class="story-arrow">→</span><div class="story-history">${traces(false)}</div></div>
      <div class="story-connector"><span>↓</span> rewrite reasoning; keep tools and observations</div>
      <div class="story-lane compact-lane">${actor('Compactor','C','amber')}<span class="story-arrow">→</span><div class="story-history">${traces(true)}</div></div>
      ${training ? `<div class="train-connection"><span class="teacher-context">Compact H꜀ → frozen teacher θ</span><span class="student-context">Original H → trainable student φ</span></div><div class="mass-stage"><div class="mass-heading"><h3>Train the ordinary-context student</h3><span class="toy-label">Illustrative next-token mass</span></div><div class="legend"><span><i class="teacher-key"></i>Frozen teacher</span><span><i class="student-key"></i>Student</span></div><svg class="mass-plot" role="img" aria-label="Illustrative teacher and student token distributions converge during KL training"></svg><div class="loss-line"><span>KL(teacher ∥ student)</span><strong id="kl-value"></strong><b class="weight-update">∇L → update φ</b></div><label class="slider-label" for="match-slider">Student match<input id="match-slider" type="range" min="0" max="100" value="0"></label></div>` : `<div class="story-next"><span class="pill green">Same weights</span><span>→</span><div class="next-thought"><b>Continue from compact history</b><div class="trace-lines"><i></i><i></i><i></i></div></div></div><p class="demo-caption">Illustration of the history intervention. The target gets no new instruction to be brief.</p>`}`;
    const play = root.querySelector(`#${playId}`);
    let playing = !reduced.matches, visible = false, frame = 0, epoch = performance.now(), progress = 0, width = 600;
    let alpha = 0;
    const plot = root.querySelector('.mass-plot');
    function preparePlot() {
      if (!training) return;
      width = Math.max(240, plot.getBoundingClientRect().width);
      plot.setAttribute('viewBox', `0 0 ${width} 210`);
      const p = curve(teacher, width);
      plot.innerHTML = `<line x1="22" x2="${width-22}" y1="172" y2="172" class="mass-baseline"/><path class="teacher-fill" d="${p.area}"/><path class="student-fill"/><path class="teacher-line" d="${p.line}"/><path class="student-line"/>${tokens.map((token,i)=>`<text x="${22+i*(width-44)/7}" y="197" text-anchor="middle">${token}</text>`).join('')}`;
      drawMass(alpha);
    }
    function drawMass(value) {
      if (!training) return;
      const q=initial.map((v,i)=>v*(1-value)+teacher[i]*value);
      const p=curve(q,width);
      plot.querySelector('.student-line').setAttribute('d',p.line);
      plot.querySelector('.student-fill').setAttribute('d',p.area);
      const loss=teacher.reduce((sum,v,i)=>sum+v*Math.log(v/q[i]),0);
      root.querySelector('#kl-value').textContent=Math.max(0,loss).toFixed(3);
      root.querySelector('#match-slider').value=String(value*100);
    }
    function render(value) {
      progress=value;
      const stage=value<.25?0:value<.45?1:2;
      root.dataset.stage=String(stage);
      root.style.setProperty('--rollout', Math.min(1,value/.25));
      root.style.setProperty('--rewrite', Math.max(0,Math.min(1,(value-.25)/.20)));
      root.querySelectorAll('.story-steps button').forEach((button,i)=>button.setAttribute('aria-pressed',String(i===stage)));
      alpha=Math.max(0,Math.min(1,(value-.45)/.45));
      drawMass(alpha);
    }
    function queue() {
      if (frame || !playing || !visible || document.hidden) return;
      frame=requestAnimationFrame((now)=>{ frame=0;render(((now-epoch)%12000)/12000);queue(); });
    }
    function setPlaying(value) {
      playing=value;root.classList.toggle('paused',!value);
      play.textContent=value?'Ⅱ':'▶';play.setAttribute('aria-label',`${value?'Pause':'Play'} ${training?'training':'method'} animation`);
      if (value) { epoch=performance.now()-progress*12000;queue(); }
      else { cancelAnimationFrame(frame);frame=0; }
    }
    play.addEventListener('click',()=>setPlaying(!playing));
    root.querySelectorAll('.story-steps button').forEach((button,i)=>button.addEventListener('click',()=>{setPlaying(false);render([.24,.44,.80][i]);}));
    if (training) {
      root.querySelector('#match-slider').addEventListener('input',(event)=>{setPlaying(false);render(.45+Number(event.target.value)/100*.45);});
      new ResizeObserver(preparePlot).observe(plot);
      preparePlot();
    }
    new IntersectionObserver((entries)=>{visible=entries[0].isIntersecting;if(!visible){cancelAnimationFrame(frame);frame=0;}else queue();},{threshold:.15}).observe(root);
    document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frame);frame=0;}else queue();});
    function motion(){play.disabled=reduced.matches;setPlaying(!reduced.matches);if(reduced.matches)render(.90);}
    reduced.addEventListener('change',motion);
    render(0);motion();root.dataset.storyReady='true';
  }
  mount('hero-story',false);
  mount('training-story',true);
})();
