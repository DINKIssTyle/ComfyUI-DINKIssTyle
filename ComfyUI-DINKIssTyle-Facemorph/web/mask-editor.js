// Manual corrections are saved as normalized strokes, independent of preview resolution.
export async function editMask({asset,image,popup,button,el,request,url,apply}){
    let edits=structuredClone(asset.mask_edit??{}),mode='exclude',radius=.025,stroke=null,ticket=0,mask=null,dead=false;
    const canvas=el('canvas'),message=el('p',{class:'small',text:'Green: usable face · Pink: protected hair. Paint to correct detection.'});
    const tools=el('div',{class:'bar'}),settings=el('div',{class:'bar'}),actions=el('div',{class:'bar'});
    const panel=popup('Face and hair mask',tools,settings,canvas,message,actions);panel.classList.add('mask-popup');
    panel.addEventListener('close',()=>{dead=true;ticket++;});
    for(const [value,label] of [['include','Include face'],['exclude','Exclude hair'],['restore','Restore']]){
        const b=button(label,()=>{mode=value;sync();});b.dataset.mode=value;tools.append(b);
    }
    function sync(){tools.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));}
    sync();
    const brush=el('input',{type:'range',min:2,max:150,value:25});brush.setAttribute('aria-label','Mask brush size');brush.addEventListener('input',()=>radius=Number(brush.value)/1000);
    tools.append(el('label',{},document.createTextNode('Brush size'),brush));
    for(const [key,label,max,defaultValue] of [['forehead','Forehead inset',100,0],['temple','Temple inset',100,0],['feather','Edge feather',8,1]]){
        const input=el('input',{type:'range',min:0,max,step:key==='feather'?.25:1,value:edits[key]??defaultValue});input.setAttribute('aria-label',label);
        input.addEventListener('change',()=>{edits[key]=Number(input.value);reload().catch(fail);});settings.append(el('label',{},document.createTextNode(label),input));
    }
    canvas.width=Math.round(640*image.width/Math.max(image.width,image.height));canvas.height=Math.round(640*image.height/Math.max(image.width,image.height));
    const ctx=canvas.getContext('2d');
    function draw(){
        ctx.drawImage(image,0,0,canvas.width,canvas.height);
        if(mask){
            const c=document.createElement('canvas');c.width=canvas.width;c.height=canvas.height;const x=c.getContext('2d');x.drawImage(mask,0,0,c.width,c.height);const pixels=x.getImageData(0,0,c.width,c.height);
            for(let i=0;i<pixels.data.length;i+=4){const f=pixels.data[i]/255,h=pixels.data[i+1]/255;pixels.data[i]=h*255;pixels.data[i+1]=f*220;pixels.data[i+2]=h*150+f*100;pixels.data[i+3]=Math.max(f,h)*110;}x.putImageData(pixels,0,0);ctx.drawImage(c,0,0);
        }
        if(stroke){ctx.strokeStyle=mode==='include'?'#63ffc8aa':mode==='exclude'?'#ff68b0bb':'#ffffff88';ctx.lineWidth=radius*2*Math.max(canvas.width,canvas.height);ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();stroke.points.forEach(([x,y],i)=>i?ctx.lineTo(x*(canvas.width-1),y*(canvas.height-1)):ctx.moveTo(x*(canvas.width-1),y*(canvas.height-1)));if(stroke.points.length===1){const [x,y]=stroke.points[0];ctx.lineTo(x*(canvas.width-1)+.01,y*(canvas.height-1));}ctx.stroke();}
    }
    function fail(error){message.textContent=error.message;}
    async function reload(){
        const revision=++ticket;message.textContent='Updating mask…';
        const result=await request('/mask',{asset:{...asset,mask_edit:edits}});
        const next=new Image();next.src=url(result.mask_ids[0]);await next.decode();
        if(dead||revision!==ticket)return;mask=next;draw();message.textContent='Green: usable face · Pink: protected hair. Paint to correct detection.';
    }
    function coords(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];}
    canvas.addEventListener('pointerdown',e=>{if(!mask)return;if((edits.strokes?.length??0)>=200||(edits.strokes??[]).reduce((n,s)=>n+s.points.length,0)>=12000){message.textContent='A photo supports up to 200 mask strokes. Reset corrections to start over.';return;}stroke={mode,radius,points:[coords(e)]};canvas.setPointerCapture(e.pointerId);draw();});
    canvas.addEventListener('pointermove',e=>{if(!stroke||stroke.points.length>=2000||(edits.strokes??[]).reduce((n,s)=>n+s.points.length,0)+stroke.points.length>=12000)return;const p=coords(e),last=stroke.points.at(-1);if(Math.hypot(p[0]-last[0],p[1]-last[1])>.002){stroke.points.push(p);draw();}});
    const end=()=>{if(!stroke)return;(edits.strokes??=[]).push(stroke);stroke=null;reload().catch(fail);};
    canvas.addEventListener('pointerup',end);canvas.addEventListener('pointercancel',end);
    let applying=false;
    panel.addEventListener('cancel',event=>{if(applying)event.preventDefault();});
    const accept=button('Apply',async()=>{
        end();applying=true;panel.querySelectorAll('button,input').forEach(control=>control.disabled=true);
        try{await apply(structuredClone(edits));panel.close();}catch(error){fail(error);}
        finally{applying=false;panel.querySelectorAll('button,input').forEach(control=>control.disabled=false);}
    });
    actions.append(button('Reset corrections',()=>{edits={};settings.querySelectorAll('input').forEach((input,i)=>input.value=i===2?1:0);reload().catch(fail);}),
        button('Cancel',()=>panel.close()),accept);
    draw();try{await reload();}catch(error){fail(error);}
}
