import { app } from '../../scripts/app.js';
import { openEditor } from './editor.js';
app.registerExtension({
    name:'DINKIssTyle.Facemorph',
    async beforeRegisterNodeDef(nodeType,nodeData){
        if(nodeData.name!=='DKST_FaceMorphStudio')return;
        const created=nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated=function(){
            const result=created?.apply(this,arguments);
            const project=this.widgets.find(w=>w.name==='project');
            project.computeSize=()=>[0,-4];project.type='converted-widget';
            if(project.inputEl)project.inputEl.style.display='none';
            this.addWidget('button','Open Face Morph Studio',null,()=>openEditor(this).catch(e=>window.alert(e.message)),{serialize:false});
            this.setSize([330,160]);return result;
        };
        const removed=nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved=function(){this.dkstMorphClose?.();return removed?.apply(this,arguments);};
    }
});
