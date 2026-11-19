// Execute real inline UI code against a minimal DOM to catch startup/state errors.
// Usage: node tests/demo_ui_smoke.cjs TEMPLATE REPORT_JSON
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8'),report=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
class Element {
 constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this.textContent='';this.style={};this.dataset={};this.attributes={};this.currentTime=0;this.duration=12;this.classList={toggle(){}};}
 append(...nodes){this.children.push(...nodes);if(this.tag==='select'&&this.value===''&&nodes.length)this.value=String(nodes[0].value)}
 replaceChildren(...nodes){this.children=[];if(this.tag==='select')this.value='';this.append(...nodes)}
 set id(value){this._id=value;elements[value]=this} get id(){return this._id}
 get options(){return this.children}
 set src(value){this.attributes.src=value} get src(){return this.attributes.src}
 getAttribute(name){return this.attributes[name]||null}removeAttribute(name){delete this.attributes[name]}
 load(){}addEventListener(){}click(){}
 getContext(){return {clearRect(){},fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){}}}
 getBoundingClientRect(){return {left:0,width:900}}
}
const elements={};for(const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"[^>]*>/g))elements[match[2]]=new Element(match[1]);
for(const match of html.matchAll(/<select id="([^"]+)"[^>]*>(.*?)<\/select>/gs)){for(const option of match[2].matchAll(/<option value="([^"]+)"[^>]*>(.*?)<\/option>/gs)){const e=new Element('option');e.value=option[1];e.textContent=option[2];elements[match[1]].append(e)}}
for(const match of html.matchAll(/<script id="([^"]+)" type="application\/json">(.*?)<\/script>/gs)){elements[match[1]]=new Element();elements[match[1]].textContent=match[2]}
elements['demo-data']=new Element();elements['demo-data'].textContent=JSON.stringify(report);
const modes=['visual','audio','fusion','audio_yamnet','fusion_yamnet','fusion_dogevents','cv_selected','neural_visual','neural_audio','neural_fusion','sequence_cnn','sequence_gru','sequence_bigru'];
const buttons=modes.map(mode=>{const e=new Element('button');e.dataset.mode=mode;return e});
const context=vm.createContext({document:{getElementById:id=>{assert(elements[id],`Missing element ${id}`);return elements[id]},createElement:tag=>new Element(tag),querySelectorAll:selector=>selector==='.experiment-tab'?buttons.slice(3,6):selector==='.neural-tab'?buttons.slice(7,10):selector==='.sequence-tab'?buttons.slice(10):buttons},console,Blob,setTimeout,localStorage:{getItem(){return null},setItem(){}},URL:{revokeObjectURL(){},createObjectURL(){return 'blob:test'}},fetch(){throw Error('Unexpected fetch for embedded demo')}});
const script=[...html.matchAll(/<script>(.*?)<\/script>/gs)].map(m=>m[1]).join('\n');
vm.runInContext(script,context);
setImmediate(()=>{
 if(elements.videoSelect){
  assert.equal(elements.videoSelect.options.length,30);assert.equal(elements.progress.value,0);
  assert(vm.runInContext('records.every(r=>r.status==="pending")',context));
  assert(vm.runInContext('records.every(r=>!r.labels.aktivite)',context),'Model suggestion populated human label');
  vm.runInContext('for(const t of Object.keys(targets))$("field_"+t).value=initial.labels[t][0];$("status").value="preview_only";edited()',context);
  assert.equal(elements.progress.value,0);
  vm.runInContext('$("status").value="reviewed_full";edited()',context);assert.equal(elements.progress.value,1);
  const payload=JSON.parse(vm.runInContext('JSON.stringify(compact())',context));assert(!('suggestions' in payload.records[0]));assert(!('preview' in payload.records[0]));
  assert.throws(()=>vm.runInContext('merge({schema_version:1,manifest_digest:"wrong",records:[]})',context));
  elements.next.onclick();assert.equal(Number(elements.videoSelect.value),1);assert.equal(elements.status.value,'pending');
  console.log('Annotation UI smoke passed: 30 videos, no automatic labels, preview/full status, compact save and stale-import rejection.');return
 }
 if(elements.candidates){assert.equal(elements.candidates.children.length,90);elements.target.value='durus';vm.runInContext('draw()',context);assert.equal(elements.candidates.children.length,18);elements.algorithm.value='logistic';vm.runInContext('draw()',context);assert.equal(elements.candidates.children.length,6);assert.equal(elements.candidates.children.filter(r=>r.className==='chosen').length,1);console.log('Experiment UI smoke passed: 90 candidates, target/algorithm filters and CV selection.');return}
 assert.equal(elements.error.textContent,'','Startup failed: '+elements.error.textContent);
 assert.equal(elements.total.textContent,127);assert.equal(elements.cvMetrics.children.length,5);
 if(report.temporal_protocol){
  assert.equal(elements.temporalPanel.hidden,false);
  const windows=report.samples[0].temporal_behavior.windows;
  assert(windows.length>1);
  assert.equal(vm.runInContext('activeTemporalWindow(temporalSample().windows,2).start',context),2);
  elements.video.currentTime=2;vm.runInContext('updateTemporal()',context);
  assert.equal(elements.temporalCurrent.children.length,5);
  elements.video.currentTime=10000;vm.runInContext('updateTemporal()',context);
  assert.equal(elements.temporalCurrent.children.length,0);
  assert(elements.temporalStatus.textContent.includes('penceresi yok'));
  elements.video.currentTime=0;
 }

 for(const mode of modes.filter(m=>report.samples[0].predictions[m])){vm.runInContext(`mode=${JSON.stringify(mode)};display()`,context);assert.equal(elements.behaviors.children.length,5)}
 vm.runInContext("$('sample').value=String(data.samples.length-1);display()",context);
 assert.equal(elements.behaviors.children.length,5);
 elements.matrixMode.value='cv_selected';vm.runInContext('matrix()',context);assert.equal(elements.cm.children.length,1);
 elements.partition.value='test';vm.runInContext('populate()',context);assert.equal(elements.sample.options.length,25);
 vm.runInContext("seekAudio(50)",context);assert(elements.audioStatus.textContent.includes('önizlemenin dışında'));
 console.log('UI smoke passed: available variants, 127-video startup, test filter, selected confusion matrix, out-of-preview seeking.');
});
