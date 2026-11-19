// Execute the actual gallery script with real final-label results.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('src/demo/labels_gallery.html','utf8');
const report=JSON.parse(fs.readFileSync('results/demo_v13_release/report.json','utf8'));
const day=JSON.parse(fs.readFileSync('results/day_validation_v14/report.json','utf8'));
report.day_validation={protocol:day.protocol,models:day.models};
const elements={};
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.value='';this.currentTime=0;this.textContent=''}
 append(...nodes){this.children.push(...nodes);if(this.tag==='select'&&!this.value&&nodes.length)this.value=nodes[0].value}
 replaceChildren(...nodes){this.children=[];this.value='';this.append(...nodes)}
 addEventListener(){}click(){}
}
for(const m of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g))elements[m[2]]=new Element(m[1]);
elements.filter.value='all';
const context=vm.createContext({location:{protocol:'http:',href:'http://127.0.0.1:8768/gallery'},document:{getElementById:id=>{assert(elements[id],id);return elements[id]},createElement:tag=>new Element(tag)},console,Blob,setTimeout,URL:{createObjectURL(){return 'blob:test'},revokeObjectURL(){}}});
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1].replace('/*REPORT_DATA*/',JSON.stringify(report));
vm.runInContext(script,context);
assert.equal(elements.videoChoice.children.length,127);
assert.equal(elements.behaviors.children.length,6);
assert.equal(elements.modelChoice.children.length,11);
assert.equal(elements.metrics.children.length,11);
elements.openUpload.onclick();assert.equal(elements.uploadFrame.src,'/analyze');assert.equal(elements.uploadFrame.hidden,false);
assert.equal(elements.dayPanel.hidden,false);
assert.equal(elements.dayMetrics.children.length,11);
assert(elements.dayProtocol.textContent.includes('103'));
elements.filter.value='new';elements.filter.onchange();assert.equal(elements.videoChoice.children.length,30);
elements.filter.value='test';elements.filter.onchange();assert.equal(elements.videoChoice.children.length,25);
elements.filter.value='excluded';elements.filter.onchange();assert.equal(elements.videoChoice.children.length,9);
assert(elements.quality.textContent.includes('Güvensiz'));
assert(elements.behaviors.children.every(row=>row.children[1].textContent==='Değerlendirme dışı'));
for(const m of report.models){elements.modelChoice.value=m.model;elements.modelChoice.onchange();assert.equal(elements.behaviors.children.length,6)}
assert(!elements.behaviors.children.some(row=>row.children[0].textContent==='Vücut'));
// Compile the actual upload UI with its server-injected token.
const upload=fs.readFileSync('src/demo/upload.html','utf8');
new vm.Script(upload.match(/<script>([\s\S]*?)<\/script>/)[1].replace('"__CANINE_APP_TOKEN__"','"test-token"'));
// Opening the actual template from disk must explain the launcher, not crash.
const uploadElements={};for(const m of upload.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g))uploadElements[m[2]]=new Element(m[1]);
const offline=vm.createContext({document:{getElementById:id=>{assert(uploadElements[id],id);return uploadElements[id]},createElement:tag=>new Element(tag)},location:{protocol:'file:',href:'file:///tmp/upload.html'},console,Blob,setTimeout,URL,fetch(){throw Error('Offline template must not issue API requests')}});
vm.runInContext(upload.match(/<script>([\s\S]*?)<\/script>/)[1],offline);
assert.equal(uploadElements.file.disabled,true);assert(uploadElements.status.textContent.includes('START_ANALYSIS.command'));
uploadElements.analyze.onclick();assert.equal(vm.runInContext('location.href',offline),'http://127.0.0.1:8768/analyze');
console.log('Passed: 127 videos, 30 newly labelled, 25 preserved test files, unsafe masks, 11 models, six tasks, upload JS syntax.');
