const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('src/demo/upload.html','utf8'),job=JSON.parse(fs.readFileSync('results/affect_v15_clean/upload_ui_fixture.json','utf8'));
const elements={};
class Element{constructor(tag){this.tag=tag;this.children=[];this.value='';this.currentTime=0;this.textContent=''}append(...nodes){this.children.push(...nodes);if(this.tag==='select'&&!this.value&&nodes.length)this.value=nodes[0].value}replaceChildren(...nodes){this.children=[];this.value='';this.append(...nodes)}addEventListener(){}click(){}}
for(const m of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g))elements[m[2]]=new Element(m[1]);
const context=vm.createContext({document:{getElementById:id=>{assert(elements[id],id);return elements[id]},createElement:tag=>new Element(tag)},location:{protocol:'http:'},console,Blob,setTimeout,URL,fetch:async()=>({json:async()=>[]})});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1].replace('"__CANINE_APP_TOKEN__"','"test"'),context);
context.job=job;vm.runInContext('display(job)',context);
assert(elements.affectHeadline.textContent.includes(job.result.audio_affect.label));
assert(elements.affectSupport.textContent.includes('sınırlı'));
assert.equal(elements.affectScores.children[0].children.length,2);
assert.equal(elements.modelChoice.value,'Önerilen karar birleşimi');
assert.equal(elements.behaviors.children[0].children.length,6);
vm.runInContext("current.audio_affect={status:'no_audio',reason:'Ses kanalı yok'};renderAssessment()",context);
assert.equal(elements.affectHeadline.textContent,'Duygu durumu belirlenemedi');
assert.equal(elements.affectScores.children.length,0);
console.log('Affect UI passed: actual six-target job, one main mood estimate, limited support, two affect axes and no-audio abstention.');

// Four-class layout uses actual cached evidence, and exposes uncertainty.
const revised=JSON.parse(JSON.stringify(job.result));
revised.behavior_emotion={status:'human_label_estimate',label:'Öfkeli',class_id:'Angry',decision_status:'uncertain',modality:'audio',visual_coverage:.2,reason:'Görüntü kanıtı yetersiz.',behavior_evidence:{},scores:{Happy:.21,Sad:.26,Angry:.33,Relaxed:.20},audio_evidence:[{event:'Growling',windows:2,peak:.66}]};
context.revised={id:'synthetic-ui-check',name:'Kanıt kontrolü',result:revised};
vm.runInContext('display(revised)',context);
assert.equal(elements.emotionCards.children.length,4);
assert(elements.affectHeadline.textContent.includes('Kararsız'));
assert(elements.affectHeadline.textContent.includes(revised.behavior_emotion.label));
assert(elements.affectSupport.textContent.includes('Yalnız ses'));
assert(elements.evidenceChips.children.some(e=>e.textContent.includes('Hırlama')));
assert(elements.evidenceChips.children.some(e=>e.textContent.includes('Görüntü kanıtı düşük')));
assert(!elements.evidenceChips.children.some(e=>e.textContent.includes('Kuyruk')));
assert(elements.emotionCards.children.every(e=>e.children.length===4));
assert(html.includes('Puanlar doğruluk oranı değildir.'));
console.log('Four-emotion UI passed: four cards, uncertainty, growl evidence, low-coverage warning, no unsupported tail claims.');
