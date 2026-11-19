const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('src/demo/upload.html','utf8'),elements={},storage=new Map(),timers=[];
class Element{constructor(){this.children=[];this.value='';this.textContent='';this.hidden=false}append(...n){this.children.push(...n)}replaceChildren(...n){this.children=n}addEventListener(){}}
for(const m of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g))elements[m[2]]=new Element();
let failJob=true,session='fresh-session';
const requests=[];
const context=vm.createContext({document:{getElementById:id=>elements[id],createElement:()=>new Element()},location:{protocol:'http:'},console,Blob,URL,
 localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},clearTimeout(){},setTimeout(fn){timers.push(fn);return timers.length},
 fetch:async(path,options)=>{requests.push([path,options]);if(path==='/api/session')return {ok:true,json:async()=>({token:session})};if(path==='/api/jobs')return {ok:true,json:async()=>[]};if(path.startsWith('/api/jobs/')){if(failJob)throw Error('Server disconnected');return {ok:true,status:200,json:async()=>({status:'failed',error:'Previous incomplete analysis'})}}throw Error(path)}});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
(async()=>{
 const key='a'.repeat(32);context.key=key;
 await vm.runInContext('poll(key)',context);
 assert(elements.connection.textContent.includes('Bağlantı kesildi'));
 assert.equal(storage.get('canine-last-job'),key);
 assert.equal(timers.length,1);
 failJob=false;
 await vm.runInContext('reconnect()',context);
 assert.equal(vm.runInContext('TOKEN',context),'fresh-session');
 await new Promise(resolve=>setImmediate(resolve));
 assert(elements.connection.textContent.includes('Bağlı'));
 assert.equal(elements.status.textContent,'Previous incomplete analysis');
 assert.equal(elements.analyze.disabled,false);
 assert.equal(requests.filter(([p])=>p.startsWith('/api/jobs?')).length,0);
 console.log('Reconnect UI passed: same job retained, fresh session acquired, no duplicate upload, incomplete job reported honestly.');
})().catch(e=>{console.error(e);process.exitCode=1});
