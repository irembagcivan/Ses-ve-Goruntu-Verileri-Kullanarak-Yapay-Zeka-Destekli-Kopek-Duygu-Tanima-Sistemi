"""Export the same measured slide content as vector PDF text/tables with figures."""
from pathlib import Path
from html.parser import HTMLParser
import base64,io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
class Node:
 def __init__(self,tag='',attrs=None):self.tag=tag;self.attrs=dict(attrs or []);self.children=[]
 def text(self):return ''.join(c.text() if isinstance(c,Node) else c for c in self.children)
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.root=Node();self.stack=[self.root]
 def handle_starttag(self,tag,attrs):
  node=Node(tag,attrs);self.stack[-1].children.append(node)
  if tag=='br':node.children=['\n']
  if tag not in ['br','meta','img','input','link','hr']:self.stack.append(node)
 def handle_endtag(self,tag):
  for i in range(len(self.stack)-1,0,-1):
   if self.stack[i].tag==tag:self.stack=self.stack[:i];break
 def handle_data(self,text):self.stack[-1].children.append(text)
def descendants(node,tag):
 result=[]
 for child in node.children:
  if isinstance(child,Node):
   if child.tag==tag:result.append(child)
   result+=descendants(child,tag)
 return result
parser=Parser();parser.feed((ROOT/'docs/hoca_sunumu.html').read_text());slides=descendants(parser.root,'section');font=FontProperties(family='DejaVu Sans')
W,H=13.333,7.5
checks=[]
def wrap(text,width,size):
 lines=[]
 for paragraph in text.split('\n'):
  line=''
  for word in paragraph.split():
   candidate=(line+' '+word).strip();length=TextPath((0,0),candidate,size=size,prop=font).get_extents().width if candidate else 0
   if line and length>width*72:lines.append(line);line=word
   else:line=candidate
  lines.append(line)
 return '\n'.join(lines)
def text(fig,value,x,y,width,size=19,color='#eff5ed',bold=False):
 value=wrap(value,width,size);height=(value.count('\n')+1)*size*1.32/72
 fig.text(x/W,y/H,value,va='top',fontsize=size,color=color,fontfamily='DejaVu Sans',fontweight='bold' if bold else 'normal',linespacing=1.32)
 if y-height<.22:checks.append(('overflow',value[:60],y-height))
 return y-height
svgs=['ablation','curves','activity_confusion']
def picture(fig,node,x,y,width,max_height):
 kind,data=node.attrs['src'].split(',',1);raw=base64.b64decode(data)
 if 'svg' in kind:
  name=next(name for name in svgs if (ROOT/'results/defense_build'/(name+'.svg')).read_bytes()==raw)
  im=Image.open(ROOT/'results/defense_build'/(name+'.png'))
 else:im=Image.open(io.BytesIO(raw))
 ratio=im.width/im.height;height=min(max_height,width/ratio);draw_width=height*ratio
 ax=fig.add_axes([(x+(width-draw_width)/2)/W,(y-height)/H,draw_width/W,height/H]);ax.imshow(im,cmap='gray',vmin=0,vmax=255) if im.mode in ['1','L','I'] else ax.imshow(im);ax.axis('off')
 return y-height

def render_nodes(fig,nodes,x,y,width):
 for node in nodes:
  if not isinstance(node,Node) or node.tag in ['h1','aside']:continue
  if node.tag in ['p','h2']:
   cls=node.attrs.get('class','');size=23 if 'lead' in cls else 14 if 'muted' in cls else 22 if node.tag=='h2' else 18
   y=text(fig,node.text(),x,y,width,size,'#afbfaf' if 'muted' in cls else '#d5be89' if node.tag=='h2' else '#eff5ed',node.tag=='h2')-.18
  elif node.tag=='img':y=picture(fig,node,x,y,width,min(4.45,y-.55))-.12
  elif node.tag=='div' and 'columns' in node.attrs.get('class',''):
   children=[c for c in node.children if isinstance(c,Node)];column=(width-.4)/2;bottom=[]
   for i,child in enumerate(children):bottom.append(render_nodes(fig,child.children if child.tag=='div' else [child],x+i*(column+.4),y,column))
   y=min(bottom)
  elif node.tag=='table':
   rows=descendants(node,'tr');cells=[[c.text() for c in row.children if isinstance(c,Node) and c.tag in ['td','th']] for row in rows]
   n=len(cells[0]);ratios=[.25,.75] if n==2 else [1/n]*n;size=16
   wrapped=[[wrap(value,width*ratios[j]-.16,size) for j,value in enumerate(row)] for row in cells]
   heights=[max(value.count('\n')+1 for value in row)*size*1.3/72+.18 for row in wrapped];height=sum(heights)
   ax=fig.add_axes([x/W,(y-height)/H,width/W,height/H]);ax.axis('off');table=ax.table(cellText=wrapped,colWidths=ratios,cellLoc='left',bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(size)
   for (i,j),cell in table.get_celld().items():
    cell.set_facecolor('#101815');cell.set_edgecolor('#34463b');cell.set_linewidth(.4);cell.set_height(heights[i]/height);cell.set_text_props(color='#d5be89' if i==0 else '#eff5ed',fontfamily='DejaVu Sans');cell.PAD=.04
   y-=height+.22
  elif node.tag=='div':y=render_nodes(fig,node.children,x,y,width)
 return y
out=ROOT/'results/defense_build/canine_sunum.pdf'
with PdfPages(out) as pdf:
 for index,slide in enumerate(slides,1):
  fig=plt.figure(figsize=(W,H),facecolor='#101815');title=next(c for c in slide.children if isinstance(c,Node) and c.tag=='h1')
  y=text(fig,title.text(),.65,H-.5,W-1.3,31,'#b6ef83',True)-.26
  bottom=render_nodes(fig,slide.children,.65,y,W-1.3)
  if bottom<.18:checks.append(('slide_bottom',index,bottom))
  fig.text(.95,.02,str(index)+' / '+str(len(slides)),ha='right',fontsize=10,color='#81978a')
  pdf.savefig(fig,facecolor=fig.get_facecolor());fig.savefig(ROOT/'results/defense_build'/f'slide_{index:02}.png',dpi=110,facecolor=fig.get_facecolor());plt.close(fig)
print('PDF:',out,'pages:',len(slides),'layout checks:',checks)
if checks:raise SystemExit('Fix slide layout before delivery')
