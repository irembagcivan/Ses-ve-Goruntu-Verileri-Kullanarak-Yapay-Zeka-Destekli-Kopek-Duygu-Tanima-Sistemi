"""Portable local launcher with explicit dependency/checkpoint checks."""
import argparse,importlib.util,json,subprocess,sys,time,urllib.request,webbrowser
from pathlib import Path
from src.demo.release import active_release
MODULES=['torch','numpy','pandas','scipy','sklearn','cv2','ultralytics','deeplabcut','h5py','imageio_ffmpeg']
WEIGHTS=['models/yolov8n.pt','models/yolov8m.pt','models/yamnet.h5','models/yolov8s-seg.pt',
         'models/pose/superanimal_quadruped_rtmpose_s.pt','results/demo_v9/neural_fusion.pt',
         'results/demo_v9/neural_fusion.json']+[f'results/demo_v11/sequence_{kind}{suffix}' for kind in ['cnn','gru','bigru'] for suffix in ['.pt','.json']]

def check(root):
    release=active_release()
    files=WEIGHTS if release['version']=='legacy-v12' else WEIGHTS[:5]+[release['model'],str(Path(release['model']).with_suffix('.json'))]+[str(Path(release['model']).parent/('sequence_'+kind+suffix)) for kind in ['cnn','gru','bigru','lstm_lstm_gru'] for suffix in ['.pt','.json']]
    if release.get('affect_model'):
        files+=[str(Path(release['affect_model']).parent/name) for name in ['model.json','classical.joblib','neural.pt']]
    if release.get('late_fusion'):files.append(release['late_fusion'])
    return {'missing_modules':[name for name in MODULES if importlib.util.find_spec(name) is None],
            'missing_files':[name for name in files if not (root/name).is_file()]}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];status=check(root)
    if args.check_only:print(json.dumps(status));return 1 if any(status.values()) else 0
    if any(status.values()):
        print('Eksik bağımlılıklar: '+', '.join(status['missing_modules']))
        print('Eksik model dosyaları: '+', '.join(status['missing_files']))
        print('Python 3.11 ortamında requirements.txt ve requirements-demo.txt dosyalarını kur. Model ağırlıkları için tam paylaşım paketini kullan.');return 1
    url='http://127.0.0.1:'+str(active_release().get('port',8766))
    def healthy():
        try:
            response=json.load(urllib.request.urlopen(url+'/api/health',timeout=1))
            return response.get('application')=='canine-local-analyzer' and response.get('version',0)>=6 and response.get('annotation_version')==active_release()['version']
        except Exception:return False
    if not healthy():
        logs=root/'results/live_analysis';logs.mkdir(parents=True,exist_ok=True)
        with (logs/'server.log').open('ab') as output:
            options=dict(cwd=root,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
            if sys.platform=='win32':options['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
            else:options['start_new_session']=True
            child=subprocess.Popen([sys.executable,'-m','src.demo.upload_server'],**options)
        for _ in range(40):
            if healthy():break
            if child.poll() is not None:
                print('Analiz servisi başlayamadı. Aynı portta eski uygulama varsa onu kapatıp yeniden başlat. Ayrıntı: '+str(logs/'server.log'))
                return 1
            time.sleep(.25)
        else:
            print('Servis henüz hazır değil. Biraz sonra yeniden START_ANALYSIS dosyasını aç. Günlük: '+str(logs/'server.log'))
            return 1
    print('Analiz servisi hazır: '+url+'/analyze')
    print('Başlatıcı penceresini kapatabilirsin; analiz servisi ayrı çalışır.')
    if not args.no_browser:webbrowser.open(url+'/analyze')
    return 0

if __name__=='__main__':raise SystemExit(main())
