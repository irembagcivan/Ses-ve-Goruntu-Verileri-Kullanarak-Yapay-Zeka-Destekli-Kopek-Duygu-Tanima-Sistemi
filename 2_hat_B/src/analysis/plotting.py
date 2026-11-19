"""Optional bounded-point plotting; matplotlib is never required for CSV/JSON."""
import csv
import math
from pathlib import Path


def plot_motion(csv_path,output_path,frames,max_points=2000):
    output=Path(output_path)
    if output.exists():return {'status':'output_exists'}
    figure=None;created=False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        step=max(1,math.ceil(frames/max_points))
        times=[];values={name:[] for name in ('global_mean_speed_px_s','head_relative_mean_speed_px_s',
                                              'body_relative_mean_speed_px_s','tail_vector_speed_px_s')}
        with open(csv_path,encoding='utf-8',newline='') as stream:
            for i,row in enumerate(csv.DictReader(stream)):
                if i%step:continue
                times.append(float(row['timestamp_seconds']) if row['timestamp_seconds'] else math.nan)
                for key in values:values[key].append(float(row[key]) if row[key] else math.nan)
        figure,axis=plt.subplots(figsize=(11,5))
        for key,data in values.items():axis.plot(times,data,label=key)
        axis.set(xlabel='Input timestamp (seconds)',ylabel='Speed (pixels/second)',title='Descriptive keypoint motion')
        axis.legend(fontsize=7);figure.tight_layout()
        with output.open('xb') as stream:
            created=True;figure.savefig(stream,format='png',dpi=140)
        return {'status':'created','sample_stride':step,'points':len(times)}
    except Exception:
        if created:output.unlink(missing_ok=True)
        return {'status':'unavailable_or_failed','reason':'optional_plot_backend_unavailable_or_unusable'}
    finally:
        if figure is not None:
            plt.close(figure)
