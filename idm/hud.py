"""Render camera action predictions over a source video."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .postprocessing import DEFAULT_SHAKE_THRESHOLD, validate_shake_threshold
from .postprocessing import DEFAULT_ROTATION_THRESHOLD, select_rotation_sector, validate_rotation_threshold

# Label geometry: mathematical angles, zero points right and 90 points up.
MOUSE_SECTOR_CENTERS = (0, 45, 90, 135, 180, 225, 270, 315)
MOUSE_SECTOR_HALF_WIDTHS = (10, 35, 10, 35, 10, 35, 10, 35)


def ffmpeg_executable():
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise RuntimeError('HUD export requires ffmpeg on PATH.')
    return executable


def bucket_for_frame(frame_index, sampling):
    """Align bins to the source frames used by inference, including short clips."""
    if 'bucket_start_seconds' in sampling:
        starts = np.asarray(sampling['bucket_start_seconds'], dtype=np.float64)
        if starts.ndim != 1 or not len(starts) or not np.isfinite(starts).all() or np.any(np.diff(starts) <= 0):
            raise ValueError('HUD requires ordered bucket start times.')
        seconds = frame_index / sampling['source_fps']
        return int(np.clip(np.searchsorted(starts, seconds, side='right') - 1, 0, len(starts) - 1))
    indices = np.asarray(sampling['source_frame_indices'], dtype=np.float64)
    if indices.shape != (144,) or not np.isfinite(indices).all() or np.any(np.diff(indices) < 0):
        raise ValueError('HUD requires 144 ordered source frame indices.')
    boundaries = (indices[5:-1:6] + indices[6::6]) / 2
    return int(np.searchsorted(boundaries, frame_index, side='right'))


def draw_hud(frame, sample, frame_index, threshold=0.5):
    import cv2
    if not 0 <= threshold <= 1:
        raise ValueError('HUD threshold must be between 0 and 1.')
    shake_threshold = validate_shake_threshold(sample.get('shake_threshold', DEFAULT_SHAKE_THRESHOLD))
    rotation_threshold = validate_rotation_threshold(sample.get('rotation_threshold', DEFAULT_ROTATION_THRESHOLD))
    index = bucket_for_frame(frame_index, sample['sampling'])
    motion = np.asarray(sample['motion_prob'])[index]
    zoom = np.asarray(sample['zoom_prob'])[index]
    velocity = np.asarray(sample['velocity6'])[index]
    height, width = frame.shape[:2]
    panel = np.zeros((254, 430, 3), dtype=np.uint8)
    panel[:] = (24, 24, 18)
    green, white, muted = (166, 230, 75), (235, 239, 236), (140, 152, 146)

    def text(value, x, y, scale=.40, color=white, thickness=1):
        cv2.putText(panel, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def button(label, x, y, probability, w=33, cutoff=None):
        active = probability >= (threshold if cutoff is None else cutoff)
        cv2.rectangle(panel, (x, y), (x+w, y+23), green if active else (60, 64, 55), -1)
        text(label, x+5, y+16, .34, (22, 30, 20) if active else muted)

    cv2.rectangle(panel, (0, 0), (429, 253), green, 2)
    text('Hy-Nomas IDM', 13, 25, .55, green, 1)
    seconds = frame_index / sample['sampling']['source_fps']
    text(f'{seconds:06.2f}s  BIN {index+1}/{len(sample["motion_prob"])}', 238, 25, .38)
    cv2.line(panel, (13, 38), (416, 38), (67, 78, 68), 1)
    for label, x, y, channel, w in [
        ('W',47,59,0,33), ('A',11,86,1,33), ('S',47,86,2,33), ('D',83,86,3,33),
        ('SP',83,59,12,33), ('UP',29,115,13,39), ('DN',72,115,14,39),
        (f'SHK {motion[15]:.0%}',29,144,15,82), ('CW',29,173,16,39), ('CCW',72,173,17,39),
    ]:
        button(label,x,y,motion[channel],w,shake_threshold if channel == 15 else None)
    center=(182,115)
    top = int(motion[4:12].argmax())
    active = select_rotation_sector(motion[4:12], rotation_threshold)
    for i, (angle, half) in enumerate(zip(MOUSE_SECTOR_CENTERS, MOUSE_SECTOR_HALF_WIDTHS)):
        color = green if i == active else (65, 71, 61)
        cv2.ellipse(panel, center, (50,50), 0, -angle-half, -angle+half, color, -1)
    for angle, half in zip(MOUSE_SECTOR_CENTERS, MOUSE_SECTOR_HALF_WIDTHS):
        boundary = np.deg2rad(-angle-half)
        endpoint = (round(center[0]+50*np.cos(boundary)), round(center[1]+50*np.sin(boundary)))
        cv2.line(panel, center, endpoint, (24,24,18), 1, cv2.LINE_AA)
    cv2.circle(panel, center, 19, (24,24,18), -1)
    cv2.circle(panel, center, 19, muted, 1)
    direction = ('RIGHT','UP-RIGHT','UP','UP-LEFT','LEFT','DOWN-LEFT','DOWN','DOWN-RIGHT')
    text('NO ROTATION' if active < 0 else direction[active],135,184,.34)
    text(f'{motion[4+top]:.1%}',157,202,.4,green if active >= 0 else muted)
    text('ZOOM',250,58,.36,muted)
    button('IN',297,43,zoom[0],48)
    button('OUT',350,43,zoom[1],58)
    text('MODEL VELOCITY',250,86,.34,muted)
    for i, label in enumerate(('X RIGHT','Y DOWN','Z FORWARD','YAW','PITCH','ROLL')):
        text(label,250,104+i*17,.34)
        text(f'{velocity[i]:+8.2f}',341,104+i*17,.34)
    cv2.line(panel, (13, 214), (416, 214), (67,78,68), 1)
    text(f'KEY >= {threshold:.2f} | ROT >= {rotation_threshold:.2f} | SHK >= {shake_threshold:.2f}',13,238,.38,muted)
    scale = min(1., width*.36/430, height*.40/254)
    panel = cv2.resize(panel, (max(1,round(430*scale)),max(1,round(254*scale))))
    ph,pw=panel.shape[:2]; margin=max(2,round(16*scale))
    x,y=width-pw-margin,height-ph-margin
    result=frame.copy()
    result[y:y+ph,x:x+pw]=cv2.addWeighted(result[y:y+ph,x:x+pw],.10,panel,.90,0)
    return result


def render_video(source, output, sample, *, threshold=0.5):
    """Export H.264 MP4 at source resolution/FPS, retaining audio when present."""
    import cv2
    executable = ffmpeg_executable()
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('HUD output must differ from the input video.')
    if not 0 <= threshold <= 1:
        raise ValueError('HUD threshold must be between 0 and 1.')
    shake_threshold = validate_shake_threshold(sample.get('shake_threshold', DEFAULT_SHAKE_THRESHOLD))
    rotation_threshold = validate_rotation_threshold(sample.get('rotation_threshold', DEFAULT_ROTATION_THRESHOLD))
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f'Cannot decode video: {source.name}')
    fps = float(sample['sampling']['source_fps'])
    expected = int(sample['sampling']['source_frames'])
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError('Invalid source FPS')
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='.hud-',dir=output.parent) as temporary:
            temp=Path(temporary)/'video.mp4'
            command=[executable,'-nostdin','-hide_banner','-loglevel','error','-y',
                     '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
                     '-r',str(fps),'-i','pipe:0','-i',str(source),
                     '-map','0:v:0','-map','1:a:0?','-map_metadata','-1',
                     '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
                     '-c:v','libx264','-preset','fast','-crf','20','-threads','2',
                     '-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',
                     '-movflags','+faststart',str(temp)]
            with tempfile.TemporaryFile() as errors:
                encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errors)
                count=0
                try:
                    while True:
                        ok,frame=capture.read()
                        if not ok:break
                        encoder.stdin.write(draw_hud(frame,sample,count,threshold).tobytes())
                        count+=1
                    encoder.stdin.close()
                    status=encoder.wait()
                    if status:
                        errors.seek(0)
                        raise RuntimeError('HUD encoding failed: '+errors.read().decode(errors='replace'))
                    if count != expected:
                        raise RuntimeError(f'HUD decoded {count} frames; inference recorded {expected}.')
                    os.replace(temp,output)
                finally:
                    if encoder.poll() is None:
                        encoder.kill()
                        encoder.wait()
    finally:
        capture.release()
    return {'file':output.name,'frames':count,'fps':fps,'threshold':threshold,
            'shake_threshold':shake_threshold, 'rotation_threshold':rotation_threshold}
