"""Raw-media extraction matching the pinned M2/M3 source, with timing and provenance."""
import hashlib
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / 'vendor/visual/handcrafted'))
from .setup_assets import ASSETS, REVISION, WEIGHT_SHA, sha

ADAPTER = PROJECT / 'multimodal_fusion/data/final_delivery_20261027/upstream_audio_best.pt'
ADAPTER_SHA = '0db8f833f3d32f21b676c7ed008acdf094bb20dda31e3498e642249d597f04bb'
_audio_model = None

def readiness():
    missing = [m for m in ('transformers', 'librosa', 'soundfile', 'mediapipe', 'imageio_ffmpeg')
               if importlib.util.find_spec(m) is None]
    files = [p.name for p in [ASSETS/'config.json', ASSETS/'preprocessor_config.json', ASSETS/'pytorch_model.bin', ADAPTER] if not p.exists()]
    return {'ready': not missing and not files, 'missing_packages': missing, 'missing_assets': files}

class AudioExtractor:
    def __init__(self):
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor
        from .vendor.audio.learned.research.speech_transfer.refinement.models import build
        if sha(ASSETS/'pytorch_model.bin') != WEIGHT_SHA or sha(ADAPTER) != ADAPTER_SHA:
            raise ValueError('Audio weight SHA256 does not match the final delivery.')
        saved = torch.load(ADAPTER, map_location='cpu', weights_only=True)
        self.adapter = build(saved['config'], saved['state_dict']['mean'], saved['state_dict']['std']).eval()
        self.adapter.load_state_dict(saved['state_dict'], strict=True)
        self.backbone = WavLMModel.from_pretrained(ASSETS, local_files_only=True).eval().requires_grad_(False)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(ASSETS, local_files_only=True)
        if saved['upstream']['revision'] != REVISION: raise ValueError('WavLM revision mismatch')

    @torch.inference_mode()
    def extract(self, wav):
        import librosa
        y, _ = librosa.load(wav, sr=16000, mono=True)
        original_samples = len(y)
        y, intervals = librosa.effects.trim(y, top_db=35)
        if len(y) < 400 or not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-7:
            raise ValueError('The audio track is empty, silent, or too short for audio fusion.')
        # Match M2: processor settings are loaded unchanged; cache float16 round trip.
        values = self.processor(y, sampling_rate=16000, return_tensors='pt')
        hidden = self.backbone(**values, output_hidden_states=True).hidden_states[6][0].half().float()
        lengths = torch.tensor([len(hidden)])
        sequence, pooled, _ = self.adapter.representations(hidden[None], lengths)
        logits = self.adapter.head(pooled)[0]
        # CNN token centres: receptive field 400 samples and stride 320 for this checkpoint.
        hop, receptive = 1, 1
        for kernel, stride in zip(self.backbone.config.conv_kernel, self.backbone.config.conv_stride):
            receptive += (kernel - 1) * hop; hop *= stride
        times = (int(intervals[0]) + (receptive-1)/2 + np.arange(len(hidden))*hop)/16000
        return {'audio_pooled': pooled.numpy(), 'audio_sequence': sequence.numpy(),
                'audio_lengths': np.array([len(hidden)], np.int64),
                'audio_probs': logits.softmax(-1).numpy()[None],
                'audio_times': times.tolist(),
                'audio_info': {'sample_rate': 16000, 'original_duration_sec': original_samples/16000,
                    'trim_start_sec': int(intervals[0])/16000, 'trim_end_sec': int(intervals[1])/16000,
                    'tokens': len(hidden), 'pooled_dim': 128, 'sequence_dim': 128,
                    'model_revision': REVISION, 'adapter_sha256': ADAPTER_SHA,
                    'time_definition': 'estimated CNN token centres on source-video audio timeline; contextual tokens are not point events'}}

def extract_video(path, output, progress=lambda stage: None, face_mode='mediapipe', audio_representation='learned'):
    """An uploaded file is always decoded; its name never selects cached features."""
    import cv2
    import imageio_ffmpeg
    from m3 import extract_faces as face
    from m3.dwt_features import clip_features
    global _audio_model
    if face_mode not in ('mediapipe', 'center'): raise ValueError('Unknown face mode')
    if audio_representation not in ('learned','handcrafted','combined'): raise ValueError('Unknown audio representation')
    state = readiness()
    if not state['ready']: raise ValueError('Raw-video dependencies are not ready: '+json.dumps(state, ensure_ascii=False))
    path, output = Path(path), Path(output)
    progress('Checking video and decoding the audio track')
    cap = cv2.VideoCapture(str(path));fps = float(cap.get(cv2.CAP_PROP_FPS));n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if not np.isfinite(fps) or fps <= 0 or n <= 0: raise ValueError('Cannot read valid video frames or FPS.')
    duration = n/fps
    if duration > 30: raise ValueError('The demo supports clips up to 30 seconds. Trim the video first.')
    wav = output/'audio.wav'
    # Preserve decoded sample rate/channels in float32 WAV. Librosa performs
    # mono conversion and resampling, matching the upstream M1/M2 path rather
    # than adding FFmpeg's resampler and an extra PCM16 quantisation step.
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostdin', '-i', str(path),
               '-map', '0:a:0', '-vn', '-c:a', 'pcm_f32le', '-t', '30', '-y', str(wav)]
    decoded = subprocess.run(command, capture_output=True, timeout=60)
    if decoded.returncode: raise ValueError('Audio track decoding failed. Upload a video with an audio track. '+decoded.stderr.decode(errors='replace')[-250:])
    progress('Uniformly sampling 16 frames and extracting faces')
    frames, indices, fps = face.read_frames(path)
    if face_mode == 'mediapipe':
        face._init_worker()
        try: boxes = [face.detect(f) for f in frames]
        finally:
            face._detector.close();face._detector=None
    else: boxes = [None]*16
    found = np.array([b is not None for b in boxes]);warnings=[]
    if not found.any():
        h,w=frames[0].shape[:2];boxes=[(w/2,h/2,min(h,w),0.)]*16
        warnings.append('No face detected; using a center crop. This is an out-of-distribution demo, with no validated recognition quality.' if face_mode=='mediapipe' else 'Manual center cropping is a preprocessing diagnostic and differs from the standard MediaPipe-aligned input.')
    else:
        good=np.flatnonzero(found)
        boxes=[b if b is not None else boxes[good[np.abs(good-i).argmin()]] for i,b in enumerate(boxes)]
        if not found.all(): warnings.append(f'{int((~found).sum())}/16 frames reused a nearby face bounding box.')
    faces=np.stack([face.align_crop(f,b) for f,b in zip(frames,boxes)])
    for i,frame in enumerate(faces):
        cv2.imwrite(str(output/f'face_{i:02}.jpg'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    progress('Extracting Haar DWT visual features (16×320)')
    sequence=clip_features(faces, grid=4, inner_crop=180).astype(np.float32)
    progress('Loading WavLM and extracting learned audio features (first run takes longer)')
    if _audio_model is None: _audio_model=AudioExtractor()
    audio=_audio_model.extract(wav)
    arrays={k:v for k,v in audio.items() if isinstance(v,np.ndarray)}
    if audio_representation != 'learned':
        progress('Extracting M1 handcrafted audio (48D) and verifying feature order')
        sys.path.insert(0,str(ROOT/'vendor/audio/handcrafted'))
        from src.baseline_v1.feature_pipeline import extract_recording_features
        features=extract_recording_features(wav)
        with (PROJECT/'features/Audio_Handoff/features/handcrafted_features.csv').open() as f:
            columns=[c for c in next(csv.reader(f)) if c not in ('sample_id','trial_id','actor_id','emotion_label','split')]
        if len(columns)!=48 or set(columns)!=set(features):raise ValueError('M1 feature names do not match the training delivery.')
        hc=np.asarray([[features[c] for c in columns]],np.float32)
        if not np.isfinite(hc).all():raise ValueError('M1 nonfinite features')
        arrays['audio_pooled']=hc if audio_representation=='handcrafted' else np.concatenate((hc,arrays['audio_pooled']),axis=1)
        audio['audio_info']['wavlm_pooled_dim']=128
        audio['audio_info']['pooled_dim']=arrays['audio_pooled'].shape[1]
        audio['audio_info']['selected_pooled_dim']=arrays['audio_pooled'].shape[1]
        audio['audio_info']['handcrafted_feature_names']=columns
    arrays.update(visual_pooled=sequence.mean(0)[None],visual_sequence=sequence[None],visual_lengths=np.array([16],np.int64))
    from .engine import CLASSES
    np.savez_compressed(output/'extracted_features.npz',**arrays,
                       sample_ids=np.array(['external_'+sha(path)[:20]]),class_names=np.array(CLASSES),
                       data_kind=np.array('external_unlabelled'),audio_representation=np.array(audio_representation))
    return arrays, {'source_kind':'uploaded_video','sample_id':'external_'+sha(path)[:20],
        'video_sha256':sha(path),'duration_sec':duration,'face_mode':face_mode,
        'frames_found':int(found.sum()),'frame_times':(indices/fps).tolist(),
        'faces':[f'face_{i:02}.jpg' for i in range(16)],'audio_times':audio['audio_times'],
        'audio':audio['audio_info'],'warnings':warnings,
        'visual_source':json.loads((ROOT/'vendor/manifest.json').read_text())['revision'],
        'visual_extractor':'Haar DWT; inner_crop180; grid4; five statistics × four subbands ×16 cells; temporal mean',
        'ground_truth':None}
