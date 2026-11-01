import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from apps.reference_image import resolve_reference_image, save_reference_image
from apps.seg_ar_gateway import create_gateway_app
from apps.seg_ar_job_store import JobStore
from apps.kimi_k3_prompt_enhancer import (
    PromptEnhancementError, build_local_fallback_prompt,
)
from apps.k3_image_captioner import K3ImageCaptioner, NomasImageCaptioner, caption_to_enhanced, normalize_caption
from hymm.pipeline.reference_image import encode_reference_image
from hymm.utils.image_resize import fit_image_to_frame


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64), "red").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize('reference', [False, True])
def test_saved_reference_shows_only_last_half_second(tmp_path, monkeypatch, reference):
    from apps.seg_ar_web_server import GenerationService
    import imageio.v2 as imageio
    import numpy as np
    service = GenerationService.__new__(GenerationService)
    service.config = SimpleNamespace(inference=SimpleNamespace(video_fps=24))
    service.output_dir = tmp_path
    service.rollout_type = 'seg_ar'
    captured = []
    monkeypatch.setattr(imageio, 'mimwrite', lambda path, frames, **kwargs: captured.append(frames.copy()))
    monkeypatch.setattr('apps.seg_ar_web_server._publish_local_file', lambda *_: None)
    monkeypatch.setattr('apps.seg_ar_web_server.video_overlays_enabled', lambda: False)
    output = torch.arange(120, dtype=torch.uint8).view(1, 1, 120, 1, 1).expand(1, 3, 120, 16, 16)
    params = {'request_id': 'trim-test', 'seed': 42, 'num_frames': 120,
        'timing_source': 'request', 'segment_durations_seconds': [5.0],
        'reference_image_id': 'image' if reference else None}
    result = service._save_output(output, params, 0.1)
    assert result['frames'] == (12 if reference else 120)
    assert result['duration_seconds'] == (0.5 if reference else 5.0)
    np.testing.assert_array_equal(captured[0][:, 0, 0, 0], np.arange(108 if reference else 0, 120))


def test_reference_mp4_and_stream_join_exclude_hidden_prefix(tmp_path, monkeypatch):
    from apps.seg_ar_web_server import GenerationService
    import imageio.v2 as imageio
    service = GenerationService.__new__(GenerationService)
    service.config = SimpleNamespace(inference=SimpleNamespace(video_fps=24))
    service.output_dir = tmp_path
    service.rollout_type = 'seg_ar'
    monkeypatch.setattr('apps.seg_ar_web_server.video_overlays_enabled', lambda: False)
    params = {'request_id': 'reference', 'seed': 42, 'num_frames': 120,
        'timing_source': 'request', 'segment_durations_seconds': [5.0], 'reference_image_id': 'image'}
    video = torch.zeros(1, 3, 120, 32, 32, dtype=torch.uint8)
    video[:, 0, :108] = 255
    video[:, 1, 108:] = 255
    first = service._save_output(video, params, 0)
    reader = imageio.get_reader(str(service._resolve_output_url(first['video_url'])))
    try:
        assert reader.count_frames() == 12
        frame = reader.get_data(0)
        assert frame[:, :, 1].mean() > 240 and frame[:, :, 0].mean() < 10
    finally:
        reader.close()


def test_reference_keeps_full_model_run_and_publishes_half_second(tmp_path):
    from apps.seg_ar_web_server import GenerationService
    store = JobStore(tmp_path / 'jobs.json')
    service = GenerationService.__new__(GenerationService)
    service.generate = Mock(return_value={'id': 'ref', 'video_url': 'outputs/private.mp4',
        'frames': 12, 'fps': 24, 'provisional_pixel_frames': 36})
    service._write_mse_fragment = Mock(return_value={'stream_sequence': 0})
    service._concat_session_video = Mock()
    pe_session = {'session_id': 'hidden', 'segment_frame_lengths': [12], 'reference_hidden_frames': 108}
    job = store.submit('generate', {'reference_image_id': 'image', 'reset_session': True,
        'num_frames': 120, '_pe_session': pe_session})
    service.execute_queue_job(store, store.claim_next('test-worker'))
    assert service.generate.call_args.args[0].preserve_kv_cache
    assert service.generate.call_args.args[0].num_frames == 120
    service._write_mse_fragment.assert_called_once()
    service._concat_session_video.assert_not_called()
    saved = store.get_session('hidden')
    assert saved['video_url'] == 'outputs/private.mp4'
    assert saved['last_result']['frames'] == 12
    assert saved['last_result']['duration_seconds'] == 0.5
    assert saved['last_result']['reference_image_segment']
    assert not saved['last_result'].get('initialization_only')
    assert store.get(job['id'])['status'] == 'succeeded'


def test_text_segment_follows_reference_without_reintroducing_hidden_frames(tmp_path, monkeypatch):
    from apps.seg_ar_web_server import GenerationService
    import imageio.v2 as imageio
    service = GenerationService.__new__(GenerationService)
    service.config = SimpleNamespace(inference=SimpleNamespace(video_fps=24))
    service.output_dir = tmp_path
    service.rollout_type = 'seg_ar'
    frames = []
    monkeypatch.setattr(imageio, 'mimwrite', lambda path, video, **kwargs: frames.append(video.copy()))
    monkeypatch.setattr('apps.seg_ar_web_server._publish_local_file', lambda *_: None)
    monkeypatch.setattr('apps.seg_ar_web_server.video_overlays_enabled', lambda: False)
    output = torch.arange(84, dtype=torch.uint8).view(1, 1, 84, 1, 1).expand(1, 3, 84, 16, 16)
    service.pipeline = SimpleNamespace(_last_pixelumm_streaming_metadata={'replace_pixel_frames': 36})
    result = service._save_output(output, {'request_id': 'next', 'seed': 42, 'num_frames': 48,
        'timing_source': 'request', 'segment_durations_seconds': [2],
        'stable_prefix_carry_frames': 36, 'reference_carry_visible_frames': 12}, 0)
    assert len(frames[0]) == 60
    assert frames[0][0, 0, 0, 0] == 24
    result.update(replace_pixel_frames=12, provisional_pixel_frames=4)
    service.generate = Mock(return_value=result)
    service._write_mse_fragment = Mock(return_value={'stream_sequence': 1})
    service._concat_session_video = Mock(return_value='outputs/joined.mp4')
    store = JobStore(tmp_path / 'jobs.json')
    previous = {'frames': 12, 'fps': 24, 'reference_image_segment': True,
        'stream_sequence': 0, 'provisional_pixel_frames': 36}
    pe_session = {'session_id': 'hidden', 'video_url': 'outputs/reference.mp4', 'last_result': previous,
        'result_history': [previous], 'segment_frame_lengths': [12, 48],
        'reference_hidden_frames': 108}
    job = store.submit('generate', {'prompt': '{}', 'stable_prefix_carry_frames': 36,
        'reference_carry_visible_frames': 12, '_pe_session': pe_session})
    service.execute_queue_job(store, store.claim_next('test-worker'))
    saved = store.get(job['id'])['result']
    assert saved['frames'] == 60
    assert saved['segment_frames'] == 48
    assert saved['playback_start_seconds'] == 0.5
    assert saved['stream_sequence'] == 1
    assert saved['stream_replace_start_seconds'] == 0
    assert service._concat_session_video.call_args.kwargs['replace_tail_frames'] == 12


@pytest.mark.parametrize("mode, options", [
    ("RGB", {}), ("RGB", {"progressive": True}), ("CMYK", {}),
])
def test_jpeg_variants_are_accepted(tmp_path, mode, options):
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64), "red").convert(mode).save(buffer, format="JPEG", **options)
    result = save_reference_image(buffer.getvalue(), tmp_path)
    with Image.open(resolve_reference_image(result["id"], tmp_path)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (1280, 720)


def test_jpeg_with_auxiliary_image_uses_primary_frame(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64), "red").save(
        buffer, format="MPO", save_all=True,
        append_images=[Image.new("RGB", (96, 64), "blue")],
    )
    data = buffer.getvalue()
    assert data.startswith(b"\xff\xd8\xff")
    with Image.open(io.BytesIO(data)) as source:
        assert source.format == "MPO" and source.n_frames == 2
    captioner = Mock(configured=True)
    captioner.caption.return_value = {"caption_status": "ready"}
    app = create_gateway_app(JobStore(tmp_path / "jobs.json"), frontend_dir=None,
                             output_dir=tmp_path, image_captioner=captioner)
    with TestClient(app) as client:
        response = client.post("/v1/reference-images", content=data,
                               headers={"Content-Type": "image/jpeg"})
        assert response.status_code == 201
    with Image.open(resolve_reference_image(response.json()["id"], tmp_path)) as image:
        assert image.format == "JPEG"
        assert getattr(image, "n_frames", 1) == 1
        assert image.size == (1280, 720)
        red, green, blue = image.getpixel((640, 360))
        assert red > 240 and green < 15 and blue < 15
    captioner.caption.assert_called_once()


def test_unsupported_format_error_identifies_decoded_format(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64)).save(buffer, format="BMP")
    with pytest.raises(ValueError, match="detected as BMP"):
        save_reference_image(buffer.getvalue(), tmp_path)


@pytest.mark.parametrize("format", ["PNG", "WEBP"])
def test_animated_images_remain_rejected(tmp_path, format):
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64), "red").save(
        buffer, format=format, save_all=True,
        append_images=[Image.new("RGB", (96, 64), "blue")], duration=100,
    )
    with pytest.raises(ValueError, match="single still image"):
        save_reference_image(buffer.getvalue(), tmp_path)


def test_upload_decodes_image_and_rejects_invalid_files_and_paths(tmp_path):
    captioner = Mock(configured=True)
    captioner.caption.return_value = {"caption_status": "ready"}
    app = create_gateway_app(JobStore(tmp_path / "jobs.json"), frontend_dir=None, output_dir=tmp_path, image_captioner=captioner)
    with TestClient(app) as client:
        result = client.post("/v1/reference-images", content=image_bytes(), headers={"Content-Type": "image/png"})
        assert result.status_code == 201
        assert client.get('/v1/reference-images/' + result.json()['id']).json()['caption_status'] == 'ready'
        stored = resolve_reference_image(result.json()["id"], tmp_path)
        with Image.open(stored) as image:
            assert image.format == "JPEG"
            assert image.size == (1280, 720)
        assert client.post("/v1/reference-images", content=b"not an image").status_code == 400
        assert client.post("/v1/reference-images", content=b"x" * (10 * 1024 * 1024 + 1)).status_code == 413
        assert client.post("/v1/generate", json={"reference_image_id": "../../etc/passwd"}).status_code == 400
        assert client.post("/v1/generate", json={"reference_image_id": result.json()["id"]}).status_code == 400
    with pytest.raises(ValueError):
        resolve_reference_image("../bad", tmp_path)


@pytest.mark.parametrize("source_size, expected_box", [
    ((600, 1200), (460, 0, 820, 720)),
    ((2400, 600), (0, 200, 1280, 520)),
    ((800, 800), (280, 0, 1000, 720)),
    ((1600, 900), (0, 0, 1280, 720)),
])
def test_fit_preserves_all_four_corners_and_aspect_ratio(source_size, expected_box):
    image = Image.new('RGB', source_size, 'white')
    draw = ImageDraw.Draw(image)
    width, height = source_size
    tile = min(width, height) // 5
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for (x, y), color in zip([(0, 0), (width-tile, 0), (0, height-tile), (width-tile, height-tile)], colors):
        draw.rectangle((x, y, x+tile-1, y+tile-1), fill=color)
    fitted = fit_image_to_frame(image, (1280, 720))
    assert fitted.size == (1280, 720)
    assert fitted.getbbox() == expected_box
    left, top, right, bottom = expected_box
    assert (right-left) / (bottom-top) == pytest.approx(width/height)
    inset = max(2, round(tile * (right-left) / width / 2))
    for point, color in zip([
        (left+inset, top+inset), (right-inset-1, top+inset),
        (left+inset, bottom-inset-1), (right-inset-1, bottom-inset-1),
    ], colors):
        assert fitted.getpixel(point) == color


def test_upload_orientation_and_encoder_fit_keep_all_content(tmp_path):
    source = Image.new('RGB', (120, 60), 'white')
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 0, 29, 59), fill='red')
    draw.rectangle((90, 0, 119, 59), fill='blue')
    exif = Image.Exif()
    exif[274] = 6  # Camera orientation: rotate 90 degrees clockwise.
    buffer = io.BytesIO()
    source.save(buffer, format='JPEG', quality=100, subsampling=0, exif=exif)
    saved = save_reference_image(buffer.getvalue(), tmp_path)
    assert (saved['width'], saved['height']) == (1280, 720)
    with Image.open(resolve_reference_image(saved['id'], tmp_path)) as uploaded:
        assert uploaded.getexif().get(274, 1) == 1
        expected = fit_image_to_frame(uploaded, (1280, 704))
        top, bottom = expected.getpixel((640, 55)), expected.getpixel((640, 648))
        assert top[0] > 230 and top[2] < 20
        assert bottom[2] > 230 and bottom[0] < 20

    calls = []
    def encode(pixels):
        calls.append(tuple(pixels.shape))
        assert pixels.shape == (1, 3, 48, 704, 1280)
        first = pixels[0, :, 0]
        assert first[0, 55, 640] > 0.8 and first[2, 648, 640] > 0.8
        assert torch.equal(pixels[:, :, :1], pixels[:, :, -1:])
        latent = torch.zeros(1, 32, 12, 22, 40)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    vae = SimpleNamespace(spatial_compression_ratio=32, time_compression_ratio=4, encode=encode)
    latent = encode_reference_image(vae, resolve_reference_image(saved['id'], tmp_path),
        height=720, width=1280, num_frames=48, device='cpu', dtype=torch.float32)
    assert latent.shape == (1, 32, 12, 22, 40)
    assert len(calls) == 1


def test_encoding_repeats_pixels_for_five_seconds_before_vae(tmp_path):
    saved = save_reference_image(image_bytes(), tmp_path)
    calls = []
    encoded = torch.arange(32 * 30 * 2 * 3).reshape(1, 32, 30, 2, 3).float()

    def encode(pixels):
        calls.append(pixels)
        assert pixels.shape == (1, 3, 120, 64, 96)
        assert pixels.min() >= -1 and pixels.max() <= 1
        assert torch.equal(pixels, pixels[:, :, :1].expand_as(pixels))
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: encoded))

    vae = SimpleNamespace(spatial_compression_ratio=32, time_compression_ratio=4, encode=encode)
    result = encode_reference_image(
        vae, resolve_reference_image(saved["id"], tmp_path),
        height=64, width=96, num_frames=120, device="cpu", dtype=torch.float32,
    )
    assert len(calls) == 1
    assert result.shape == (1, 32, 30, 2, 3)
    # Temporal outputs must pass through unchanged, even for identical pixels.
    assert torch.equal(result, encoded)
    assert not torch.equal(result[:, :, 0], result[:, :, -1])


@pytest.mark.parametrize("fail", [False, True])
def test_reference_encoding_restores_idle_models_on_success_and_failure(tmp_path, monkeypatch, fail):
    from apps.seg_ar_web_server import GenerationService

    saved = save_reference_image(image_bytes(), tmp_path)
    moves = []

    class Model:
        def __init__(self, name):
            self.name = name
            self.device = torch.device("cuda:0")

        def parameters(self):
            return iter([SimpleNamespace(device=self.device)])

        def to(self, *, device):
            self.device = torch.device(device)
            moves.append((self.name, self.device.type))

    models = [Model("dit"), Model("decoder")]
    latent = torch.zeros(1, 32, 12, 2, 3)

    def encode(*args, **kwargs):
        assert all(m.device.type == "cpu" for m in models)
        if fail:
            raise RuntimeError("encoder failed")
        return latent

    monkeypatch.setattr("hymm.models.vae.load_vae", lambda **kwargs: (object(), None, 32, 4))
    monkeypatch.setattr("hymm.pipeline.reference_image.encode_reference_image", encode)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    service = GenerationService.__new__(GenerationService)
    service.rank, service.world_size, service.device = 0, 1, torch.device("cuda:0")
    service.pipeline = SimpleNamespace(transformer=models[0], vae=models[1])
    service.config = SimpleNamespace(vae=SimpleNamespace(encoder_checkpoint="encoder.pt"))
    service.output_dir = tmp_path
    params = {"reference_image_id": saved["id"], "height": 64, "width": 96, "num_frames": 48}
    if fail:
        with pytest.raises(ValueError, match="encoder failed"):
            service._encode_reference(params)
    else:
        assert service._encode_reference(params) is latent
    assert moves == [("dit", "cpu"), ("decoder", "cpu"), ("dit", "cuda"), ("decoder", "cuda")]


@pytest.mark.parametrize("captioner_class", [K3ImageCaptioner, NomasImageCaptioner])
def test_captioner_receives_user_perspective_and_retimes_output_to_five_seconds(tmp_path, monkeypatch, captioner_class):
    monkeypatch.setenv("K3_API_URL", "http://k3.test/v1/chat/completions")
    monkeypatch.setenv("APP_ID", "test-app")
    monkeypatch.setenv("APP_KEY", "test-secret")
    monkeypatch.delenv("K3_API_KEY", raising=False)
    monkeypatch.delenv("AIHUB_API_KEY", raising=False)
    monkeypatch.setenv("PE_API_URL", "http://nomas.test/v1/chat/completions")
    monkeypatch.setenv("PE_MODEL", "Nomas-PE")
    monkeypatch.delenv("PE_API_KEY", raising=False)
    saved = save_reference_image(image_bytes(), tmp_path)
    request = {
        "session_id": "image-world", "perspective": "first-person",
        "subject_description": "A red wall", "environment_description": "A quiet red room",
        "reference_image_path": str(resolve_reference_image(saved["id"], tmp_path)),
    }
    # K3's inferred viewpoint and duration must not override the user's contract.
    response = build_local_fallback_prompt(request, None, duration_seconds=8).prompt
    session = Mock()
    http = Mock(status_code=200, headers={"Content-Type": "application/json"})
    http.json.return_value = {"choices": [{"message": {"content": response}}]}
    session.post.return_value.__enter__ = Mock(return_value=http)
    session.post.return_value.__exit__ = Mock(return_value=False)
    caption = captioner_class(session=session).caption(resolve_reference_image(saved["id"], tmp_path), perspective='second-person')
    enhanced = caption_to_enhanced(caption, session_id="image-world")
    assert enhanced.num_frames == 120
    assert enhanced.segment_boundaries == [0, 120]
    assert enhanced.session["segment_frame_lengths"] == [12]
    assert enhanced.session["reference_hidden_frames"] == 108
    assert enhanced.session["previous_end"] == "00:05.000"
    assert json.loads(enhanced.prompt)["segments"][0]["end_time"] == "00:05.000"
    assert caption['perspective'] == enhanced.session['perspective'] == 'second-person'
    messages = session.post.call_args.kwargs['json']['messages']
    assert any(item.get("type") == "image_url" for item in messages[-1]["content"])
    assert json.loads(enhanced.prompt)['segments'][0]['contained_actions'] == []
    if captioner_class is K3ImageCaptioner:
        assert 'User-selected perspective_view: second-person' in messages[-1]['content'][0]['text']
        assert 'test-app:test-secret?provider=ali&model=kimi-k3' in session.post.call_args.kwargs['headers']['Authorization']
    else:
        user_input = json.loads(messages[-1]['content'][0]['text'])
        assert user_input['perspective_view'] == 'second-person'
        assert 'initial 5-second segment' in user_input['user_prompt']
        from pathlib import Path
        prompt_path = Path(__file__).resolve().parents[2] / 'apps/prompts_i2v/online_user_video_rewriter_system_prompt.md'
        assert messages[0]['content'] == prompt_path.read_text().strip()
        assert session.post.call_args.args[0] == 'http://nomas.test/v1/chat/completions'
        assert 'Authorization' not in session.post.call_args.kwargs['headers']
        payload = session.post.call_args.kwargs['json']
        assert payload['model'] == caption['caption_model'] == 'Nomas-PE'
        assert payload['chat_template_kwargs'] == {'enable_thinking': False}
        assert payload['max_tokens'] == 4096
        assert 'max_completion_tokens' not in payload


def test_nomas_captioner_uses_only_pe_credentials(monkeypatch):
    monkeypatch.setenv('APP_ID', 'unrelated-app')
    monkeypatch.setenv('APP_KEY', 'unrelated-secret')
    monkeypatch.setenv('PE_API_KEY', 'local-pe-key')
    captioner = NomasImageCaptioner()
    assert captioner.configured
    assert captioner._headers('image')['Authorization'] == 'Bearer local-pe-key'


def test_k3_rejects_placeholder_credentials(monkeypatch):
    monkeypatch.setenv('K3_API_URL', 'http://k3.test/v1/chat/completions')
    monkeypatch.setenv('APP_ID', '你的应用')
    monkeypatch.setenv('APP_KEY', '你的密钥')
    monkeypatch.delenv('K3_API_KEY', raising=False)
    monkeypatch.delenv('AIHUB_API_KEY', raising=False)
    captioner = K3ImageCaptioner()
    assert not captioner.configured
    with pytest.raises(PromptEnhancementError, match='valid'):
        captioner._headers('image')


@pytest.mark.parametrize("subject_count", [0, 2, 3])
def test_image_caption_requires_exactly_one_subject(subject_count):
    request = {'perspective': 'third-person observer',
               'subject_description': 'A red cube', 'environment_description': 'A table'}
    parsed = json.loads(build_local_fallback_prompt(request, None).prompt)
    subject = parsed['subject_list'][0]
    parsed['subject_list'] = [
        {**subject, 'subject_id': f'<|Subject{index + 1}|>'}
        for index in range(subject_count)
    ]
    with pytest.raises(PromptEnhancementError, match='exactly one primary subject'):
        normalize_caption(parsed)
    # Cached captions must obey the same contract when entering a new world.
    with pytest.raises(PromptEnhancementError, match='exactly one primary subject'):
        caption_to_enhanced({'prompt': parsed}, session_id='single-subject')


def test_single_subject_caption_preserves_other_objects_as_scene_context():
    request = {'perspective': 'third-person observer',
               'subject_description': 'A red cube',
               'environment_description': 'A blue sphere stands beside the red cube on a table.'}
    parsed = json.loads(build_local_fallback_prompt(request, None).prompt)
    caption = normalize_caption(parsed)
    assert len(caption['prompt']['subject_list']) == 1
    assert caption['subject_description'] == 'A red cube'
    assert 'blue sphere' in caption['environment_description']
    enhanced = caption_to_enhanced(caption, session_id='single-subject')
    assert enhanced.session['subject_ids'] == ['<|Subject1|>']


def test_generation_uses_saved_k3_caption_without_calling_pe(tmp_path):
    request = {'session_id': 'caption-world', 'perspective': 'first-person',
               'subject_description': 'A red cube', 'environment_description': 'A table'}
    caption = normalize_caption(json.loads(build_local_fallback_prompt(request, None).prompt))
    captioner = Mock(configured=True)
    captioner.caption.return_value = caption
    enhancer = Mock()
    store = JobStore(tmp_path / 'jobs.json')
    store.heartbeat('worker', 'ready', {'reference_image_enabled': True, 'fps': 24})
    app = create_gateway_app(store, frontend_dir=None, output_dir=tmp_path,
                             image_captioner=captioner, prompt_enhancer=enhancer)
    with TestClient(app) as client:
        upload = client.post('/v1/reference-images?perspective=first-person', content=image_bytes()).json()
        assert captioner.caption.call_args.kwargs['perspective'] == 'first-person'
        assert client.post('/v1/reference-images?perspective=invalid', content=image_bytes()).status_code == 400
        result = client.post('/v1/generate', json={
            'reference_image_id': upload['id'], 'reset_session': True,
            'pe_input': {'session_id': 'caption-world', 'subject_description': 'Ignore caption', 'perspective': 'second-person'},
        })
        assert result.status_code == 202
        job = store.get(result.json()['id'])
        assert job['status'] == 'queued'
        queued = store.claim_next('worker')
        assert queued['payload']['num_frames'] == 120
        assert queued['payload']['_pe_session']['subject'] == 'A red cube'
        assert json.loads(queued['payload']['prompt']) == {**caption['prompt'], 'perspective_view': 'second-person'}
        assert queued['payload']['_pe_session']['perspective'] == 'second-person'
        enhancer.enhance.assert_not_called()


def test_worker_enforces_first_segment_duration_and_encoder(tmp_path):
    from apps.seg_ar_demo_config import load_demo_config
    from apps.seg_ar_web_server import GenerateRequest, GenerationService

    saved = save_reference_image(image_bytes(), tmp_path)
    encoder = tmp_path / "encoder.pt"
    encoder.touch()
    service = GenerationService.__new__(GenerationService)
    service.config = load_demo_config([f"model.vae.encoder_checkpoint={encoder}"])
    service.output_dir = tmp_path
    service.rollout_type = "seg_ar"
    service._request_count = 0
    initial = {
        "session_id": "image-world", "perspective": "first-person",
        "subject_description": "A red wall", "environment_description": "A quiet room",
    }
    prompt = build_local_fallback_prompt(initial, None, duration_seconds=5)
    request = GenerateRequest(
        prompt=prompt.prompt, num_frames=120, segment_boundaries=[0, 120],
        reference_image_id=saved["id"], session_id="image-world",
        reset_session=True, preserve_kv_cache=True,
    )
    assert service.build_params(request)["reference_image_id"] == saved["id"]
    with pytest.raises(ValueError, match="first segment"):
        service.build_params(request.model_copy(update={"reset_session": False}))
    long_prompt = build_local_fallback_prompt(initial, None, duration_seconds=8)
    with pytest.raises(ValueError, match="exactly 5 seconds"):
        service.build_params(request.model_copy(update={
            "prompt": long_prompt.prompt, "num_frames": 192, "segment_boundaries": [0, 192],
        }))
    with pytest.raises(ValueError, match="Camera motion"):
        service.build_params(request.model_copy(update={"camera_trajectory": "forward"}))
