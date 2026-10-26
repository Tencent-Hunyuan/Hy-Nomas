import { Reactor } from "@reactor-team/js-sdk";
import "./style.css";
import { CameraControls } from "./camera-controls";

const modes: Record<string, string> = {
  text_only: "Follow the prompt", keyboard: "WASD + arrow keys", stationary: "Hold the view",
  forward: "Move forward", backward: "Move backward", strafe_left: "Move left",
  strafe_right: "Move right", look_left: "Look left", look_right: "Look right",
  look_up: "Look up", look_down: "Look down", trajectory: "Uploaded camera path",
};

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<header>
 <a class="brand" href="https://reactor.inc" target="_blank" rel="noreferrer"><img src="/reactor-symbol.svg" alt="Reactor" width="32" height="23"><strong>Hy-Nomas</strong><span>on Reactor</span></a>
 <div class="connection-bar"><label class="endpoint-label" for="endpoint">Runtime endpoint</label><input id="endpoint" type="url" spellcheck="false" value="http://localhost:8080" aria-label="Runtime endpoint"><span id="connection" class="connection-status" role="status">disconnected</span><button id="connect">Connect</button></div>
</header>
<main>
 <aside>
  <p class="eyebrow">01 / SHAPE YOUR WORLD</p><h1>Describe your world.</h1><p class="intro">Change the scene with words. Explore with WASD.</p>
  <label class="field-title" for="prompt">Prompt <span>Scene · style · action</span></label><textarea id="prompt" rows="6" maxlength="8192" placeholder="Describe the world you want to create…"></textarea>
  <button id="update" class="secondary" disabled>Update prompt</button><p class="hint">Updates guide the next segment while preserving the current world.</p>
  <button id="start" class="primary" disabled>Start world <span aria-hidden="true">↗</span></button><p id="start-hint" class="hint">Connect to your Runtime to begin.</p><p id="error" role="alert" hidden></p>
  <div class="camera-controller" id="camera-controller">
   <div class="controller-heading"><div><strong>Camera control</strong><p id="keyboard-status">Start a world to explore.</p></div><button id="keyboard-toggle" aria-pressed="false" disabled>Enable camera keys</button></div>
   <div class="control-pads">
    <div class="control-pad"><span>MOVE</span><div class="key-grid"><button data-camera-key="w" class="key-top" aria-label="Move forward (W)" disabled>W</button><button data-camera-key="a" aria-label="Move left (A)" disabled>A</button><button data-camera-key="s" aria-label="Move backward (S)" disabled>S</button><button data-camera-key="d" aria-label="Move right (D)" disabled>D</button></div></div>
    <div class="control-pad"><span>LOOK</span><div class="key-grid"><button data-camera-key="up" class="key-top" aria-label="Look up (up arrow)" disabled>↑</button><button data-camera-key="left" aria-label="Look left (left arrow)" disabled>←</button><button data-camera-key="down" aria-label="Look down (down arrow)" disabled>↓</button><button data-camera-key="right" aria-label="Look right (right arrow)" disabled>→</button></div></div>
    <div class="control-feedback"><p id="keyboard-queue" role="status">No camera movement queued.</p><button id="camera-stop" disabled>Stop movement</button><small>Each input guides the next segment. Buffered video plays first.</small></div>
   </div>
  </div>
  <details class="camera-options"><summary>Camera presets &amp; uploaded paths</summary><label class="field-title" for="camera">Camera <span>Next segment</span></label><select id="camera">${Object.entries(modes).map(([key, label]) => `<option value="${key}">${label}</option>`).join("")}</select>
  <div class="row camera-strengths"><label for="translation">Translation<input id="translation" type="number" min="0.1" max="3" step="0.1" value="1"></label><label for="rotation">Rotation<input id="rotation" type="number" min="0.1" max="3" step="0.1" value="2.5"></label></div>
  <button id="apply-camera" class="secondary" disabled>Apply camera</button>
  <label class="upload-card" for="trajectory"><span class="upload-icon" aria-hidden="true">↗</span><span><strong>Choose a camera path</strong><small>NPZ · c2ws + intrinsics · 24 FPS · up to 4 MiB</small></span><input id="trajectory" class="file-input" type="file" accept=".npz"></label><p id="selection" class="hint">Choose a path, then press Apply camera or Start.</p></details>
  <details class="settings"><summary>Generation settings</summary><div class="row"><label for="seed">Seed<input id="seed" type="number" min="0" max="2147483647" value="42"></label><label for="frames">Segment duration<select id="frames"><option value="48">2 seconds</option><option value="72">3 seconds</option><option value="120" selected>5 seconds</option><option value="240">10 seconds</option></select></label></div></details>

 </aside>
 <section class="stage">
  <div class="stage-heading"><div><p class="eyebrow">02 / LIVE OUTPUT</p><h2>Explore what comes next.</h2></div><span class="tag">Live prompts <span>·</span> Camera control</span></div>
  <div class="video-shell" id="video-shell" tabindex="0" aria-label="World viewer. Click to enable WASD and arrow-key camera controls."><video id="video" playsinline muted aria-label="Generated Hy-Nomas world"></video><div id="placeholder" class="placeholder"><div class="placeholder-mark" aria-hidden="true">▷</div><h3 id="placeholder-title">Your world starts here</h3><p id="placeholder-detail">Connect, describe your scene, then press Start world.</p><span class="placeholder-tag">TEXT + CAMERA → WORLD</span></div></div>
  <div class="player-bar"><div class="playback-status"><span id="playback-label">Waiting to connect</span><span id="playback-time" hidden>00:00</span></div><div class="player-actions"><button id="pause" disabled>Pause</button><button id="reset" disabled>Restart world</button><button id="fullscreen" disabled>Fullscreen</button></div></div>
  <p id="progress" role="status">No world generated yet.</p><div class="session-links"><a id="share" hidden target="_blank" rel="noreferrer">Join this world ↗</a><button id="forget-session" class="text-button" hidden>Clear saved session</button></div><p id="held-camera" class="hint">Camera: Follow the prompt</p>
  <div class="stage-note"><span>Change the scene. Keep the world.</span><p>Prompt and camera changes affect future segments. Buffered video continues first; responses may be gradual.</p></div>
  <footer><a href="https://github.com/Tencent-Hunyuan/Hy-Nomas" target="_blank" rel="noreferrer">Tencent Hy-Nomas ↗</a><a href="https://reactor.inc" target="_blank" rel="noreferrer">Built on Reactor ↗</a></footer>
 </section>
</main>`;

const el = <T extends HTMLElement>(id: string) => document.getElementById(id)! as T;
const button = (id: string) => el<HTMLButtonElement>(id);
const input = (id: string) => el<HTMLInputElement>(id);
const prompt = el<HTMLTextAreaElement>("prompt");
const camera = el<HTMLSelectElement>("camera");
const video = el<HTMLVideoElement>("video");
let client: Reactor | undefined;
let ready = false, busy = false, active = false, paused = true, complete = false;
let world = 0, emitted = 0;
let activeCamera = "text_only";
let media = new MediaStream();
let showing = false;
let frameCallback: number | undefined;
let mediaOrigin: number | undefined;
let uploadedFile: File | undefined;
let uploadedCameraName = "";
let sessionId = new URL(location.href).searchParams.get("session") || undefined;

// The SDK supplies a refused command's code and reason on its error event.
class Refused extends Error {}

try { input("endpoint").value = localStorage.getItem("hy-nomas.endpoint") || input("endpoint").value; } catch { /* Optional preference. */ }

const linkedEndpoint = new URL(location.href).searchParams.get("endpoint");
if (linkedEndpoint) input("endpoint").value = linkedEndpoint;

const controls = new CameraControls(el("camera-controller"), el("video-shell"), async ({keys, stop, latch, world: controlledWorld}) => {
  const connection = client;
  if (!connection || !ready || world !== controlledWorld) return;
  error("");
  await command("set_keyboard", {
    world_id: controlledWorld, stop, latch,
    ...Object.fromEntries(["w","s","a","d","up","down","left","right"].map(key => [key, keys.includes(key)])),
  });
}, () => { camera.value = "keyboard"; }, reportError);

function error(text: string) { el("error").textContent = text; el("error").hidden = !text; }
function reportError(cause: unknown) {
  if (!(cause instanceof Refused)) error(cause instanceof Error ? cause.message : String(cause));
}
function clearPlayer() {
  if (frameCallback !== undefined) video.cancelVideoFrameCallback(frameCallback);
  frameCallback = undefined; video.pause(); video.srcObject = null;
  showing = false; mediaOrigin = undefined; el("playback-time").hidden = true;
  el("playback-time").textContent = "00:00";
}
function refresh() {
  controls.update(ready && active && !complete, world, activeCamera === "keyboard");
  button("connect").disabled = busy;
  button("connect").title = client ? "End this session and its world for all viewers." : "Connect to a new or shared session.";
  button("forget-session").hidden = ready || !sessionId;
  button("forget-session").disabled = busy;
  el("share").hidden = !ready || !sessionId;
  input("endpoint").disabled = ready || busy;
  button("start").disabled = !ready || busy || !prompt.value.trim();
  button("start").innerHTML = `${active ? "Start new world" : "Start world"} <span aria-hidden="true">↗</span>`;
  button("update").disabled = !ready || busy || !active || complete || !prompt.value.trim();
  button("apply-camera").disabled = !ready || busy || complete;
  button("pause").disabled = !ready || busy || !active || complete;
  button("pause").textContent = paused ? "Resume" : "Pause";
  button("reset").disabled = !ready || busy || !active;
  button("fullscreen").disabled = !showing;
  el("placeholder").hidden = showing;
  el("video-shell").classList.toggle("has-video", showing);
  el("connection").classList.toggle("live", ready);
  el("placeholder-title").textContent = active ? "Creating your first segment" : ready ? "Ready when you are" : "Your world starts here";
  el("placeholder-detail").textContent = active ? "The first segment can take longer while the model warms up." : ready ? "Write a prompt, choose a camera, then press Start world." : "Connect, describe your scene, then press Start world.";
  el("playback-label").textContent = !ready ? "Waiting to connect" : !active ? "Connected · waiting for input" : complete ? "World complete · start or restart to continue" : paused ? "Generation paused · buffered video can still play" : showing ? "Live world" : "Generating first segment…";
  el("start-hint").textContent = !ready ? "Connect to your Runtime to begin." : active ? "Start new world resets history. Update prompt continues the current world." : "Generation starts only when you press Start.";
}
function attachMedia() {
  if (!ready || !active || emitted < 1 || !media.getVideoTracks().length || video.srcObject === media) return;
  const playbackWorld = world, playbackMedia = media;
  video.srcObject = playbackMedia;
  const onFrame = (_now: number, metadata: VideoFrameCallbackMetadata) => {
    if (!ready || world !== playbackWorld || video.srcObject !== playbackMedia) return;
    mediaOrigin ??= metadata.mediaTime;
    const seconds = Math.max(0, Math.floor(metadata.mediaTime - mediaOrigin));
    el("playback-time").textContent = `${Math.floor(seconds / 60).toString().padStart(2,"0")}:${(seconds % 60).toString().padStart(2,"0")}`;
    el("playback-time").hidden = false; showing = true; refresh();
    frameCallback = video.requestVideoFrameCallback(onFrame);
  };
  frameCallback = video.requestVideoFrameCallback(onFrame);
  void video.play().catch((cause) => {
    if (ready && world === playbackWorld && video.srcObject === playbackMedia) error(`Playback: ${String(cause)}`);
  });
}
function readState(state: Record<string, unknown>) {
  const nextWorld = Number(state.world_id);
  if (nextWorld !== world) clearPlayer();
  world = nextWorld; active = Boolean(state.active); paused = Boolean(state.paused);
  activeCamera = String(state.camera);
  complete = Boolean(state.complete); emitted = Number(state.emitted_frames);
  uploadedCameraName = String(state.camera_path || "");
  const queued = (state.queued_keyboard_keys || []) as string[];
  const held = (state.keyboard_keys || []) as string[];
  const keyLabel = (key: string) => ({up:"↑",down:"↓",left:"←",right:"→"}[key] || key.toUpperCase());
  el("keyboard-queue").textContent = queued.length ? `Queued for next segment: ${queued.map(keyLabel).join(" + ")}`
    : held.length ? `Continuing: ${held.map(keyLabel).join(" + ")}` : "No camera movement queued.";

  el("held-camera").textContent = `Camera: ${modes[String(state.camera)] || state.camera}${state.camera === "trajectory" ? ` · ${Number(state.camera_frames_remaining)} path frames remaining` : ""}`;
  const elapsed = Number(state.last_segment_seconds);
  const requested = Number(state.segment_frames);
  const lastFrames = Number(state.generated_frames) - Math.max(0, Number(state.completed_segments) - 1) * requested;
  const seconds = elapsed > 0 ? ` · ${elapsed.toFixed(2)} s / segment · ${(lastFrames / elapsed).toFixed(1)} generation FPS` : "";
  el("progress").textContent = active ? `World ${world} · ${Number(state.completed_segments)} segments · ${emitted} frames sent${seconds}` : "No world generated yet.";
  refresh(); attachMedia();
}
async function command(name: string, args: Record<string, unknown> = {}) {
  if (!client) throw new Error("Connect to your Runtime first.");
  const reply = await client.sendCommand(name, args);
  if (reply === undefined) throw new Refused(name);
  return reply;
}
async function act(action: () => Promise<unknown>) {
  if (busy) return;
  busy = true; error(""); refresh();
  try { await action(); } catch (cause) { reportError(cause); }
  finally { busy = false; refresh(); }
}
function resetConnection() {
  clearPlayer(); ready = active = complete = false; paused = true; world = emitted = 0;
  activeCamera = "text_only";
  media = new MediaStream(); uploadedFile = undefined; uploadedCameraName = "";
  el("connection").textContent = "disconnected"; button("connect").textContent = "Connect";
  refresh();
}
function forgetSession() {
  sessionId = undefined;
  const url = new URL(location.href); url.searchParams.delete("session");
  history.replaceState(null, "", url);
  el("share").removeAttribute("href");
  refresh();
}
button("connect").onclick = () => void act(async () => {
  if (client) {
    const previous = client;
    controls.release(true);
    el("connection").textContent = "ending session";
    button("connect").textContent = "Disconnecting…";
    await previous.disconnect();
    if (client === previous) { client = undefined; resetConnection(); }
    previous[Symbol.dispose]();
    forgetSession();
    return;
  }
  const endpoint = input("endpoint").value.trim();
  if (!["http:","https:"].includes(new URL(endpoint).protocol)) throw new Error("Use an HTTP or HTTPS endpoint.");
  try { localStorage.setItem("hy-nomas.endpoint", endpoint); } catch { /* Optional preference. */ }
  const connection = new Reactor({modelName:"hy-nomas", apiUrl:endpoint, local:true,
    controlRequestTimeoutMs:180000, modelTracks:[{name:"main_video",kind:"video",direction:"recvonly"}]});
  client = connection;
  connection.on("statusChanged", (status) => {
    if (client !== connection) return;
    ready = status === "ready"; el("connection").textContent = status;
    button("connect").textContent = status === "disconnected" ? "Connect" : "Disconnect";
    if (status === "disconnected") { client = undefined; connection[Symbol.dispose](); resetConnection(); }
    refresh(); attachMedia();
  });
  connection.on("sessionIdChanged", (id) => {
    if (client !== connection || !id) return;
    sessionId = id;
    const url = new URL(location.href);
    url.searchParams.set("session", id); url.searchParams.set("endpoint", endpoint);
    history.replaceState(null, "", url);
    el<HTMLAnchorElement>("share").href = url.href;
    refresh();
  });
  connection.on("error", (cause) => {
    if (client === connection && cause.code !== "world_changed") error(cause.message);
  });
  connection.on("trackReceived", (_name, track) => {
    if (client !== connection) return;
    if (!media.getTracks().some((existing) => existing.id === track.id)) media.addTrack(track);
    attachMedia();
  });
  connection.on("message", (message) => {
    if (client === connection && message.type === "state_update") readState(message.data as Record<string,unknown>);
  });
  try {
    await connection.connect(undefined, sessionId ? {sessionId} : undefined);
    if (client === connection) {
      const snapshot = await command("get_state");
      readState(snapshot.data as Record<string, unknown>);
    }
  }
  catch (cause) { if (client === connection) client = undefined; connection[Symbol.dispose](); resetConnection(); throw cause; }
});
async function uploadPath() {
  const file = input("trajectory").files?.[0];
  if (!file && !uploadedCameraName) throw new Error("Choose a camera NPZ first.");
  if (file && file !== uploadedFile) {
    if (file.size > 4 * 1024 * 1024) throw new Error("Camera NPZ must be at most 4 MiB.");
    const trajectory = await client!.uploadFile(file);
    await command("set_camera_path", {trajectory}); uploadedFile = file;
  }
}
async function applyCamera() {
  controls.release(true);
  if (camera.value === "trajectory") await uploadPath();
  await command("set_camera", {camera:camera.value});
  await command("set_translation_strength", {translation_strength:Number(input("translation").value)});
  await command("set_rotation_strength", {rotation_strength:Number(input("rotation").value)});
}
button("start").onclick = () => void act(async () => {
  controls.release(true);
  const file = input("trajectory").files?.[0];
  let trajectory: unknown;
  if (camera.value === "trajectory") {
    if (!file && !uploadedCameraName) throw new Error("Choose a camera NPZ first.");
    if (file && file !== uploadedFile) {
      if (file.size > 4 * 1024 * 1024) throw new Error("Camera NPZ must be at most 4 MiB.");
      trajectory = await client!.uploadFile(file);
    }
  }
  await command("start", {prompt:prompt.value, seed:Number(input("seed").value),
    segment_frames:Number(el<HTMLSelectElement>("frames").value), camera:camera.value,
    translation_strength:Number(input("translation").value), rotation_strength:Number(input("rotation").value),
    ...(trajectory ? {trajectory} : {})});
  if (trajectory && file) uploadedFile = file;
});
button("update").onclick = () => void act(() => command("set_prompt", {prompt:prompt.value}));
button("apply-camera").onclick = () => void act(applyCamera);
button("pause").onclick = () => void act(() => command("set_paused", {paused:!paused}));
button("reset").onclick = () => void act(() => { controls.release(true); return command("reset"); });
button("fullscreen").onclick = () => void el("video-shell").requestFullscreen();
input("trajectory").onchange = () => {
  const file = input("trajectory").files?.[0];
  el("selection").textContent = file ? `${file.name} · waiting for Apply camera or Start` : "Choose a path, then press Apply camera or Start.";
  if (file) camera.value = "trajectory";
};
prompt.oninput = refresh;
button("forget-session").onclick = () => {
  forgetSession(); error("");
};
window.addEventListener("beforeunload", () => { client?.[Symbol.dispose](); });
refresh();
