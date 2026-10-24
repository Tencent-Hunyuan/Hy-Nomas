const keyCodes: Record<string, string> = {
  KeyW: "w", KeyS: "s", KeyA: "a", KeyD: "d",
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
};

type Movement = { keys: string[]; stop: boolean; latch: boolean; world: number };

export class CameraControls {
  private physical = new Set<string>();
  private pointers = new Map<number, string>();
  private available = false;
  private enabled = false;
  private world = 0;
  private revision = 0;
  private sending = false;
  private latest: Movement | undefined;
  private motion: Movement | undefined;
  private flushTimer: ReturnType<typeof setTimeout> | undefined;
  private buttons: HTMLButtonElement[];
  private toggle: HTMLButtonElement;
  private stop: HTMLButtonElement;
  private status: HTMLElement;

  constructor(
    private root: HTMLElement,
    stage: HTMLElement,
    private send: (movement: Movement) => Promise<void>,
    private onEnable: () => void,
    private onError: (cause: unknown) => void,
  ) {
    this.buttons = Array.from(root.querySelectorAll<HTMLButtonElement>("[data-camera-key]"));
    this.toggle = root.querySelector<HTMLButtonElement>("#keyboard-toggle")!;
    this.stop = root.querySelector<HTMLButtonElement>("#camera-stop")!;
    this.status = root.querySelector<HTMLElement>("#keyboard-status")!;
    this.toggle.onclick = () => {
      if (this.enabled) this.release(true);
      else { this.enable(); stage.focus({preventScroll: true}); }
    };
    this.stop.onclick = () => this.release();
    stage.addEventListener("pointerdown", () => {
      this.enable(); stage.focus({preventScroll: true});
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && this.enabled) { this.release(true); return; }
      const key = keyCodes[event.code];
      if (!key || !this.available || !this.enabled || this.editable(event.target)
          || event.altKey || event.ctrlKey || event.metaKey) return;
      event.preventDefault();
      if (event.repeat || this.physical.has(key)) return;
      this.physical.add(key); this.changed(true);
    });
    window.addEventListener("keyup", (event) => {
      const key = keyCodes[event.code];
      if (key && this.physical.delete(key)) { event.preventDefault(); this.changed(); }
    });
    window.addEventListener("blur", () => this.release(true));
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.release(true);
    });
    document.addEventListener("focusin", (event) => {
      if (this.enabled && this.editable(event.target)) this.release();
    });
    for (const button of this.buttons) {
      button.addEventListener("pointerdown", (event) => {
        if (!this.available) return;
        event.preventDefault(); this.enable();
        button.setPointerCapture(event.pointerId);
        this.pointers.set(event.pointerId, button.dataset.cameraKey!); this.changed(true);
      });
      button.addEventListener("pointerup", (event) => {
        if (this.pointers.delete(event.pointerId)) this.changed();
      });
      button.addEventListener("pointercancel", () => this.release());
      button.addEventListener("lostpointercapture", (event) => {
        if (this.pointers.delete(event.pointerId)) this.release();
      });
    }
    this.render();
  }

  update(available: boolean, world: number) {
    if (world !== this.world || !available && this.available) {
      this.revision++; this.physical.clear(); this.pointers.clear();
      this.latest = this.motion = undefined; this.enabled = false;
      clearTimeout(this.flushTimer); this.flushTimer = undefined;
    }
    this.world = world; this.available = available; this.render();
  }

  release(disable = false) {
    if (!this.enabled && !this.physical.size && !this.pointers.size) return;
    this.physical.clear(); this.pointers.clear();
    this.motion = undefined;
    clearTimeout(this.flushTimer); this.flushTimer = undefined;
    if (this.available) { this.latest = {keys: [], stop: true, latch: false, world: this.world}; void this.pump(); }
    if (disable) this.enabled = false;
    this.render();
  }

  private editable(target: EventTarget | null) {
    return target instanceof Element && Boolean(target.closest("input,textarea,select,[contenteditable]:not([contenteditable=false])"));
  }

  private enable() {
    if (!this.available || this.enabled) return;
    this.enabled = true; this.onEnable();
    this.latest = {keys: [], stop: true, latch: false, world: this.world};
    this.schedule(); this.render();
  }

  private keys() { return [...new Set([...this.physical, ...this.pointers.values()])].sort(); }

  private changed(latch = false) {
    const movement = {keys: this.keys(), stop: false, latch, world: this.world};
    this.latest = movement;
    if (movement.keys.length && latch) this.motion = movement;
    this.render(); this.schedule();
  }

  private schedule() {
    // Collect a chord before sending; inference still receives native chunk inputs.
    if (this.flushTimer === undefined) this.flushTimer = setTimeout(() => {
      this.flushTimer = undefined; void this.pump();
    }, 32);
  }

  private async pump() {
    if (this.sending) return;
    this.sending = true;
    const revision = this.revision;
    try {
      while (this.available && revision === this.revision && (this.motion || this.latest)) {
        // Keep one latest motion plus its release, rather than a keystroke backlog.
        const movement = this.motion || this.latest!;
        this.motion = undefined;
        if (this.latest === movement) this.latest = undefined;
        await this.send(movement);
      }
    } catch (cause) {
      if (revision === this.revision) {
        this.latest = this.motion = undefined;
        this.onError(cause);
      }
    } finally {
      this.sending = false;
      if (this.available && (this.motion || this.latest)) void this.pump();
    }
  }

  private render() {
    const keys = this.keys();
    for (const button of this.buttons) {
      button.disabled = !this.available;
      const pressed = keys.includes(button.dataset.cameraKey!);
      button.classList.toggle("held", pressed); button.setAttribute("aria-pressed", String(pressed));
    }
    this.toggle.disabled = this.stop.disabled = !this.available;
    this.toggle.textContent = this.enabled ? "Camera keys on · Esc to release" : "Enable camera keys";
    this.toggle.setAttribute("aria-pressed", String(this.enabled));
    this.root.classList.toggle("keys-enabled", this.enabled);
    this.status.textContent = !this.available ? "Start a world to explore." : keys.length
      ? `Holding ${keys.map(key => ({up:"↑",down:"↓",left:"←",right:"→"}[key] || key.toUpperCase())).join(" + ")}`
      : this.enabled ? "WASD to move · arrows to look · combine keys" : "Click the video or enable keys to explore.";
  }
}
