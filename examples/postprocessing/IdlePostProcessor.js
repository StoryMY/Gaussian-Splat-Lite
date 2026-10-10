/**
 * Framework-independent scheduler. Source.capture() returns a borrowed GPU frame;
 * processor.process(frame) returns an owned result; presenter shows/hides it.
 * One task at a time. Continuous mode presents completed frames and captures the
 * latest view next; idle mode discards frames invalidated during inference.
 */
export class IdlePostProcessor {
  constructor({
    source,
    processor,
    presenter,
    onStatus = () => {},
    idleMs = 100,
    mode = "idle",
  }) {
    Object.assign(this, {
      source,
      processor,
      presenter,
      onStatus,
      idleMs,
      mode,
    });
    this.enabled = false;
    this.disposed = false;
    this.revision = 0;
    this.generation = 0;
    this.busy = null;
    this.timer = null;
    this.result = null;
  }
  setEnabled(enabled) {
    this.enabled = enabled;
    this.invalidate({ reset: true });
    if (!enabled) this.onStatus("关闭");
  }
  setMode(mode) {
    if (!["idle", "continuous"].includes(mode)) throw new Error("未知增强模式");
    if (this.mode === mode) return;
    this.mode = mode;
    this.invalidate({ reset: true });
  }
  schedule(delay) {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.run();
    }, delay);
  }
  invalidate({ reset = false } = {}) {
    this.revision++;
    if (reset) {
      this.generation++;
      this.failed = false;
    }
    if (reset || this.mode === "idle" || !this.enabled || this.disposed) {
      clearTimeout(this.timer);
      this.timer = null;
      this.presenter.hide();
      this.result?.release();
      this.result = null;
    }
    if (!this.enabled || this.disposed || this.failed) return;
    if (this.mode === "idle") {
      this.onStatus("等待画面稳定…");
      this.schedule(this.idleMs);
    } else if (!this.busy && this.timer === null) {
      this.schedule(0);
    }
  }
  async run() {
    clearTimeout(this.timer);
    this.timer = null;
    if (!this.enabled || this.disposed || this.failed) return;
    if (this.busy) {
      if (this.mode === "idle") this.schedule(this.idleMs);
      return;
    }
    const revision = this.revision;
    const generation = this.generation;
    const stale = () =>
      this.disposed ||
      !this.enabled ||
      generation !== this.generation ||
      (this.mode === "idle" && revision !== this.revision);
    const task = async () => {
      let frame;
      let result;
      try {
        if (!this.result) this.onStatus("加载 / 推理中…");
        await this.processor.initialize(this.source.device);
        if (stale()) return;
        const start = performance.now();
        frame = await this.source.capture();
        if (stale()) return;
        result = await this.processor.process(frame);
        if (stale()) return;
        const inferenceMs = performance.now() - start;
        await this.presenter.prepare?.(result, frame);
        if (stale()) return;
        this.presenter.show(result, frame);
        this.failed = false;
        this.result?.release();
        this.result = result;
        result = null;
        this.onStatus(
          `已增强 · ${inferenceMs.toFixed(1)} ms（含捕获、GPU 等待，不含加载及呈现）`,
        );
      } catch (error) {
        if (!stale()) {
          this.failed = true;
          this.presenter.hide();
          this.onStatus(`失败：${error.message}`);
        }
      } finally {
        result?.release();
        frame?.release?.();
      }
    };
    this.busy = task();
    await this.busy;
    this.busy = null;
    if (
      this.enabled &&
      !this.disposed &&
      !this.failed &&
      this.mode === "continuous"
    ) {
      this.schedule(0);
    }
  }
  dispose() {
    this.disposing ??= this.disposeOnce();
    return this.disposing;
  }
  async disposeOnce() {
    this.disposed = true;
    this.enabled = false;
    this.invalidate();
    await this.busy;
    try {
      await this.processor.dispose();
    } finally {
      this.presenter.dispose();
      this.source.dispose();
    }
  }
}
