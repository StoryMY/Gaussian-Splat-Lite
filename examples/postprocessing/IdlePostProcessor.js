/**
 * Framework-independent scheduler. Source.capture() returns a borrowed GPU frame;
 * processor.process(frame) returns an owned result; presenter shows/hides it.
 * Invalidation never cancels submitted GPU work: it discards stale results instead.
 */
export class IdlePostProcessor {
  constructor({
    source,
    processor,
    presenter,
    onStatus = () => {},
    idleMs = 450,
  }) {
    Object.assign(this, { source, processor, presenter, onStatus, idleMs });
    this.enabled = false;
    this.disposed = false;
    this.revision = 0;
    this.busy = null;
    this.timer = null;
    this.result = null;
  }
  setEnabled(enabled) {
    this.enabled = enabled;
    this.invalidate();
    if (!enabled) this.onStatus("关闭");
  }
  invalidate() {
    this.revision++;
    clearTimeout(this.timer);
    this.presenter.hide();
    this.result?.release();
    this.result = null;
    if (this.enabled && !this.disposed) {
      this.onStatus("等待画面稳定…");
      this.timer = setTimeout(() => this.run(), this.idleMs);
    }
  }
  async run() {
    clearTimeout(this.timer);
    if (!this.enabled || this.disposed) return;
    if (this.busy) {
      this.timer = setTimeout(() => this.run(), this.idleMs);
      return;
    }
    const revision = this.revision;
    const task = async () => {
      let frame;
      let result;
      try {
        this.onStatus("加载 / 推理中…");
        await this.processor.initialize(this.source.device);
        if (revision !== this.revision || this.disposed) return;
        const start = performance.now();
        frame = await this.source.capture();
        if (revision !== this.revision || this.disposed) return;
        result = await this.processor.process(frame);
        if (revision !== this.revision || this.disposed) return;
        const inferenceMs = performance.now() - start;
        await this.presenter.prepare?.(result, frame);
        if (revision !== this.revision || this.disposed) return;
        this.presenter.show(result, frame);
        this.failed = false;
        this.result?.release();
        this.result = result;
        result = null;
        this.onStatus(
          `已增强 · ${inferenceMs.toFixed(1)} ms（含捕获、GPU 等待，不含加载及呈现）`,
        );
      } catch (error) {
        if (revision === this.revision && !this.disposed) {
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
