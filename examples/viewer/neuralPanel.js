/** Thin viewer UI. Renderer/model code lives behind independent adapters. */
export function installNeuralPanel({
  parent,
  viewport,
  getState,
  scene,
  camera,
  helpers,
  controlsScene,
  hasModel,
  onValidationModeChange,
}) {
  const panel = document.createElement("section");
  panel.className = "neural-panel";
  panel.innerHTML = `
    <label><input type="checkbox" data-enable> PixRestore 后处理</label>
    <select data-model aria-label="PixRestore 精度">
      <option value="fp16">FP16 compat</option><option value="fp32">FP32 compat</option>
    </select>
    <select data-mode aria-label="PixRestore 增强模式">
      <option value="idle">静止增强</option><option value="continuous">连续增强</option>
    </select>
    <label><input type="checkbox" data-validation> 512×512 验证模式</label>
    <button type="button" data-run>重新处理</button>
    <small data-status role="status">关闭 · 停止操作后增强</small>`;
  parent.prepend(panel);
  const enabled = panel.querySelector("[data-enable]");
  const model = panel.querySelector("[data-model]");
  const mode = panel.querySelector("[data-mode]");
  const run = panel.querySelector("[data-run]");
  const status = panel.querySelector("[data-status]");
  let token = 0;
  const setStatus = (message) => {
    status.textContent = message;
  };
  async function configure() {
    const id = ++token;
    const state = getState();
    const old = state.neural;
    if (old) await old.dispose().catch(console.error);
    if (state.neural === old) state.neural = null;
    if (id !== token) return;
    if (!enabled.checked) {
      setStatus("关闭");
      return;
    }
    if (!state.renderer.backend?.isWebGPUBackend) {
      enabled.checked = false;
      setStatus("需要原生 WebGPU");
      return;
    }
    if (!hasModel()) {
      enabled.checked = false;
      setStatus("请先加载场景");
      return;
    }
    setStatus("准备后处理…");
    try {
      const [
        { IdlePostProcessor },
        { GslFrameSource },
        { GpuPresenter },
        { PixRestoreProcessor },
      ] = await Promise.all([
        import("../postprocessing/IdlePostProcessor.js"),
        import("../postprocessing/GslFrameSource.js"),
        import("../postprocessing/GpuPresenter.js"),
        import("../postprocessing/PixRestoreProcessor.js"),
      ]);
      if (id !== token || state.disposed || state !== getState()) return;
      const url =
        new URLSearchParams(location.search).get("postprocessManifest") ||
        `/models/pixrestore-s-ft-d6-${model.value}-compat/pixrestore-manifest.json`;
      state.neural = new IdlePostProcessor({
        mode: mode.value,
        source: new GslFrameSource({
          ...state,
          scene,
          camera,
          helpers,
          controlsScene,
        }),
        processor: new PixRestoreProcessor(url),
        presenter: new GpuPresenter(state.renderer.backend.device, viewport),
        onStatus: (message) => {
          if (id === token) setStatus(message);
        },
      });
      state.neural.setEnabled(true);
    } catch (error) {
      if (id === token) setStatus(`失败：${error.message}`);
    }
  }
  panel
    .querySelector("[data-validation]")
    .addEventListener("change", (event) => {
      onValidationModeChange(event.target.checked);
    });
  enabled.addEventListener("change", () => void configure());
  model.addEventListener("change", () => void configure());
  mode.addEventListener("change", () => getState().neural?.setMode(mode.value));
  run.addEventListener("click", () => {
    if (getState().neural && !getState().neural.failed) {
      getState().neural.invalidate({ reset: true });
      void getState().neural.run();
    } else if (enabled.checked) void configure();
  });
  return {
    invalidate(options) {
      getState().neural?.invalidate(options);
    },
    backendChanged() {
      token++;
      enabled.checked = false;
      getState().neural?.setEnabled(false);
      setStatus("关闭 · 切换渲染器后可重新开启");
    },
    dispose() {
      token++;
      panel.remove();
    },
  };
}
