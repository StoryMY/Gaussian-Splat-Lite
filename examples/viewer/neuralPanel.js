import { describeModel, modelFromUrl } from "../postprocessing/modelSource.js";

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
    <select data-model aria-label="PixRestore 模型" disabled><option value="">正在扫描模型…</option></select>
    <button type="button" data-refresh>刷新模型列表</button>
    <small data-model-info role="status">尚未选择模型</small>
    <select data-mode aria-label="PixRestore 增强模式">
      <option value="idle">静止增强</option><option value="continuous">连续增强</option>
    </select>
    <label><input type="checkbox" data-validation> 512×512 验证模式</label>
    <button type="button" data-run>重新处理</button>
    <small data-status role="status">关闭 · 停止操作后增强</small>`;
  parent.prepend(panel);
  const enabled = panel.querySelector("[data-enable]");
  const modelInfo = panel.querySelector("[data-model-info]");
  const modelSelect = panel.querySelector("[data-model]");
  let selectedModel = null;
  let listToken = 0;
  let selectionToken = 0;
  enabled.disabled = true;
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
    if (!selectedModel) {
      enabled.checked = false;
      setStatus("请先选择模型");
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
      state.neural = new IdlePostProcessor({
        mode: mode.value,
        source: new GslFrameSource({
          ...state,
          scene,
          camera,
          helpers,
          controlsScene,
        }),
        processor: new PixRestoreProcessor(selectedModel),
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
  async function selectModel(load) {
    const id = ++selectionToken;
    modelInfo.textContent = "读取模型清单…";
    enabled.disabled = true;
    try {
      const source = await load();
      if (id !== selectionToken) return;
      selectedModel = source;
      modelInfo.textContent = describeModel(source);
      enabled.disabled = false;
      await configure();
    } catch (error) {
      if (id !== selectionToken) return;
      modelInfo.textContent = selectedModel
        ? describeModel(selectedModel)
        : "尚未选择模型";
      selectedModel = null;
      enabled.checked = false;
      await configure();
      setStatus(`选择失败：${error.message}`);
    }
  }
  function loadSelected() {
    const option = modelSelect.selectedOptions[0];
    if (option?.value)
      void selectModel(async () => {
        const source = await modelFromUrl(option.value);
        source.label = option.dataset.label;
        return source;
      });
  }
  async function refreshModels() {
    const id = ++listToken;
    const previous = modelSelect.value;
    modelSelect.disabled = true;
    try {
      const response = await fetch(
        `${import.meta.env.BASE_URL}pixrestore-models.json`,
        { cache: "no-store" },
      );
      if (!response.ok)
        throw new Error(`模型列表加载失败 (${response.status})`);
      const index = await response.json();
      if (id !== listToken) return;
      const custom = new URLSearchParams(location.search).get(
        "postprocessManifest",
      );
      const models = [...index.models];
      if (custom && !models.some((m) => m.url === custom))
        models.unshift({ url: custom, label: "自定义模型" });
      modelSelect.replaceChildren();
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model.url;
        option.dataset.label = model.label;
        option.textContent = `${model.label}${model.precision ? ` · ${model.precision.toUpperCase()}` : ""}`;
        modelSelect.append(option);
      }
      if (!models.length) {
        selectionToken++;
        selectedModel = null;
        enabled.checked = false;
        enabled.disabled = true;
        modelInfo.textContent = "public 下未找到有效模型清单";
        await configure();
      } else {
        modelSelect.disabled = false;
        modelSelect.value = models.some((m) => m.url === previous)
          ? previous
          : models[0].url;
        loadSelected();
      }
      if (index.errors?.length)
        setStatus(`已跳过 ${index.errors.length} 份无效清单`);
    } catch (error) {
      if (id !== listToken) return;
      modelSelect.disabled = !modelSelect.options.length || !modelSelect.value;
      setStatus(`扫描失败：${error.message}`);
    }
  }
  modelSelect.addEventListener("change", loadSelected);
  panel
    .querySelector("[data-refresh]")
    .addEventListener("click", () => void refreshModels());
  void refreshModels();
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
      listToken++;
      selectionToken++;
      selectedModel = null;
      token++;
      panel.remove();
    },
  };
}
