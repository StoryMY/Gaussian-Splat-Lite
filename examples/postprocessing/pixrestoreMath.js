export function fitRect(width, height, size) {
  const scale = size / Math.max(width, height);
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  return {
    x: Math.floor((size - w) / 2),
    y: Math.floor((size - h) / 2),
    width: w,
    height: h,
  };
}

// Same seed and Box–Muller sequence as onnx-webgpu-runner; uploaded once per model.
export function fixedNoise(count, seed = 42) {
  let state = seed >>> 0;
  const random = () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
  return Float32Array.from(
    { length: count },
    () =>
      Math.sqrt(-2 * Math.log(Math.max(random(), Number.MIN_VALUE))) *
      Math.cos(2 * Math.PI * random()),
  );
}

export function validateManifest(m) {
  const shape = (a, b) => Array.isArray(a) && a.join() === b.join();
  const size = m.resolution;
  const d = m.dino?.inputSize;
  if (
    m.format !== "pixrestore-webgpu-v1" ||
    !["fp16", "fp32"].includes(m.precision) ||
    ![384, 512].includes(size) ||
    !Number.isInteger(d) ||
    d <= 0 ||
    d > 1024 ||
    !shape(m.pixelDit?.input?.shape, [1, 6, size, size]) ||
    !shape(m.dino?.input?.shape, [1, 3, d, d]) ||
    !m.singleStepSampling?.usesAelq ||
    m.singleStepSampling.lqConditionStrength !== 1 ||
    m.singleStepSampling.dinoConditionStrength !== 1 ||
    m.pixelDit?.timestep?.value !== 1 ||
    !m.models?.dinoEncoder ||
    !m.models?.pixelDit ||
    !m.dino?.input?.name ||
    !m.pixelDit?.input?.name ||
    !m.pixelDit?.timestep?.name ||
    !m.pixelDit?.output ||
    !Array.isArray(m.dino?.outputs) ||
    !m.dino.outputs.length ||
    m.dino.outputs.some((n) => typeof n !== "string") ||
    new Set(m.dino.outputs).size !== m.dino.outputs.length
  ) {
    throw new Error(
      "不支持的 PixRestore manifest；需要单步 compat FP32/FP16 模型",
    );
  }
  const n = m.dino.input.preprocessing?.normalization;
  if (
    !n ||
    n.mean?.length !== 3 ||
    n.std?.length !== 3 ||
    !n.mean.every(Number.isFinite) ||
    !n.std.every((v) => Number.isFinite(v) && v > 0)
  ) {
    throw new Error("DINO normalization 无效");
  }
  return m;
}
