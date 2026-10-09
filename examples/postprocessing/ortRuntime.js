import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import * as ort from "onnxruntime-web/webgpu";

ort.env.wasm.wasmPaths = { wasm: wasmUrl };
let device;
let tail = Promise.resolve();
// ORT's WebGPU context is shared across sessions: serialize initialization,
// inference and release, as in Visionary's exclusive GPU I/O execution.
export function exclusive(task) {
  const result = tail.then(task);
  tail = result.catch(() => {});
  return result;
}
export function bindDevice(value) {
  if (device && device !== value)
    throw new Error("ORT 已绑定另一 GPUDevice；请刷新页面");
  if (!device) {
    ort.env.webgpu.device = value;
    device = value;
  }
  return ort;
}
