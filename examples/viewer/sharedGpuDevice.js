// App owns the device so backend switches do not invalidate ORT's shared context.
let pending;
export function getSharedGpuDevice() {
  pending ??= (async () => {
    const adapter = await navigator.gpu?.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) return undefined;
    const device = await adapter.requestDevice({
      requiredFeatures: [...adapter.features],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    device.lost.then((info) =>
      console.error("Shared WebGPU device lost:", info.reason, info.message),
    );
    device.addEventListener("uncapturederror", (event) =>
      console.error("Shared WebGPU error:", event.error.message),
    );
    return device;
  })();
  return pending;
}
