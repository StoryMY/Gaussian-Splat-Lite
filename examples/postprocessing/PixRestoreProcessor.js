import { bindDevice, exclusive } from "./ortRuntime.js";
import { modelFromUrl } from "./modelSource.js";
import { fitRect, fixedNoise, validateManifest } from "./pixrestoreMath.js";

/** No renderer imports. Input: top-left GPUTexture (premultiplied RGB on black).
 * Output: owned normalized sRGB planar RGB GPUBuffer + content rect.
 */
export class PixRestoreProcessor {
  constructor(modelSource) {
    this.modelSource = modelSource;
    this.buffers = [];
    this.tensors = [];
  }
  initialize(device) {
    this.initializing ??= exclusive(async () => {
      this.device = device;
      this.ort = bindDevice(device);
      const source =
        typeof this.modelSource === "string"
          ? await modelFromUrl(this.modelSource)
          : this.modelSource;
      const m = validateManifest(source.manifest);
      this.manifest = m;
      if (m.precision === "fp16" && !device.features.has("shader-f16")) {
        throw new Error("设备不支持 shader-f16，请选择 FP32");
      }
      const options = {
        executionProviders: [{ name: "webgpu", device }],
        preferredOutputLocation: "gpu-buffer",
        graphOptimizationLevel: "disabled",
      };
      this.dino = await this.ort.InferenceSession.create(
        await source.loadModel(m.models.dinoEncoder),
        options,
      );
      this.pixel = await this.ort.InferenceSession.create(
        await source.loadModel(m.models.pixelDit),
        options,
      );
      if (
        !m.dino.outputs.every(
          (n) =>
            this.dino.outputNames.includes(n) &&
            this.pixel.inputNames.includes(n),
        ) ||
        !this.pixel.outputNames.includes(m.pixelDit.output)
      )
        throw new Error("ONNX 输入输出与 manifest 不匹配");
      this.type = m.precision === "fp16" ? "float16" : "float32";
      this.scalar = m.precision === "fp16" ? "f16" : "f32";
      this.bytes = m.precision === "fp16" ? 2 : 4;
      const allocate = (
        size,
        usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      ) => {
        const b = device.createBuffer({
          size: Math.ceil(size / 16) * 16,
          usage,
        });
        this.buffers.push(b);
        return b;
      };
      this.pixelBuffer = allocate(6 * m.resolution ** 2 * this.bytes);
      this.dinoBuffer = allocate(3 * m.dino.inputSize ** 2 * this.bytes);
      this.noiseBuffer = allocate(3 * m.resolution ** 2 * 4);
      device.queue.writeBuffer(
        this.noiseBuffer,
        0,
        fixedNoise(3 * m.resolution ** 2),
      );
      this.params = allocate(
        32,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      );
      this.pixelTensor = this.ort.Tensor.fromGpuBuffer(this.pixelBuffer, {
        dataType: this.type,
        dims: m.pixelDit.input.shape,
      });
      this.dinoTensor = this.ort.Tensor.fromGpuBuffer(this.dinoBuffer, {
        dataType: this.type,
        dims: m.dino.input.shape,
      });
      this.timestep = new this.ort.Tensor(
        "float32",
        Float32Array.of(m.pixelDit.timestep.value),
        [1],
      );
      this.tensors.push(this.pixelTensor, this.dinoTensor, this.timestep);
      this.preprocess = await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: device.createShaderModule({ code: this.preprocessCode() }),
          entryPoint: "main",
        },
      });
    });
    return this.initializing;
  }
  preprocessCode() {
    const m = this.manifest;
    const n = m.dino.input.preprocessing.normalization;
    const vec = (v) => `vec3f(${v.map((x) => Number(x).toFixed(9)).join(",")})`;
    return `
${this.scalar === "f16" ? "enable f16;" : ""}
struct Params { rect: vec4f, size: vec2f, decodeSrgb: u32, pad: u32 }
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> pixel: array<${this.scalar}>;
@group(0) @binding(2) var<storage, read_write> dino: array<${this.scalar}>;
@group(0) @binding(3) var<storage, read> noise: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
fn loadRGB(at: vec2i) -> vec3f {
  var c = textureLoad(image, clamp(at, vec2i(0), vec2i(p.size)-1), 0).rgb;
  // An sRGB GPU texture decodes on load; restore display RGB before normalization.
  if (p.decodeSrgb != 0u) {
    c = select(12.92*c, 1.055*pow(max(c, vec3f(0)), vec3f(1.0/2.4))-0.055, c > vec3f(0.0031308));
  }
  return c;
}
fn sampleRGB(at: vec2f) -> vec3f {
  let uv = (at-p.rect.xy)/p.rect.zw;
  if (any(uv < vec2f(0)) || any(uv >= vec2f(1))) { return vec3f(0); }
  let pos = uv*p.size-0.5;
  let a = vec2i(floor(pos)); let f = fract(pos);
  return mix(mix(loadRGB(a), loadRGB(a+vec2i(1,0)), f.x),
             mix(loadRGB(a+vec2i(0,1)), loadRGB(a+vec2i(1,1)), f.x), f.y);
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i < ${m.resolution ** 2}u) {
    let c = sampleRGB(vec2f(f32(i % ${m.resolution}u)+0.5, f32(i / ${m.resolution}u)+0.5))*2.0-1.0;
    for (var k=0u; k<3u; k++) {
      pixel[k*${m.resolution ** 2}u+i] = ${this.scalar}(c[k]);
      pixel[(k+3u)*${m.resolution ** 2}u+i] = ${this.scalar}(noise[k*${m.resolution ** 2}u+i]);
    }
  }
  if (i < ${m.dino.inputSize ** 2}u) {
    let at = (vec2f(f32(i % ${m.dino.inputSize}u)+0.5, f32(i / ${m.dino.inputSize}u)+0.5)) * ${m.resolution.toFixed(1)}/${m.dino.inputSize.toFixed(1)};
    let c = (sampleRGB(at)-${vec(n.mean)})/${vec(n.std)};
    for (var k=0u; k<3u; k++) { dino[k*${m.dino.inputSize ** 2}u+i] = ${this.scalar}(c[k]); }
  }
}`;
  }
  process(frame) {
    return exclusive(async () => {
      if (frame.device !== this.device)
        throw new Error("输入纹理和模型必须共用 GPUDevice");
      const m = this.manifest;
      const rect = fitRect(frame.width, frame.height, m.resolution);
      const values = new Float32Array([
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        frame.width,
        frame.height,
        0,
        0,
      ]);
      new Uint32Array(values.buffer)[6] =
        frame.colorSpace === "linear-srgb" ? 1 : 0;
      this.device.queue.writeBuffer(this.params, 0, values);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.preprocess);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: this.preprocess.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: frame.texture.createView() },
            { binding: 1, resource: { buffer: this.pixelBuffer } },
            { binding: 2, resource: { buffer: this.dinoBuffer } },
            { binding: 3, resource: { buffer: this.noiseBuffer } },
            { binding: 4, resource: { buffer: this.params } },
          ],
        }),
      );
      pass.dispatchWorkgroups(
        Math.ceil(Math.max(m.resolution ** 2, m.dino.inputSize ** 2) / 64),
      );
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      let features;
      let outputs;
      try {
        features = await this.dino.run({
          [m.dino.input.name]: this.dinoTensor,
        });
        outputs = await this.pixel.run({
          [m.pixelDit.input.name]: this.pixelTensor,
          [m.pixelDit.timestep.name]: this.timestep,
          ...Object.fromEntries(
            m.dino.outputs.map((name) => [name, features[name]]),
          ),
        });
        const tensor = outputs[m.pixelDit.output];
        if (
          tensor.location !== "gpu-buffer" ||
          tensor.type !== this.type ||
          tensor.dims.join() !== [1, 3, m.resolution, m.resolution].join()
        ) {
          throw new Error("模型输出必须是 GPU 上的 NCHW RGB");
        }
        // Synchronize for honest end-to-end timing, without downloading image tensors.
        await this.device.queue.onSubmittedWorkDone();
        const owned = outputs;
        outputs = null;
        return {
          device: this.device,
          buffer: tensor.gpuBuffer,
          width: m.resolution,
          height: m.resolution,
          dataType: this.type,
          rect,
          release: () => {
            for (const t of Object.values(owned)) t.dispose();
          },
        };
      } finally {
        for (const t of Object.values(features ?? {})) t.dispose();
        for (const t of Object.values(outputs ?? {})) t.dispose();
      }
    });
  }
  async dispose() {
    await this.initializing?.catch(() => {});
    await exclusive(async () => {
      for (const t of this.tensors) t.dispose();
      for (const b of this.buffers) b.destroy();
      await this.pixel?.release();
      await this.dino?.release();
    });
  }
}
