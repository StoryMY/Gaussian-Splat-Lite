/** Renderer-independent presenter: planar [-1,1] sRGB RGB -> canvas. No readback. */
export class GpuPresenter {
  constructor(device, container) {
    this.device = device;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "neural-output";
    this.canvas.hidden = true;
    this.canvas.setAttribute("aria-label", "增强后的渲染结果");
    container.append(this.canvas);
    this.context = this.canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device,
      format: this.format,
      alphaMode: "opaque",
    });
    this.pipelines = new Map();
    this.params = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
  async prepare(result, frame) {
    this.overlay = frame.overlay;
    const half = result.dataType === "float16";
    const scalar = half ? "f16" : "f32";
    let pipeline = this.pipelines.get(scalar);
    if (!pipeline) {
      const module = this.device.createShaderModule({
        code: `
${half ? "enable f16;" : ""}
struct Params { rect: vec4f, size: vec2f, pad: vec2f }
@group(0) @binding(0) var<storage, read> rgb: array<${scalar}>;
@group(0) @binding(1) var<uniform> p: Params;
@group(0) @binding(2) var overlay: texture_2d<f32>;
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> Vertex {
  let xy = array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var v: Vertex; v.position=vec4f(xy[i],0,1); v.uv=vec2f((xy[i].x+1)*0.5,(1-xy[i].y)*0.5); return v;
}
fn readPixel(at: vec2i) -> vec3f {
  let pos=clamp(at,vec2i(p.rect.xy),vec2i(p.rect.xy+p.rect.zw)-1);
  let i=u32(pos.y)*u32(p.size.x)+u32(pos.x); let n=u32(p.size.x*p.size.y);
  return vec3f(f32(rgb[i]),f32(rgb[n+i]),f32(rgb[2u*n+i]));
}
@fragment fn fs(v: Vertex) -> @location(0) vec4f {
  let pos=p.rect.xy+v.uv*p.rect.zw-0.5; let a=vec2i(floor(pos)); let f=fract(pos);
  let c=mix(mix(readPixel(a),readPixel(a+vec2i(1,0)),f.x),mix(readPixel(a+vec2i(0,1)),readPixel(a+vec2i(1,1)),f.x),f.y);
  let size = textureDimensions(overlay);
  let at = clamp(vec2i(v.uv*vec2f(size)), vec2i(0), vec2i(size)-1);
  let helpers = textureLoad(overlay, at, 0);
  let restored = clamp(c*0.5+0.5,vec3f(0),vec3f(1));
  return vec4f(helpers.rgb + restored * (1.0-helpers.a),1);
}`,
      });
      pipeline = await this.device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: {
          module,
          entryPoint: "fs",
          targets: [{ format: this.format }],
        },
        primitive: { topology: "triangle-list" },
      });
      this.pipelines.set(scalar, pipeline);
    }
  }
  show(result) {
    const pipeline = this.pipelines.get(
      result.dataType === "float16" ? "f16" : "f32",
    );
    if (!pipeline) throw new Error("显示管线尚未初始化");
    this.canvas.width = Math.max(
      1,
      Math.round(
        this.canvas.parentElement.clientWidth * Math.min(devicePixelRatio, 2),
      ),
    );
    this.canvas.height = Math.max(
      1,
      Math.round(
        this.canvas.parentElement.clientHeight * Math.min(devicePixelRatio, 2),
      ),
    );
    const r = result.rect;
    this.device.queue.writeBuffer(
      this.params,
      0,
      Float32Array.of(
        r.x,
        r.y,
        r.width,
        r.height,
        result.width,
        result.height,
        0,
        0,
      ),
    );
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: result.buffer } },
          { binding: 1, resource: { buffer: this.params } },
          { binding: 2, resource: this.overlay.createView() },
        ],
      }),
    );
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.canvas.hidden = false;
  }
  hide() {
    this.canvas.hidden = true;
  }
  dispose() {
    this.context.unconfigure();
    this.params.destroy();
    this.canvas.remove();
  }
}
