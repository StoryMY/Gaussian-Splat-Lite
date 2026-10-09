import { SplatCapture } from "gaussian-splat-lite";
/** Only this adapter knows Gaussian-Splat-Lite / Three.js internals.
 * Capture retains the camera aspect; the processor owns letterboxing.
 */
export class GslFrameSource {
  constructor({ renderer, splatRenderer, scene, camera }) {
    Object.assign(this, { renderer, splatRenderer, scene, camera });
    this.device = renderer.backend.device;
  }
  async capture() {
    const width = Math.max(1, this.renderer.domElement.width);
    const height = Math.max(1, this.renderer.domElement.height);
    // Keep capture cost bounded; model-specific resize belongs in the processor.
    const scale = Math.min(1, 1024 / Math.max(width, height));
    const w = Math.max(1, Math.round(width * scale));
    const h = Math.max(1, Math.round(height * scale));
    if (!this.captureHelper || this.width !== w || this.height !== h) {
      this.captureHelper?.dispose();
      this.captureHelper = new SplatCapture({
        splatRenderer: this.splatRenderer,
        target: { width: w, height: h },
      });
      this.width = w;
      this.height = h;
    }
    const target = await this.captureHelper.renderTarget({
      scene: this.scene,
      camera: this.camera,
    });
    const texture = this.renderer.backend.get(target.texture).texture;
    if (!texture) throw new Error("无法获取 Three.js 离屏 GPUTexture");
    return {
      device: this.device,
      texture,
      width: w,
      height: h,
      colorSpace: texture.format.endsWith("-srgb") ? "linear-srgb" : "srgb",
    };
  }
  dispose() {
    this.captureHelper?.dispose();
  }
}
