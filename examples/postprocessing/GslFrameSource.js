import { SplatCapture } from "gaussian-splat-lite";
import * as THREE from "three";
/** Change scene state only during the synchronous draw, never across sorting awaits. */
class ViewerCapture extends SplatCapture {
  constructor(options, helpers, overlay = false, controlsScene = null) {
    super(options);
    Object.assign(this, { helpers, overlay, controlsScene });
  }
  renderCapture(view, draw, ...args) {
    return super.renderCapture(
      view,
      () => {
        if (!this.overlay) {
          const visible = this.helpers.visible;
          try {
            this.helpers.visible = false;
            draw();
          } finally {
            this.helpers.visible = visible;
          }
          return;
        }
        const { renderer, captureRenderer, controlsScene } = this;
        const material = captureRenderer.material;
        const background = view.scene.background;
        const clearColor = renderer.getClearColor(new THREE.Color());
        const clearAlpha = renderer.getClearAlpha();
        const keys = [
          "blending",
          "blendSrc",
          "blendDst",
          "blendEquation",
          "blendSrcAlpha",
          "blendDstAlpha",
          "blendEquationAlpha",
        ];
        const saved = Object.fromEntries(
          keys.map((key) => [key, material[key]]),
        );
        const sortedBlending = captureRenderer.sortedBlending;
        try {
          view.scene.background = null;
          renderer.setClearColor(0, 0);
          // Helpers write their usual depth. Splats in front attenuate both their
          // RGB and coverage, without adding any Splat color to this layer.
          captureRenderer.sortedBlending = THREE.CustomBlending;
          Object.assign(material, {
            blending: THREE.CustomBlending,
            blendSrc: THREE.ZeroFactor,
            blendDst: THREE.OneMinusSrcAlphaFactor,
            blendEquation: THREE.AddEquation,
            blendSrcAlpha: THREE.ZeroFactor,
            blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
            blendEquationAlpha: THREE.AddEquation,
          });
          draw();
          if (controlsScene) {
            renderer.autoClear = false;
            renderer.render(controlsScene, view.camera);
          }
        } finally {
          Object.assign(material, saved);
          captureRenderer.sortedBlending = sortedBlending;
          view.scene.background = background;
          renderer.setClearColor(clearColor, clearAlpha);
        }
      },
      ...args,
    );
  }
}
/** Only this adapter knows Gaussian-Splat-Lite / Three.js internals.
 * Capture retains the camera aspect; the processor owns letterboxing.
 */
export class GslFrameSource {
  constructor({
    renderer,
    splatRenderer,
    scene,
    camera,
    helpers,
    controlsScene,
  }) {
    Object.assign(this, {
      renderer,
      splatRenderer,
      scene,
      camera,
      helpers,
      controlsScene,
    });
    this.device = renderer.backend.device;
  }
  async capture() {
    // Both captures must use the same pose while the live camera keeps moving.
    this.camera.updateWorldMatrix(true, false);
    const camera = this.camera.clone();
    camera.matrixAutoUpdate = false;
    camera.matrix.copy(this.camera.matrixWorld);
    const width = Math.max(1, this.renderer.domElement.width);
    const height = Math.max(1, this.renderer.domElement.height);
    // Keep capture cost bounded; model-specific resize belongs in the processor.
    const scale = Math.min(1, 1024 / Math.max(width, height));
    const w = Math.max(1, Math.round(width * scale));
    const h = Math.max(1, Math.round(height * scale));
    if (!this.captureHelper || this.width !== w || this.height !== h) {
      this.captureHelper?.dispose();
      this.captureHelper = new ViewerCapture(
        {
          splatRenderer: this.splatRenderer,
          target: { width: w, height: h },
        },
        this.helpers,
      );
      this.width = w;
      this.height = h;
    }
    const target = await this.captureHelper.renderTarget({
      scene: this.scene,
      camera,
    });
    if (
      !this.overlayHelper ||
      this.overlayWidth !== width ||
      this.overlayHeight !== height
    ) {
      this.overlayHelper?.dispose();
      this.overlayHelper = new ViewerCapture(
        {
          splatRenderer: this.splatRenderer,
          target: {
            width,
            height,
            type: THREE.HalfFloatType,
            colorSpace: THREE.SRGBColorSpace,
          },
        },
        this.helpers,
        true,
        this.controlsScene,
      );
      this.overlayWidth = width;
      this.overlayHeight = height;
    }
    const overlayTarget = await this.overlayHelper.renderTarget({
      scene: this.scene,
      camera,
    });
    const overlay = this.renderer.backend.get(overlayTarget.texture).texture;
    const texture = this.renderer.backend.get(target.texture).texture;
    if (!texture) throw new Error("无法获取 Three.js 离屏 GPUTexture");
    return {
      device: this.device,
      texture,
      overlay,
      width: w,
      height: h,
      colorSpace: texture.format.endsWith("-srgb") ? "linear-srgb" : "srgb",
    };
  }
  dispose() {
    this.captureHelper?.dispose();
    this.overlayHelper?.dispose();
  }
}
