import assert from "node:assert/strict";
import test from "node:test";
import { IdlePostProcessor } from "../examples/postprocessing/IdlePostProcessor.js";
import {
  fitRect,
  fixedNoise,
  validateManifest,
} from "../examples/postprocessing/pixrestoreMath.js";

function deferred() {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
}
function harness(options = {}) {
  const gate = deferred();
  const events = [];
  const result = {
    release() {
      events.push("release-result");
    },
  };
  const source = {
    device: {},
    async capture() {
      events.push("capture");
      return {
        release() {
          events.push("release-frame");
        },
      };
    },
    dispose() {
      events.push("dispose-source");
    },
  };
  const processor = {
    async initialize() {},
    async process() {
      events.push("process");
      return gate.promise;
    },
    async dispose() {
      events.push("dispose-processor");
    },
  };
  const presenter = {
    show() {
      events.push("show");
    },
    hide() {
      events.push("hide");
    },
    dispose() {
      events.push("dispose-presenter");
    },
  };
  const scheduler = new IdlePostProcessor({
    source,
    processor,
    presenter,
    idleMs: 100000,
    ...options,
  });
  scheduler.setEnabled(true);
  return { gate, events, result, scheduler };
}
async function started(h) {
  const task = h.scheduler.run();
  await new Promise((r) => setImmediate(r));
  return { task };
}
test("camera invalidation discards an in-flight result and releases its frame", async () => {
  const h = harness();
  const { task } = await started(h);
  h.scheduler.invalidate();
  h.gate.resolve(h.result);
  await task;
  assert.ok(!h.events.includes("show"));
  assert.ok(h.events.includes("release-result"));
  assert.ok(h.events.includes("release-frame"));
  await h.scheduler.dispose();
});
test("valid result displays once; disabling immediately hides and releases it", async () => {
  const h = harness();
  const { task } = await started(h);
  h.gate.resolve(h.result);
  await task;
  assert.equal(h.events.filter((e) => e === "show").length, 1);
  h.scheduler.setEnabled(false);
  assert.equal(h.events.at(-1), "release-result");
  await h.scheduler.dispose();
});
test("dispose waits for inference before destroying source and processor", async () => {
  const h = harness();
  const { task } = await started(h);
  const disposed = h.scheduler.dispose();
  assert.ok(!h.events.includes("dispose-source"));
  h.gate.resolve(h.result);
  await Promise.all([task, disposed]);
  assert.ok(!h.events.includes("show"));
  assert.ok(
    h.events.indexOf("release-result") < h.events.indexOf("dispose-source"),
  );
});
test("repeated requests cannot start concurrent inference", async () => {
  const h = harness();
  const { task } = await started(h);
  await h.scheduler.run();
  assert.equal(h.events.filter((e) => e === "process").length, 1);
  h.scheduler.setEnabled(false);
  h.gate.resolve(h.result);
  await task;
  await h.scheduler.dispose();
});
test("16:9 and portrait frames preserve their aspect through letterboxing", () => {
  assert.deepEqual(fitRect(1920, 1080, 512), {
    x: 0,
    y: 112,
    width: 512,
    height: 288,
  });
  assert.deepEqual(fitRect(1080, 1920, 512), {
    x: 112,
    y: 0,
    width: 288,
    height: 512,
  });
  assert.deepEqual(fitRect(512, 512, 512), {
    x: 0,
    y: 0,
    width: 512,
    height: 512,
  });
});
test("noise is deterministic, finite and approximately standard normal", () => {
  const values = fixedNoise(30000);
  assert.deepEqual(values, fixedNoise(values.length));
  assert.ok(values.every(Number.isFinite));
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  assert.ok(Math.abs(mean) < 0.03);
  assert.ok(Math.abs(variance - 1) < 0.05);
});
test("unsupported manifests fail before allocating GPU resources", () => {
  assert.throws(() => validateManifest({}), /manifest/);
  assert.throws(
    () =>
      validateManifest({ format: "pixrestore-webgpu-v1", precision: "int4" }),
    /manifest/,
  );
});

// The overlay is borrowed from the source, and must remain alive until presentation.
test("presenter receives the captured overlay before the frame is released", async () => {
  const h = harness();
  const frame = {
    overlay: {},
    release() {
      h.events.push("release-frame");
    },
  };
  h.scheduler.source.capture = async () => frame;
  h.scheduler.presenter.prepare = async (result, captured) => {
    assert.equal(captured, frame);
    assert.equal(result, h.result);
    assert.ok(!h.events.includes("release-frame"));
  };
  h.scheduler.presenter.show = (result, captured) => {
    assert.equal(captured.overlay, frame.overlay);
    assert.equal(result, h.result);
    assert.ok(!h.events.includes("release-frame"));
    h.events.push("show");
  };
  const { task } = await started(h);
  h.gate.resolve(h.result);
  await task;
  assert.ok(h.events.indexOf("show") < h.events.indexOf("release-frame"));
  await h.scheduler.dispose();
});

test("idle enhancement debounces scene changes for 100 ms by default", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness({ idleMs: undefined });
  t.mock.timers.tick(99);
  assert.ok(!h.events.includes("capture"));
  h.scheduler.invalidate();
  t.mock.timers.tick(99);
  assert.ok(!h.events.includes("capture"));
  t.mock.timers.tick(1);
  await new Promise((r) => setImmediate(r));
  assert.equal(h.events.filter((e) => e === "capture").length, 1);
  h.gate.resolve(h.result);
  await h.scheduler.busy;
  await h.scheduler.dispose();
});

test("continuous mode presents completed frames and captures only the latest view next", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness({ mode: "continuous" });
  let view = 0;
  const captured = [];
  const second = deferred();
  h.scheduler.source.capture = async () => {
    captured.push(view);
    return {};
  };
  let calls = 0;
  h.scheduler.processor.process = () =>
    ++calls === 1 ? h.gate.promise : second.promise;
  t.mock.timers.tick(0);
  await new Promise((r) => setImmediate(r));
  const firstTask = h.scheduler.busy;
  const hides = h.events.filter((e) => e === "hide").length;
  for (view = 1; view <= 20; view++) {
    h.scheduler.invalidate();
    await h.scheduler.run();
  }
  assert.deepEqual(captured, [0]);
  h.gate.resolve(h.result);
  await firstTask;
  assert.equal(h.events.filter((e) => e === "show").length, 1);
  assert.equal(h.events.filter((e) => e === "hide").length, hides);
  assert.ok(!h.events.includes("release-result"));
  t.mock.timers.tick(0);
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(captured, [0, 21]);
  h.scheduler.setEnabled(false);
  const pending = h.scheduler.busy;
  second.resolve(h.result);
  await pending;
  t.mock.timers.tick(1000);
  assert.equal(calls, 2);
  assert.equal(h.events.filter((e) => e === "show").length, 1);
  await h.scheduler.dispose();
});

test("switching to idle rejects a continuous result already in flight", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness({ mode: "continuous", idleMs: 100 });
  const { task } = await started(h);
  h.scheduler.setMode("idle");
  h.gate.resolve(h.result);
  await task;
  assert.ok(!h.events.includes("show"));
  assert.ok(h.events.includes("release-result"));
  t.mock.timers.tick(99);
  assert.equal(h.events.filter((e) => e === "capture").length, 1);
  await h.scheduler.dispose();
});

test("resizing rejects the old continuous frame before capturing a new one", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness({ mode: "continuous" });
  const { task } = await started(h);
  h.scheduler.invalidate({ reset: true });
  h.gate.resolve(h.result);
  await task;
  assert.ok(!h.events.includes("show"));
  assert.ok(h.events.includes("release-result"));
  assert.notEqual(h.scheduler.timer, null);
  await h.scheduler.dispose();
});

test("continuous inference failures stop automatic retries", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness({ mode: "continuous" });
  let calls = 0;
  h.scheduler.processor.process = async () => {
    calls++;
    throw new Error("device lost");
  };
  await h.scheduler.run();
  assert.equal(h.scheduler.failed, true);
  h.scheduler.invalidate();
  t.mock.timers.tick(1000);
  await new Promise((r) => setImmediate(r));
  assert.equal(calls, 1);
  await h.scheduler.dispose();
});
