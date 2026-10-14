import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, writeFile, rename, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { scanModels, pixrestoreModels } from "../scripts/pixrestore-models.js";

function fixture(root = "renamed-model", precision = "fp16") {
  const manifest = {
    format: "pixrestore-webgpu-v1",
    precision,
    resolution: 512,
    models: { dinoEncoder: "./weights/dino.onnx", pixelDit: "pixel.onnx" },
    dino: {
      inputSize: 448,
      input: {
        name: "image",
        shape: [1, 3, 448, 448],
        preprocessing: { normalization: { mean: [0, 0, 0], std: [1, 1, 1] } },
      },
      outputs: ["feature"],
    },
    pixelDit: {
      input: { name: "pixel_input", shape: [1, 6, 512, 512] },
      timestep: { name: "timestep", value: 1 },
      output: "x_pred",
      depth: { keptBlocks: [0, 1, 2, 3, 4, 5] },
      attention: { operator: "com.microsoft.MultiHeadAttention" },
    },
    singleStepSampling: {
      usesAelq: true,
      lqConditionStrength: 1,
      dinoConditionStrength: 1,
    },
  };
  return manifest;
}

test("scanner discovers renamed folders, encodes URLs and skips invalid manifests", async (t) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "gsl-model-index-"));
  t.after(async () => {
    assert.ok(
      path.resolve(root).startsWith(path.resolve(os.tmpdir()) + path.sep),
    );
    await rm(root, { recursive: true, force: true });
  });
  await mkdir(path.join(root, "nested", "模型 a"), { recursive: true });
  const manifestPath = path.join(
    root,
    "nested",
    "模型 a",
    "pixrestore-manifest.json",
  );
  await writeFile(manifestPath, JSON.stringify(fixture()));
  await writeFile(path.join(root, "pixrestore-manifest.json"), "{");
  let index = await scanModels(root, "/viewer/");
  assert.equal(index.models.length, 1);
  assert.equal(
    index.models[0].url,
    "/viewer/nested/%E6%A8%A1%E5%9E%8B%20a/pixrestore-manifest.json",
  );
  assert.deepEqual(index.errors, ["pixrestore-manifest.json"]);
  await rename(path.join(root, "nested"), path.join(root, "renamed"));
  index = await scanModels(root);
  assert.match(index.models[0].url, /^\/renamed\//);

  const plugin = pixrestoreModels();
  plugin.configResolved({ publicDir: root, base: "/" });
  let asset;
  await plugin.generateBundle.call({
    emitFile(value) {
      asset = value;
    },
  });
  assert.equal(asset.fileName, "pixrestore-models.json");
  assert.deepEqual(JSON.parse(asset.source), index);

  let middleware;
  plugin.configureServer({
    middlewares: {
      use(fn) {
        middleware = fn;
      },
    },
  });
  const headers = {};
  let body;
  await middleware(
    { url: "/pixrestore-models.json" },
    {
      setHeader(name, value) {
        headers[name] = value;
      },
      end(value) {
        body = value;
      },
    },
    (error) => {
      throw error || new Error("unexpected next");
    },
  );
  assert.equal(headers["Cache-Control"], "no-store");
  assert.deepEqual(JSON.parse(body), index);
});

test("missing public directory yields an empty index", async () => {
  assert.deepEqual(await scanModels(false), { models: [], errors: [] });
});
