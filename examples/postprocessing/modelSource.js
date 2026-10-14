import { validateManifest } from "./pixrestoreMath.js";

export function describeModel(source) {
  const m = source.manifest;
  const depth = m.pixelDit.depth?.keptBlocks?.length;
  const attention = m.pixelDit.attention?.operator;
  return `${source.label} · ${m.precision.toUpperCase()} · ${m.resolution}×${m.resolution}${depth ? ` · depth ${depth}` : ""}${attention ? ` · ${attention}` : ""}`;
}

export async function modelFromUrl(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok)
    throw new Error(`模型清单加载失败 (${response.status})：${url}`);
  const manifest = validateManifest(await response.json());
  return {
    label:
      new URL(response.url).pathname.split("/").slice(-2, -1)[0] || "远程模型",
    manifest,
    async loadModel(name) {
      return new URL(name, response.url).href;
    },
  };
}
