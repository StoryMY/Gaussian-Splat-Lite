import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { validateManifest } from "../examples/postprocessing/pixrestoreMath.js";

export async function scanModels(publicDir, base = "/") {
  const models = [];
  const errors = [];
  if (!publicDir) return { models, errors };
  async function visit(dir) {
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch (error) {
      if (error.code === "ENOENT") return;
      throw error;
    }
    for (const entry of entries) {
      const filename = path.join(dir, entry.name);
      // Do not follow directory symlinks outside public.
      if (entry.isDirectory()) await visit(filename);
      else if (entry.isFile() && entry.name === "pixrestore-manifest.json") {
        const relative = path
          .relative(publicDir, filename)
          .split(path.sep)
          .join("/");
        try {
          const m = validateManifest(
            JSON.parse(await readFile(filename, "utf8")),
          );
          models.push({
            label: path.posix.dirname(relative),
            url: `${base}${relative.split("/").map(encodeURIComponent).join("/")}`,
            precision: m.precision,
            resolution: m.resolution,
          });
        } catch {
          errors.push(relative);
        }
      }
    }
  }
  await visit(publicDir);
  models.sort((a, b) => a.url.localeCompare(b.url));
  return { models, errors };
}

export function pixrestoreModels() {
  let config;
  const filename = "pixrestore-models.json";
  return {
    name: "pixrestore-models",
    configResolved(value) {
      config = value;
    },
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const pathname = req.url?.split("?")[0];
        if (
          pathname !== `/${filename}` &&
          pathname !== `${config.base}${filename}`
        )
          return next();
        try {
          const index = await scanModels(config.publicDir, config.base);
          res.setHeader("Content-Type", "application/json; charset=utf-8");
          res.setHeader("Cache-Control", "no-store");
          res.end(JSON.stringify(index));
        } catch (error) {
          next(error);
        }
      });
    },
    async generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: filename,
        source: JSON.stringify(await scanModels(config.publicDir, config.base)),
      });
    },
  };
}
