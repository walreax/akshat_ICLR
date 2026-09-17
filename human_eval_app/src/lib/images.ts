import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";

export type Model = "sd3" | "pixart" | "flux";

export const MODELS: Model[] = ["sd3", "pixart", "flux"];

export type ImageEntry = {
  id: string; // e.g. "poem_0000__sd3"
  promptId: string; // e.g. "poem_0000"
  model: Model;
  promptText: string;
  imageUrl: string; // e.g. "/images/sd3/poem_0000.png"
};

type CsvRow = {
  id: string;
  dataset_type: string;
  name_of_work: string;
  content: string;
  source: string;
};

let cachedImages: ImageEntry[] | null = null;

/**
 * Builds the master list of all ratable images (600 prompts x 3 models
 * = 1800 entries) from the shared sampled-prompts CSV. Cached in module
 * scope -- fine for a single long-lived Node process (local dev,
 * traditional server); a serverless deployment would re-parse once per
 * cold start, which is cheap (600 rows) so not worth optimizing further.
 */
export function getAllImages(): ImageEntry[] {
  if (cachedImages) return cachedImages;

  const csvPath = path.join(
    process.cwd(),
    "public",
    "images",
    "sd3_sampled_prompts.csv"
  );
  const raw = fs.readFileSync(csvPath, "utf-8");

  const rows: CsvRow[] = parse(raw, {
    columns: true,
    skip_empty_lines: true,
  });

  const entries: ImageEntry[] = [];

  for (const row of rows) {
    for (const model of MODELS) {
      entries.push({
        id: `${row.id}__${model}`,
        promptId: row.id,
        model,
        promptText: row.content,
        imageUrl: `/images/${model}/${row.id}.png`,
      });
    }
  }

  cachedImages = entries;
  return entries;
}

const imageById: Map<string, ImageEntry> = new Map();

export function getImageById(id: string): ImageEntry | undefined {
  if (imageById.size === 0) {
    for (const img of getAllImages()) imageById.set(img.id, img);
  }
  return imageById.get(id);
}
