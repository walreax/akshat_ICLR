import fs from "fs";
import path from "path";
import { getAllImages, type Model } from "./images";

export type Rating = {
  raterId: string;
  imageId: string;
  promptId: string;
  model: Model;
  promptCorrectness: number; // 1-5
  overallCoherence: number; // 1-5
  ratedAt: string; // ISO timestamp
  userAgent: string; // audit trail (bot/duplicate-session spotting) -- no IP, per consent screen
};

const DATA_DIR = path.join(process.cwd(), "data");
const RATINGS_PATH = path.join(DATA_DIR, "ratings.json");

/**
 * Local JSON-file-backed rating store, for local development before
 * swapping in Google Sheets. Read-whole-file-modify-write is fine at
 * this scale (a handful of concurrent local raters) but is NOT safe
 * under real concurrent write load -- the Sheets-backed version this
 * gets swapped for should replace this module's read/write pair, not
 * be layered on top of it.
 */
function ensureDataFile(): void {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  if (!fs.existsSync(RATINGS_PATH)) fs.writeFileSync(RATINGS_PATH, "[]");
}

export function loadRatings(): Rating[] {
  ensureDataFile();
  const raw = fs.readFileSync(RATINGS_PATH, "utf-8");
  try {
    return JSON.parse(raw) as Rating[];
  } catch {
    return [];
  }
}

export function appendRating(rating: Rating): void {
  ensureDataFile();
  const ratings = loadRatings();
  ratings.push(rating);
  fs.writeFileSync(RATINGS_PATH, JSON.stringify(ratings, null, 2));
}

export function getRatedImageIdsForRater(raterId: string): Set<string> {
  const ratings = loadRatings();
  return new Set(
    ratings.filter((r) => r.raterId === raterId).map((r) => r.imageId)
  );
}

/**
 * How many times each image has been rated across ALL raters -- used
 * to bias new-round selection toward under-covered images, so ratings
 * spread out across the full 1800-image pool instead of clustering on
 * whatever a random pick happens to favor early on.
 */
export function getGlobalRatingCounts(): Map<string, number> {
  const ratings = loadRatings();
  const counts = new Map<string, number>();
  for (const r of ratings) {
    counts.set(r.imageId, (counts.get(r.imageId) ?? 0) + 1);
  }
  return counts;
}

/**
 * Picks `count` images for a new round: excludes anything this rater
 * has already rated (ever, not just this round), then prefers the
 * least-globally-rated images (shuffled within that preference) for
 * coverage balance across the whole pool.
 */
export function pickImagesForRound(raterId: string, count: number) {
  const allImages = getAllImages();
  const alreadyRated = getRatedImageIdsForRater(raterId);
  const globalCounts = getGlobalRatingCounts();

  const eligible = allImages.filter((img) => !alreadyRated.has(img.id));

  const shuffled = [...eligible].sort(() => Math.random() - 0.5);
  shuffled.sort(
    (a, b) => (globalCounts.get(a.id) ?? 0) - (globalCounts.get(b.id) ?? 0)
  );

  return shuffled.slice(0, count);
}
