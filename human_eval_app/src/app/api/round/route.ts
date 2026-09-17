import { randomUUID } from "crypto";
import { NextRequest, NextResponse } from "next/server";
import { getImageById } from "@/lib/images";
import { getRatedImageIdsForRater, pickImagesForRound } from "@/lib/ratings";

const ROUND_SIZE = 20;
const RATER_COOKIE = "raterId";
const ROUND_COOKIE = "roundImageIds";
const COOKIE_MAX_AGE = 60 * 60 * 24 * 365; // 1 year -- lets raters come back later

export async function GET(request: NextRequest) {
  let raterId = request.cookies.get(RATER_COOKIE)?.value;
  const isNewRater = !raterId;
  if (!raterId) raterId = randomUUID();

  const ratedByThisRater = getRatedImageIdsForRater(raterId);

  let roundIds: string[] = [];
  const roundCookieRaw = request.cookies.get(ROUND_COOKIE)?.value;
  if (roundCookieRaw) {
    try {
      roundIds = JSON.parse(roundCookieRaw);
    } catch {
      roundIds = [];
    }
  }

  const roundComplete =
    roundIds.length === 0 ||
    roundIds.every((id) => ratedByThisRater.has(id));

  if (roundComplete) {
    roundIds = pickImagesForRound(raterId, ROUND_SIZE).map((img) => img.id);
  }

  // Blind by omission: never send `model` to the client, only what's
  // needed to render the card and submit a rating.
  const images = roundIds
    .map((id) => getImageById(id))
    .filter((img): img is NonNullable<typeof img> => img !== undefined)
    .map((img) => ({
      id: img.id,
      promptText: img.promptText,
      imageUrl: img.imageUrl,
    }));

  const ratedIds = roundIds.filter((id) => ratedByThisRater.has(id));

  const response = NextResponse.json({
    raterId,
    images,
    ratedIds,
    roundSize: ROUND_SIZE,
  });

  if (isNewRater) {
    response.cookies.set(RATER_COOKIE, raterId, {
      httpOnly: true,
      sameSite: "lax",
      maxAge: COOKIE_MAX_AGE,
      path: "/",
    });
  }

  if (roundComplete) {
    response.cookies.set(ROUND_COOKIE, JSON.stringify(roundIds), {
      httpOnly: true,
      sameSite: "lax",
      maxAge: COOKIE_MAX_AGE,
      path: "/",
    });
  }

  return response;
}
