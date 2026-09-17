import { NextRequest, NextResponse } from "next/server";
import { getImageById } from "@/lib/images";
import { appendRating, getRatedImageIdsForRater } from "@/lib/ratings";

const RATER_COOKIE = "raterId";
const ROUND_COOKIE = "roundImageIds";

function isValidScore(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= 5;
}

export async function POST(request: NextRequest) {
  const raterId = request.cookies.get(RATER_COOKIE)?.value;

  if (!raterId) {
    return NextResponse.json(
      { error: "No rater session -- fetch /api/round first." },
      { status: 400 }
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const { imageId, promptCorrectness, overallCoherence } = (body ?? {}) as {
    imageId?: unknown;
    promptCorrectness?: unknown;
    overallCoherence?: unknown;
  };

  if (typeof imageId !== "string") {
    return NextResponse.json({ error: "imageId is required." }, { status: 400 });
  }

  if (!isValidScore(promptCorrectness) || !isValidScore(overallCoherence)) {
    return NextResponse.json(
      { error: "promptCorrectness and overallCoherence must be integers 1-5." },
      { status: 400 }
    );
  }

  const image = getImageById(imageId);
  if (!image) {
    return NextResponse.json({ error: "Unknown imageId." }, { status: 404 });
  }

  appendRating({
    raterId,
    imageId: image.id,
    promptId: image.promptId,
    model: image.model,
    promptCorrectness,
    overallCoherence,
    ratedAt: new Date().toISOString(),
    userAgent: request.headers.get("user-agent") ?? "",
  });

  let roundIds: string[] = [];
  const roundCookieRaw = request.cookies.get(ROUND_COOKIE)?.value;
  if (roundCookieRaw) {
    try {
      roundIds = JSON.parse(roundCookieRaw);
    } catch {
      roundIds = [];
    }
  }

  const ratedByThisRater = getRatedImageIdsForRater(raterId);
  const done = roundIds.filter((id) => ratedByThisRater.has(id)).length;

  return NextResponse.json({
    ok: true,
    progress: { done, total: roundIds.length },
  });
}
