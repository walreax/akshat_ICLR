"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type RoundImage = {
  id: string;
  promptText: string;
  imageUrl: string;
};

type RoundResponse = {
  raterId: string;
  images: RoundImage[];
  ratedIds: string[];
  roundSize: number;
};

const CONSENT_COOKIE = "consent";

function hasConsentCookie(): boolean {
  return document.cookie.split("; ").some((c) => c.startsWith(`${CONSENT_COOKIE}=1`));
}

function setConsentCookie(): void {
  const maxAge = 60 * 60 * 24 * 365; // 1 year
  document.cookie = `${CONSENT_COOKIE}=1; path=/; max-age=${maxAge}; samesite=lax`;
}

function ConsentScreen({ onAgree }: { onAgree: () => void }) {
  return (
    <div className="flex flex-1 items-center justify-center overflow-y-auto bg-zinc-50 px-4 py-10 dark:bg-black">
      <div className="w-full max-w-xl rounded-xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-950">
        <h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
          Research study: rating AI-generated images
        </h1>
        <div className="mt-4 space-y-3 text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
          <p>
            This task is part of an academic research project (submitted for
            peer review at ICLR) studying text-to-image diffusion models. You
            will be shown short text passages paired with AI-generated images
            and asked to rate each image on two dimensions: how well it
            matches the passage, and how visually coherent it is. A round is
            20 images.
          </p>
          <p>
            <strong>What we collect:</strong> only your numeric ratings and a
            randomly generated, anonymous session identifier stored in a
            cookie on this server, so you can leave and resume later. We do
            not collect your name, email, IP address, or any other personal
            information, and no account is required.
          </p>
          <p>
            <strong>Content note:</strong> images are machine-generated from
            captioned photo descriptions and may occasionally look unusual or
            low quality, but are not expected to contain disturbing content.
            If anything you see is inappropriate, please stop and let the
            researchers know.
          </p>
          <p>
            <strong>Voluntary participation:</strong> taking part is entirely
            voluntary. You may stop at any time by closing this page, with no
            penalty. Ratings already submitted may still be used in aggregate
            for the research.
          </p>
          <p>
            Questions or concerns about this study can be directed to the
            researcher at{" "}
            <a
              href="mailto:jhaakshat33@gmail.com"
              className="underline hover:no-underline"
            >
              jhaakshat33@gmail.com
            </a>
            .
          </p>
        </div>
        <button
          type="button"
          onClick={onAgree}
          className="mt-6 w-full rounded-full bg-zinc-900 px-6 py-2.5 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
        >
          I am 18+ and agree to participate
        </button>
      </div>
    </div>
  );
}

function ScoreSelector({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number | null;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs font-medium leading-snug text-zinc-700 dark:text-zinc-300">
        {label}
      </span>
      <div className="flex gap-1.5">
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            type="button"
            onClick={() => onChange(n)}
            className={`h-8 w-8 shrink-0 rounded-full border text-xs font-medium transition-colors ${
              value === n
                ? "border-zinc-900 bg-zinc-900 text-white dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900"
                : "border-zinc-300 bg-white text-zinc-700 hover:border-zinc-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-300"
            }`}
          >
            {n}
          </button>
        ))}
      </div>
    </div>
  );
}

function PromptBox({ text }: { text: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [canScroll, setCanScroll] = useState(false);

  const checkOverflow = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 4;
    setCanScroll(el.scrollHeight > el.clientHeight + 4 && !atBottom);
  }, []);

  useEffect(() => {
    checkOverflow();
    window.addEventListener("resize", checkOverflow);
    return () => window.removeEventListener("resize", checkOverflow);
  }, [text, checkOverflow]);

  return (
    <div className="relative min-h-0 flex-1">
      <div
        ref={ref}
        onScroll={checkOverflow}
        className="h-full overflow-y-auto rounded-xl border border-zinc-200 bg-white px-4 py-3 text-sm leading-relaxed text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-300"
      >
        {text}
      </div>
      {canScroll && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-center rounded-b-xl bg-gradient-to-t from-white to-transparent pb-1 pt-6 dark:from-zinc-950">
          <span className="rounded-full bg-zinc-900/80 px-2 py-0.5 text-[10px] font-medium text-white dark:bg-zinc-100/80 dark:text-zinc-900">
            scroll for more ▾
          </span>
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [consentGiven, setConsentGiven] = useState<boolean | null>(null);
  const [images, setImages] = useState<RoundImage[] | null>(null);
  const [ratedIds, setRatedIds] = useState<Set<string>>(new Set());
  const [roundSize, setRoundSize] = useState(20);
  const [promptCorrectness, setPromptCorrectness] = useState<number | null>(null);
  const [overallCoherence, setOverallCoherence] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setConsentGiven(hasConsentCookie());
  }, []);

  const fetchRound = useCallback(async () => {
    setError(null);
    setImages(null);
    try {
      const res = await fetch("/api/round");
      if (!res.ok) throw new Error("Failed to load a round.");
      const data: RoundResponse = await res.json();
      setImages(data.images);
      setRatedIds(new Set(data.ratedIds));
      setRoundSize(data.roundSize);
    } catch {
      setError("Couldn't load images. Please refresh and try again.");
    }
  }, []);

  useEffect(() => {
    if (consentGiven) fetchRound();
  }, [consentGiven, fetchRound]);

  const remaining = images?.filter((img) => !ratedIds.has(img.id)) ?? [];
  const current = remaining[0];
  const done = (images?.length ?? 0) - remaining.length;

  async function handleSubmit() {
    if (!current || promptCorrectness === null || overallCoherence === null) return;

    setSubmitting(true);
    setError(null);

    try {
      const res = await fetch("/api/rate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageId: current.id,
          promptCorrectness,
          overallCoherence,
        }),
      });

      if (!res.ok) throw new Error("Submit failed.");

      setRatedIds((prev) => new Set(prev).add(current.id));
      setPromptCorrectness(null);
      setOverallCoherence(null);
    } catch {
      setError("Couldn't submit that rating. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  if (consentGiven === null) {
    return <div className="h-dvh w-full bg-zinc-50 dark:bg-black" />;
  }

  if (!consentGiven) {
    return (
      <ConsentScreen
        onAgree={() => {
          setConsentCookie();
          setConsentGiven(true);
        }}
      />
    );
  }

  return (
    <div className="flex h-dvh w-full flex-col overflow-hidden bg-zinc-50 dark:bg-black">
      <header className="shrink-0 border-b border-zinc-200 px-6 py-3 text-center dark:border-zinc-800">
        <h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
          Image Evaluation
        </h1>
        {current && (
          <p className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
            Image {done + 1} of {roundSize}
          </p>
        )}
      </header>

      {error && (
        <div className="mx-6 mt-3 shrink-0 rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          {error}
        </div>
      )}

      {!images && !error && (
        <p className="flex flex-1 items-center justify-center text-center text-zinc-500 dark:text-zinc-400">
          Loading…
        </p>
      )}

      {images && !current && (
        <div className="flex flex-1 flex-col items-center justify-center gap-4 px-8 text-center">
          <h2 className="text-xl font-medium text-zinc-900 dark:text-zinc-50">
            Round complete!
          </h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">
            Thanks for rating {roundSize} images. You can start another round
            whenever you like.
          </p>
          <button
            type="button"
            onClick={fetchRound}
            className="mt-2 rounded-full bg-zinc-900 px-6 py-2.5 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
          >
            Start another round
          </button>
        </div>
      )}

      {current && (
        <div className="flex min-h-0 flex-1 gap-4 overflow-x-auto overflow-y-hidden p-4">
          <div className="flex min-h-0 min-w-[320px] flex-1 items-center justify-center overflow-hidden rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={current.imageUrl}
              alt="Generated image to evaluate"
              className="max-h-full max-w-full object-contain"
            />
          </div>

          <div className="flex min-h-0 w-80 shrink-0 flex-col gap-3">
            <PromptBox text={current.promptText} />

            <div className="flex shrink-0 flex-col gap-3 rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-950">
              <ScoreSelector
                label="Prompt correctness — how well does the image match the passage? (1 = not at all, 5 = perfectly)"
                value={promptCorrectness}
                onChange={setPromptCorrectness}
              />
              <ScoreSelector
                label="Overall coherence — how coherent/well-formed is the image itself? (1 = incoherent, 5 = fully coherent)"
                value={overallCoherence}
                onChange={setOverallCoherence}
              />
            </div>

            <button
              type="button"
              disabled={
                promptCorrectness === null || overallCoherence === null || submitting
              }
              onClick={handleSubmit}
              className="shrink-0 rounded-full bg-zinc-900 px-6 py-2.5 text-sm font-medium text-white transition-opacity hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
            >
              {submitting ? "Submitting…" : "Submit & next"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
