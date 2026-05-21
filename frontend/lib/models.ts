import type { ModelId } from "./api";

export const MODEL_IDS: readonly ModelId[] = [
  "classic_350",
  "himalayan",
  "meteor_350",
  "bullet_350",
];

export const MODEL_DISPLAY_NAMES: Record<ModelId, string> = {
  classic_350: "Classic 350",
  himalayan: "Himalayan",
  meteor_350: "Meteor 350",
  bullet_350: "Bullet 350",
};

// Keep in sync with backend MODEL_DESCRIPTIONS in main.py.
export const MODEL_DESCRIPTIONS: Record<ModelId, string> = {
  classic_350: "The timeless icon, reborn on the J-platform",
  himalayan: "Built for adventure, the LS410 explorer",
  meteor_350: "The modern cruiser for open highways",
  bullet_350: "The legend, refined for a new generation",
};

export function isModelId(value: string | null | undefined): value is ModelId {
  return (
    value === "classic_350" ||
    value === "himalayan" ||
    value === "meteor_350" ||
    value === "bullet_350"
  );
}

export function modelDisplayName(modelId: string | null | undefined): string {
  if (modelId && isModelId(modelId)) return MODEL_DISPLAY_NAMES[modelId];
  return modelId ?? "";
}
