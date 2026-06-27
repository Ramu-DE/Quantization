/**
 * Zustand store for quantization state management
 */
import { create } from "zustand";
import type { QuantizeResponse, Bfloat16Response } from "@/lib/api";

interface QuantizationState {
  // Weights
  weights: number[];
  setWeights: (weights: number[]) => void;

  // Quantization results
  symmetricResult: QuantizeResponse | null;
  asymmetricResult: QuantizeResponse | null;
  clippedResult: QuantizeResponse | null;
  bfloat16Result: Bfloat16Response | null;

  setSymmetricResult: (result: QuantizeResponse | null) => void;
  setAsymmetricResult: (result: QuantizeResponse | null) => void;
  setClippedResult: (result: QuantizeResponse | null) => void;
  setBfloat16Result: (result: Bfloat16Response | null) => void;

  // Parameters
  bits: number;
  setBits: (bits: number) => void;

  clipMin: number;
  setClipMin: (min: number) => void;

  clipMax: number;
  setClipMax: (max: number) => void;

  // Outlier thresholds
  outlierThresholdNeg: number;
  setOutlierThresholdNeg: (threshold: number) => void;

  outlierThresholdPos: number;
  setOutlierThresholdPos: (threshold: number) => void;

  // UI state
  activeTab: string;
  setActiveTab: (tab: string) => void;

  // Reset
  reset: () => void;
}

export const useQuantizationStore = create<QuantizationState>((set) => ({
  // Initial state
  weights: [],
  symmetricResult: null,
  asymmetricResult: null,
  clippedResult: null,
  bfloat16Result: null,
  bits: 8,
  clipMin: -0.15,
  clipMax: 0.15,
  outlierThresholdNeg: -0.1,
  outlierThresholdPos: 0.1,
  activeTab: "formats",

  // Actions
  setWeights: (weights) => set({ weights }),
  setSymmetricResult: (result) => set({ symmetricResult: result }),
  setAsymmetricResult: (result) => set({ asymmetricResult: result }),
  setClippedResult: (result) => set({ clippedResult: result }),
  setBfloat16Result: (result) => set({ bfloat16Result: result }),
  setBits: (bits) => set({ bits }),
  setClipMin: (clipMin) => set({ clipMin }),
  setClipMax: (clipMax) => set({ clipMax }),
  setOutlierThresholdNeg: (outlierThresholdNeg) => set({ outlierThresholdNeg }),
  setOutlierThresholdPos: (outlierThresholdPos) => set({ outlierThresholdPos }),
  setActiveTab: (activeTab) => set({ activeTab }),

  reset: () =>
    set({
      weights: [],
      symmetricResult: null,
      asymmetricResult: null,
      clippedResult: null,
      bfloat16Result: null,
      bits: 8,
      clipMin: -0.15,
      clipMax: 0.15,
      outlierThresholdNeg: -0.1,
      outlierThresholdPos: 0.1,
    }),
}));
