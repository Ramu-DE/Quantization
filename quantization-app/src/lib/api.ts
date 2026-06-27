/**
 * API client for backend communication
 */
import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Types
export interface QuantizeRequest {
  weights: number[];
  bits?: number;
  clip_min?: number;
  clip_max?: number;
}

export interface QuantizeResponse {
  quantized_weights: number[];
  scale: number;
  zero_point: number;
  mse: number;
  rss: number;
  min_value: number;
  max_value: number;
  error_distribution: {
    min_error: number;
    max_error: number;
    mean_error: number;
    std_error: number;
  };
}

export interface WeightDistribution {
  histogram: {
    counts: number[];
    bin_centers: number[];
    bin_edges: number[];
  };
  cumulative: {
    values: number[];
    bins: number[];
  };
  statistics: {
    min: number;
    max: number;
    mean: number;
    std: number;
    median: number;
    range: number;
  };
}

export interface ResNetWeights {
  weights: number[];
  shape: number[];
  num_elements: number;
  dtype: string;
  statistics: {
    min: number;
    max: number;
    mean: number;
    std: number;
    median: number;
  };
}

// API Functions
export async function quantizeSymmetric(
  request: QuantizeRequest
): Promise<QuantizeResponse> {
  const response = await api.post("/api/quantize/symmetric", request);
  return response.data;
}

export async function quantizeAsymmetric(
  request: QuantizeRequest
): Promise<QuantizeResponse> {
  const response = await api.post("/api/quantize/asymmetric", request);
  return response.data;
}

export async function quantizeClipped(
  request: QuantizeRequest
): Promise<QuantizeResponse> {
  const response = await api.post("/api/quantize/clipped", request);
  return response.data;
}

export async function getWeightDistribution(
  weights: number[]
): Promise<WeightDistribution> {
  const response = await api.post("/api/weights/distribution", { weights });
  return response.data;
}

export async function loadResNet50Weights(
  sample?: number
): Promise<ResNetWeights> {
  const params = sample ? { sample } : {};
  const response = await api.get("/api/weights/resnet50", { params });
  return response.data;
}

export interface Bfloat16Response {
  original_weights: number[];
  quantized_weights: number[];
  mse: number;
  rss: number;
  error_distribution: {
    min_error: number;
    max_error: number;
    mean_error: number;
    std_error: number;
  };
}

export interface QuantizationStep {
  step: number;
  name: string;
  description: string;
  values: number[];
  reversible: boolean;
}

export interface QuantizationStepsResponse {
  steps: QuantizationStep[];
  original_weights: number[];
  final_quantized: number[];
  scale: number;
}

export interface ErrorDistributionResponse {
  bin_centers: number[];
  counts: number[];
  min_error: number;
  max_error: number;
  mean_error: number;
  std_error: number;
  num_elements: number;
}

export interface OutlierAnalysisResponse {
  negative_outliers: number[];
  positive_outliers: number[];
  total_weights: number;
  num_outliers: number;
  outlier_percentage: number;
  negative_histogram: {
    counts: number[];
    bin_centers: number[];
    bin_edges: number[];
  };
  positive_histogram: {
    counts: number[];
    bin_centers: number[];
    bin_edges: number[];
  };
}

export interface MemoryFormatInfo {
  format_name: string;
  bits_per_element: number;
  bytes: number;
  megabytes: number;
  compression_ratio: number;
}

export interface MemoryComparisonResponse {
  num_elements: number;
  formats: MemoryFormatInfo[];
}

export interface FormatInfo {
  name: string;
  total_bits: number;
  components: Record<string, number>;
  range_min: string;
  range_max: string;
  description: string;
}

export interface FormatInfoResponse {
  formats: FormatInfo[];
}

export async function quantizeBfloat16(
  request: QuantizeRequest
): Promise<Bfloat16Response> {
  const response = await api.post("/api/quantize/bfloat16", request);
  return response.data;
}

export async function getQuantizationSteps(
  request: QuantizeRequest
): Promise<QuantizationStepsResponse> {
  const response = await api.post("/api/quantize/steps", request);
  return response.data;
}

export async function getErrorDistribution(
  weights: number[],
  method: string,
  clipMin?: number,
  clipMax?: number
): Promise<ErrorDistributionResponse> {
  const body: Record<string, unknown> = { weights, method };
  if (clipMin !== undefined) body.clip_min = clipMin;
  if (clipMax !== undefined) body.clip_max = clipMax;
  const response = await api.post("/api/quantize/error-distribution", body);
  return response.data;
}

export async function getOutlierAnalysis(
  weights: number[],
  threshold: number
): Promise<OutlierAnalysisResponse> {
  const response = await api.post("/api/weights/outliers", {
    weights,
    threshold,
  });
  return response.data;
}

export async function getMemoryComparison(
  numElements: number
): Promise<MemoryComparisonResponse> {
  const response = await api.post("/api/quantize/memory", {
    num_elements: numElements,
  });
  return response.data;
}

export async function getFormatInfo(): Promise<FormatInfoResponse> {
  const response = await api.get("/api/formats");
  return response.data;
}

// FP8 simulation
export interface FP8Response {
  original: number[];
  quantized: number[];
  errors: number[];
  mse: number;
  rss: number;
  format_info: { name: string; exponent_bits: number; mantissa_bits: number; max_value: number; min_value: number };
}

export async function simulateFP8(weights: number[], format: "e4m3" | "e5m2"): Promise<FP8Response> {
  const response = await api.post("/api/fp8/convert", { weights, format });
  return response.data;
}

// GPTQ simulation
export interface GPTQStep {
  column: number;
  original_values: number[];
  quantized_values: number[];
  error: number;
  compensation_applied: boolean;
}

export interface GPTQResponse {
  steps: GPTQStep[];
  original_matrix: number[][];
  quantized_matrix: number[][];
  total_mse: number;
  compression_ratio: number;
}

export async function simulateGPTQ(weights: number[][], bits: number, groupSize: number): Promise<GPTQResponse> {
  const response = await api.post("/api/gptq/simulate", { weights, bits, group_size: groupSize });
  return response.data;
}

// SmoothQuant
export interface SmoothQuantResponse {
  original_weights: number[][];
  smoothed_weights: number[][];
  original_activations: number[][];
  smoothed_activations: number[][];
  smooth_factors: number[];
  alpha: number;
  weight_range_before: { min: number; max: number };
  weight_range_after: { min: number; max: number };
  activation_range_before: { min: number; max: number };
  activation_range_after: { min: number; max: number };
}

export async function simulateSmoothQuant(weights: number[][], activations: number[][], alpha: number): Promise<SmoothQuantResponse> {
  const response = await api.post("/api/smoothquant/simulate", { weights, activations, alpha });
  return response.data;
}

// Decision Guide
export interface DecisionGuideRequest {
  model_size_billions: number;
  hardware: "nvidia_gpu" | "amd_gpu" | "cpu" | "edge";
  latency_budget_ms: number | null;
  accuracy_tolerance: "none" | "minimal" | "moderate" | "aggressive";
  has_calibration_data: boolean;
  use_case: "inference" | "training" | "both";
}

export interface DecisionGuideResponse {
  recommended_method: string;
  recommended_bits: number;
  recommended_format: string;
  expected_compression: string;
  expected_accuracy_loss: string;
  reasoning: string[];
  alternatives: { method: string; pros: string[]; cons: string[] }[];
  warnings: string[];
}

export async function getDecisionGuide(request: DecisionGuideRequest): Promise<DecisionGuideResponse> {
  const response = await api.post("/api/decision-guide", request);
  return response.data;
}

// Benchmarks
export interface BenchmarkMethod {
  name: string;
  bits: number;
  perplexity: number;
  model_size_gb: number;
  tokens_per_sec: number;
  memory_gb: number;
}

export interface BenchmarkResponse {
  model: string;
  baseline_perplexity: number;
  methods: BenchmarkMethod[];
}

export async function getBenchmarks(): Promise<BenchmarkResponse> {
  const response = await api.get("/api/benchmarks/methods");
  return response.data;
}

export interface LiveBenchmarkResult {
  method: string;
  mse: number;
  time_ms: number;
  compression_ratio: number;
}

export interface LiveBenchmarkResponse {
  num_weights: number;
  results: LiveBenchmarkResult[];
}

export async function runLiveBenchmark(numWeights: number, methods: string[]): Promise<LiveBenchmarkResponse> {
  const response = await api.post("/api/benchmarks/live", { num_weights: numWeights, methods });
  return response.data;
}

// Hardware comparison
export interface GPUInfo {
  name: string;
  fp32_tflops: number;
  fp16_tflops: number;
  int8_tops: number;
  fp8_tops: number | null;
  memory_gb: number;
  year: number;
}

export interface HardwareResponse {
  gpus: GPUInfo[];
}

export async function getHardwareComparison(): Promise<HardwareResponse> {
  const response = await api.get("/api/hardware/comparison");
  return response.data;
}

export default api;
