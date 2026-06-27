"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class QuantizeRequest(BaseModel):
    """Request model for quantization endpoints"""
    weights: List[float] = Field(..., description="Weight values to quantize")
    bits: int = Field(8, description="Bit-width for quantization", ge=2, le=8)
    clip_min: Optional[float] = Field(None, description="Minimum clip value")
    clip_max: Optional[float] = Field(None, description="Maximum clip value")


class QuantizeResponse(BaseModel):
    """Response model for quantization endpoints"""
    quantized_weights: List[int]
    scale: float
    zero_point: int
    mse: float
    rss: float
    min_value: float
    max_value: float
    error_distribution: dict


class WeightDistributionResponse(BaseModel):
    """Response model for weight distribution"""
    histogram: dict
    cumulative: dict
    statistics: dict


class ResNetWeightsResponse(BaseModel):
    """Response model for ResNet50 weights"""
    weights: List[float]
    shape: List[int]
    num_elements: int
    dtype: str
    statistics: dict


# --- BFloat16 Quantization ---


class BFloat16Response(BaseModel):
    """Response model for bfloat16 quantization"""
    original_weights: List[float]
    quantized_weights: List[float]
    mse: float
    rss: float
    error_distribution: dict


# --- Step-by-step Symmetric Quantization ---


class QuantizationStep(BaseModel):
    """A single step in the quantization process"""
    step: int
    name: str
    description: str
    values: List[float]
    reversible: bool


class StepsResponse(BaseModel):
    """Response model for step-by-step quantization"""
    steps: List[QuantizationStep]
    original_weights: List[float]
    final_quantized: List[int]
    scale: float


# --- Error Distribution ---


class ErrorDistributionRequest(BaseModel):
    """Request model for error distribution endpoint"""
    weights: List[float] = Field(..., description="Weight values")
    method: str = Field(
        "symmetric",
        description="Quantization method: symmetric, asymmetric, clipped, or bfloat16"
    )
    bits: int = Field(8, description="Bit-width for quantization", ge=2, le=8)
    clip_min: Optional[float] = Field(None, description="Minimum clip value (for clipped method)")
    clip_max: Optional[float] = Field(None, description="Maximum clip value (for clipped method)")


class ErrorDistributionResponse(BaseModel):
    """Response model for error distribution"""
    bin_centers: List[float]
    counts: List[int]
    min_error: float
    max_error: float
    mean_error: float
    std_error: float
    num_elements: int


# --- Outlier Analysis ---


class OutlierRequest(BaseModel):
    """Request model for outlier analysis"""
    weights: List[float] = Field(..., description="Weight values")
    threshold: float = Field(0.10, description="Threshold for outlier detection")


class OutlierResponse(BaseModel):
    """Response model for outlier analysis"""
    negative_outliers: List[float]
    positive_outliers: List[float]
    total_weights: int
    num_outliers: int
    outlier_percentage: float
    negative_histogram: dict
    positive_histogram: dict


# --- Memory Comparison ---


class MemoryRequest(BaseModel):
    """Request model for memory comparison"""
    num_elements: int = Field(..., description="Number of weight elements", gt=0)


class MemoryFormatInfo(BaseModel):
    """Memory info for a single format"""
    format_name: str
    bits_per_element: int
    bytes: int
    megabytes: float
    compression_ratio: float


class MemoryResponse(BaseModel):
    """Response model for memory comparison"""
    num_elements: int
    formats: List[MemoryFormatInfo]


# --- Format Info ---


class NumberFormatInfo(BaseModel):
    """Info about a number format"""
    name: str
    total_bits: int
    components: Dict[str, int]
    range_min: str
    range_max: str
    description: str


class FormatsResponse(BaseModel):
    """Response model for number format info"""
    formats: List[NumberFormatInfo]


# --- Advanced Endpoints ---


class FP8ConvertRequest(BaseModel):
    """Request model for FP8 conversion"""
    weights: List[float] = Field(..., description="Weight values to convert")
    format: str = Field(..., description="FP8 format: 'e4m3' or 'e5m2'")


class FP8FormatInfo(BaseModel):
    """Info about the FP8 format used"""
    name: str
    exponent_bits: int
    mantissa_bits: int
    max_value: float
    min_value: float


class FP8ConvertResponse(BaseModel):
    """Response model for FP8 conversion"""
    original: List[float]
    quantized: List[float]
    errors: List[float]
    mse: float
    rss: float
    format_info: FP8FormatInfo


class GPTQSimulateRequest(BaseModel):
    """Request model for GPTQ simulation"""
    weights: List[List[float]] = Field(..., description="2D weight matrix")
    bits: int = Field(4, description="Bit-width for quantization", ge=2, le=8)
    group_size: int = Field(4, description="Group size for quantization", ge=1)


class GPTQStep(BaseModel):
    """A single step in GPTQ simulation"""
    column: int
    original_values: List[float]
    quantized_values: List[float]
    error: float
    compensation_applied: bool


class GPTQSimulateResponse(BaseModel):
    """Response model for GPTQ simulation"""
    steps: List[GPTQStep]
    original_matrix: List[List[float]]
    quantized_matrix: List[List[float]]
    total_mse: float
    compression_ratio: float


class SmoothQuantRequest(BaseModel):
    """Request model for SmoothQuant simulation"""
    weights: List[List[float]] = Field(..., description="2D weight matrix")
    activations: List[List[float]] = Field(..., description="2D activation matrix")
    alpha: float = Field(0.5, description="Smoothing factor", ge=0.0, le=1.0)


class RangeInfo(BaseModel):
    """Min/max range info"""
    min: float
    max: float


class SmoothQuantResponse(BaseModel):
    """Response model for SmoothQuant simulation"""
    original_weights: List[List[float]]
    smoothed_weights: List[List[float]]
    original_activations: List[List[float]]
    smoothed_activations: List[List[float]]
    smooth_factors: List[float]
    alpha: float
    weight_range_before: RangeInfo
    weight_range_after: RangeInfo
    activation_range_before: RangeInfo
    activation_range_after: RangeInfo


class DecisionGuideRequest(BaseModel):
    """Request model for quantization decision guide"""
    model_size_billions: float = Field(..., description="Model size in billions of parameters", gt=0)
    hardware: str = Field(..., description="Target hardware: nvidia_gpu, amd_gpu, cpu, edge")
    latency_budget_ms: Optional[float] = Field(None, description="Latency budget in milliseconds")
    accuracy_tolerance: str = Field(
        "minimal",
        description="Accuracy tolerance: none, minimal, moderate, aggressive"
    )
    has_calibration_data: bool = Field(True, description="Whether calibration data is available")
    use_case: str = Field("inference", description="Use case: inference, training, both")


class AlternativeMethod(BaseModel):
    """An alternative quantization method"""
    method: str
    pros: List[str]
    cons: List[str]


class DecisionGuideResponse(BaseModel):
    """Response model for quantization decision guide"""
    recommended_method: str
    recommended_bits: int
    recommended_format: str
    expected_compression: str
    expected_accuracy_loss: str
    reasoning: List[str]
    alternatives: List[AlternativeMethod]
    warnings: List[str]


class BenchmarkMethodEntry(BaseModel):
    """A single benchmark method entry"""
    name: str
    bits: float
    perplexity: float
    model_size_gb: float
    tokens_per_sec: float
    memory_gb: float


class BenchmarkMethodsResponse(BaseModel):
    """Response model for benchmark methods"""
    model: str
    baseline_perplexity: float
    methods: List[BenchmarkMethodEntry]


class LiveBenchmarkRequest(BaseModel):
    """Request model for live benchmark"""
    num_weights: int = Field(10000, description="Number of weights to benchmark", gt=0, le=1000000)
    methods: List[str] = Field(
        default=["symmetric", "asymmetric", "bfloat16"],
        description="Methods to benchmark"
    )


class LiveBenchmarkResult(BaseModel):
    """A single live benchmark result"""
    method: str
    mse: float
    time_ms: float
    compression_ratio: float


class LiveBenchmarkResponse(BaseModel):
    """Response model for live benchmark"""
    num_weights: int
    results: List[LiveBenchmarkResult]


class GPUInfo(BaseModel):
    """Info about a GPU"""
    name: str
    fp32_tflops: float
    fp16_tflops: float
    int8_tops: float
    fp8_tops: Optional[float]
    memory_gb: int
    year: int


class HardwareComparisonResponse(BaseModel):
    """Response model for hardware comparison"""
    gpus: List[GPUInfo]
