"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  loadResNet50Weights,
  getWeightDistribution,
  getOutlierAnalysis,
} from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
} from "recharts";
import { useQuantizationStore } from "@/store/quantizationStore";
import { Loader2, Download, AlertTriangle, TrendingDown } from "lucide-react";

export function WeightDistribution() {
  const {
    setWeights,
    outlierThresholdNeg,
    outlierThresholdPos,
    setOutlierThresholdNeg,
    setOutlierThresholdPos,
  } = useQuantizationStore();
  const [sampleSize] = useState(10000);

  const {
    data: weightsData,
    isLoading: loadingWeights,
    refetch,
  } = useQuery({
    queryKey: ["resnet50-weights", sampleSize],
    queryFn: () => loadResNet50Weights(sampleSize),
    enabled: false,
  });

  const { data: distribution, isLoading: loadingDist } = useQuery({
    queryKey: ["distribution", weightsData?.weights],
    queryFn: () => getWeightDistribution(weightsData!.weights),
    enabled: !!weightsData?.weights,
  });

  const {
    data: outlierData,
    isLoading: loadingOutliers,
    refetch: refetchOutliers,
  } = useQuery({
    queryKey: ["outliers", weightsData?.weights, outlierThresholdPos],
    queryFn: () =>
      getOutlierAnalysis(weightsData!.weights, outlierThresholdPos),
    enabled: false,
  });

  const handleLoadWeights = async () => {
    const result = await refetch();
    if (result.data) {
      setWeights(result.data.weights);
    }
  };

  const handleAnalyzeOutliers = () => {
    if (weightsData?.weights) {
      refetchOutliers();
    }
  };

  const histogramData =
    distribution?.histogram.bin_centers.map((center, idx) => ({
      value: center.toFixed(4),
      frequency: distribution.histogram.counts[idx],
    })) || [];

  const cumulativeData =
    distribution?.cumulative.bins.map((bin, idx) => ({
      value: bin.toFixed(4),
      cumulative: distribution.cumulative.values[idx],
    })) || [];

  const tailHistogramData = (() => {
    if (!outlierData) return [];
    const neg =
      outlierData.negative_histogram?.bin_centers?.map((center, idx) => ({
        value: center.toFixed(4),
        count: outlierData.negative_histogram.counts[idx],
      })) || [];
    const pos =
      outlierData.positive_histogram?.bin_centers?.map((center, idx) => ({
        value: center.toFixed(4),
        count: outlierData.positive_histogram.counts[idx],
      })) || [];
    return [...neg, ...pos];
  })();

  return (
    <div className="space-y-6">
      {/* Main Distribution Card */}
      <Card className="border-0 shadow-lg overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 pb-4">
          <CardTitle className="text-2xl flex items-center gap-2">
            <Download className="h-5 w-5 text-blue-600" />
            ResNet50 FC Layer Weight Distribution
          </CardTitle>
          <CardDescription className="text-base">
            Visualize the distribution of 2M+ parameters from ResNet50&apos;s
            fully connected layer. Click to load real model weights.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <Button
            onClick={handleLoadWeights}
            disabled={loadingWeights}
            size="lg"
            className="shadow-md hover:shadow-lg transition-all"
          >
            {loadingWeights && (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            )}
            {weightsData ? "Reload Weights" : "Load ResNet50 Weights"}
          </Button>

          {/* Stats Cards */}
          {weightsData && (
            <div className="mt-6 grid grid-cols-2 md:grid-cols-5 gap-3">
              {[
                { label: "Elements", value: weightsData.num_elements.toLocaleString(), color: "blue" },
                { label: "Min", value: weightsData.statistics.min.toFixed(4), color: "red" },
                { label: "Max", value: weightsData.statistics.max.toFixed(4), color: "green" },
                { label: "Mean", value: weightsData.statistics.mean.toFixed(4), color: "purple" },
                { label: "Std Dev", value: weightsData.statistics.std.toFixed(4), color: "orange" },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="p-3 bg-muted/50 rounded-xl border border-transparent hover:border-primary/20 hover:bg-muted transition-all duration-200 cursor-default group"
                >
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground group-hover:text-primary/70 transition-colors">
                    {stat.label}
                  </p>
                  <p className="text-sm font-mono font-semibold mt-0.5">
                    {stat.value}
                  </p>
                </div>
              ))}
            </div>
          )}

          {loadingDist && (
            <div className="mt-6 flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Computing distribution...
            </div>
          )}

          {/* Charts */}
          {distribution && (
            <div className="mt-6 space-y-8">
              <div>
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-primary" />
                  Weight Histogram
                </h3>
                <div className="bg-muted/20 rounded-xl p-4 border">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={histogramData}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                      <XAxis
                        dataKey="value"
                        tick={{ fontSize: 10 }}
                        interval="preserveStartEnd"
                      />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "10px",
                          boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
                          padding: "8px 12px",
                        }}
                        labelFormatter={(l) => `Value: ${l}`}
                        formatter={(v: number) => [v.toLocaleString(), "Frequency"]}
                      />
                      <Bar
                        dataKey="frequency"
                        fill="hsl(var(--primary))"
                        radius={[2, 2, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-green-500" />
                  Cumulative Distribution
                </h3>
                <div className="bg-muted/20 rounded-xl p-4 border">
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={cumulativeData}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                      <XAxis
                        dataKey="value"
                        tick={{ fontSize: 10 }}
                        interval="preserveStartEnd"
                      />
                      <YAxis
                        tick={{ fontSize: 10 }}
                        label={{ value: "%", position: "insideTopLeft", offset: -5, fontSize: 10 }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "10px",
                          boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
                          padding: "8px 12px",
                        }}
                        formatter={(v: number) => [`${v.toFixed(1)}%`, "Cumulative"]}
                      />
                      <Line
                        type="monotone"
                        dataKey="cumulative"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Outlier / Threshold Analysis */}
      {weightsData && (
        <Card className="border-0 shadow-lg overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950/30 dark:to-orange-950/30 pb-4">
            <CardTitle className="text-xl flex items-center gap-2">
              <TrendingDown className="h-5 w-5 text-amber-600" />
              Outlier / Threshold Analysis
            </CardTitle>
            <CardDescription>
              Explore tail distribution of weights. Outliers force the scale
              factor to be large, reducing precision for the majority of values.
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6 space-y-6">
            {/* Threshold Sliders */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3 p-4 bg-muted/30 rounded-xl border">
                <Label className="text-sm font-medium">
                  Negative Threshold:{" "}
                  <span className="font-mono text-red-600">
                    {outlierThresholdNeg.toFixed(3)}
                  </span>
                </Label>
                <Slider
                  value={[outlierThresholdNeg]}
                  onValueChange={([v]) => setOutlierThresholdNeg(v)}
                  min={-0.5}
                  max={0}
                  step={0.005}
                />
                <p className="text-[10px] text-muted-foreground">
                  Weights below this value are considered negative outliers
                </p>
              </div>
              <div className="space-y-3 p-4 bg-muted/30 rounded-xl border">
                <Label className="text-sm font-medium">
                  Positive Threshold:{" "}
                  <span className="font-mono text-green-600">
                    {outlierThresholdPos.toFixed(3)}
                  </span>
                </Label>
                <Slider
                  value={[outlierThresholdPos]}
                  onValueChange={([v]) => setOutlierThresholdPos(v)}
                  min={0}
                  max={0.5}
                  step={0.005}
                />
                <p className="text-[10px] text-muted-foreground">
                  Weights above this value are considered positive outliers
                </p>
              </div>
            </div>

            <Button
              onClick={handleAnalyzeOutliers}
              disabled={loadingOutliers || !weightsData}
              className="shadow-md hover:shadow-lg transition-all"
            >
              {loadingOutliers && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              Analyze Outliers
            </Button>

            {outlierData && (
              <div className="space-y-5 animate-fade-in-up">
                {/* Outlier stats */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-4 bg-muted/50 rounded-xl border hover:shadow-md transition-shadow">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                      Total Weights
                    </p>
                    <p className="text-lg font-bold mt-1">
                      {outlierData.total_weights.toLocaleString()}
                    </p>
                  </div>
                  <div className="p-4 bg-red-50 dark:bg-red-950/30 rounded-xl border border-red-200 dark:border-red-800 hover:shadow-md transition-shadow">
                    <p className="text-[10px] uppercase tracking-wider text-red-600">
                      Outlier Count
                    </p>
                    <p className="text-lg font-bold text-red-700 dark:text-red-400 mt-1">
                      {outlierData.num_outliers.toLocaleString()}
                    </p>
                  </div>
                  <div className="p-4 bg-red-50 dark:bg-red-950/30 rounded-xl border border-red-200 dark:border-red-800 hover:shadow-md transition-shadow">
                    <p className="text-[10px] uppercase tracking-wider text-red-600">
                      Outlier %
                    </p>
                    <p className="text-lg font-bold text-red-700 dark:text-red-400 mt-1">
                      {outlierData.outlier_percentage.toFixed(2)}%
                    </p>
                  </div>
                </div>

                {/* Tail Distribution Histogram */}
                {tailHistogramData.length > 0 && (
                  <div>
                    <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-red-500" />
                      Tail Distribution (Outliers Only)
                    </h3>
                    <div className="bg-muted/20 rounded-xl p-4 border">
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={tailHistogramData}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                          <XAxis
                            dataKey="value"
                            tick={{ fontSize: 10 }}
                            interval="preserveStartEnd"
                          />
                          <YAxis tick={{ fontSize: 10 }} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "hsl(var(--card))",
                              border: "1px solid hsl(var(--border))",
                              borderRadius: "10px",
                              boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
                              padding: "8px 12px",
                            }}
                            formatter={(v: number) => [v.toLocaleString(), "Count"]}
                          />
                          <Bar
                            dataKey="count"
                            fill="#ef4444"
                            opacity={0.8}
                            radius={[2, 2, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Insight */}
                <div className="p-4 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800 rounded-xl">
                  <div className="flex gap-3">
                    <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-semibold text-amber-800 dark:text-amber-200">
                        Impact on Quantization
                      </p>
                      <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                        Outliers force the quantization scale to accommodate extreme
                        values, reducing precision for the majority of weights near
                        zero. Clipped quantization sacrifices outlier accuracy for
                        better overall precision.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
