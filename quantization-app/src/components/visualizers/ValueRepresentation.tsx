"use client";

import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { getFormatInfo } from "@/lib/api";
import { Loader2, Cpu } from "lucide-react";

function BitLayout({
  name,
  signBits,
  exponentBits,
  mantissaBits,
  totalBits,
}: {
  name: string;
  signBits: number;
  exponentBits: number;
  mantissaBits: number;
  totalBits: number;
}) {
  const isInteger = signBits === 0 && exponentBits === 0;

  return (
    <div className="group hover:bg-muted/50 p-3 rounded-lg transition-all duration-200 cursor-default">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-sm font-bold w-20">{name}</span>
        <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
          {totalBits} bits
        </span>
      </div>
      <div className="flex h-10 rounded-lg overflow-hidden border-2 border-transparent group-hover:border-primary/20 transition-all shadow-sm group-hover:shadow-md">
        {!isInteger ? (
          <>
            <div
              className="bg-gradient-to-b from-red-400 to-red-600 flex items-center justify-center text-white text-xs font-bold border-r border-white/20"
              style={{ flex: signBits }}
              title={`Sign: ${signBits} bit`}
            >
              S
            </div>
            <div
              className="bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center text-white text-xs font-bold border-r border-white/20"
              style={{ flex: exponentBits }}
              title={`Exponent: ${exponentBits} bits`}
            >
              E({exponentBits})
            </div>
            <div
              className="bg-gradient-to-b from-green-400 to-green-600 flex items-center justify-center text-white text-xs font-bold"
              style={{ flex: mantissaBits }}
              title={`Mantissa: ${mantissaBits} bits`}
            >
              M({mantissaBits})
            </div>
          </>
        ) : (
          <div
            className="bg-gradient-to-b from-purple-400 to-purple-600 flex items-center justify-center text-white text-xs font-bold w-full"
            title={`Integer: ${totalBits} bits`}
          >
            Integer ({totalBits} bits)
          </div>
        )}
      </div>
    </div>
  );
}

const STATIC_FORMATS = [
  { name: "Float32", bits: 32, sign_bits: 1, exponent_bits: 8, mantissa_bits: 23, min_value: -3.4028235e38, max_value: 3.4028235e38, description: "Standard 32-bit floating point. Full precision for training." },
  { name: "Float16", bits: 16, sign_bits: 1, exponent_bits: 5, mantissa_bits: 10, min_value: -65504, max_value: 65504, description: "Half precision. Good for inference, limited range." },
  { name: "BFloat16", bits: 16, sign_bits: 1, exponent_bits: 8, mantissa_bits: 7, min_value: -3.39e38, max_value: 3.39e38, description: "Brain floating point. Same range as FP32, fewer mantissa bits." },
  { name: "Int8", bits: 8, sign_bits: 0, exponent_bits: 0, mantissa_bits: 0, min_value: -128, max_value: 127, description: "8-bit integer. 4x memory savings over FP32." },
  { name: "Int4", bits: 4, sign_bits: 0, exponent_bits: 0, mantissa_bits: 0, min_value: -8, max_value: 7, description: "4-bit integer. 8x memory savings, significant precision loss." },
];

export function ValueRepresentation() {
  const { data: formatData, isLoading, refetch, isError } = useQuery({
    queryKey: ["format-info"],
    queryFn: getFormatInfo,
    enabled: false,
  });

  const formats = formatData
    ? formatData.formats.map((fmt) => ({
        name: fmt.name,
        bits: fmt.total_bits,
        sign_bits: fmt.components.sign ?? 0,
        exponent_bits: fmt.components.exponent ?? 0,
        mantissa_bits: fmt.components.mantissa ?? 0,
        min_value: parseFloat(fmt.range_min),
        max_value: parseFloat(fmt.range_max),
        description: fmt.description,
      }))
    : STATIC_FORMATS;
  const hasLoaded = !!formatData || isError;

  return (
    <div className="space-y-6">
      <Card className="border-0 shadow-lg overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-950/30 dark:to-pink-950/30 pb-4">
          <CardTitle className="text-2xl flex items-center gap-2">
            <Cpu className="h-5 w-5 text-purple-600" />
            Number Format Representations
          </CardTitle>
          <CardDescription className="text-base">
            How different numeric formats represent values &mdash; from
            full-precision FP32 to compact INT4
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-8">
          <Button
            onClick={() => refetch()}
            disabled={isLoading}
            className="shadow-md hover:shadow-lg transition-all"
          >
            {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {hasLoaded ? "Refresh from API" : "Load Format Data"}
          </Button>

          {/* Bit Layout Diagrams */}
          <div className="space-y-2">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              Bit Layouts
            </h3>
            <p className="text-sm text-muted-foreground mb-4">
              Hover over each format to highlight. Colors:{" "}
              <span className="inline-flex items-center gap-1 mx-1"><span className="w-3 h-3 rounded bg-red-500" /> Sign</span>
              <span className="inline-flex items-center gap-1 mx-1"><span className="w-3 h-3 rounded bg-blue-500" /> Exponent</span>
              <span className="inline-flex items-center gap-1 mx-1"><span className="w-3 h-3 rounded bg-green-500" /> Mantissa</span>
              <span className="inline-flex items-center gap-1 mx-1"><span className="w-3 h-3 rounded bg-purple-500" /> Integer</span>
            </p>

            <div className="bg-muted/30 rounded-xl border p-2 divide-y">
              {formats.map((fmt) => (
                <BitLayout
                  key={fmt.name}
                  name={fmt.name.split(" ")[0]}
                  signBits={fmt.sign_bits}
                  exponentBits={fmt.exponent_bits}
                  mantissaBits={fmt.mantissa_bits}
                  totalBits={fmt.bits}
                />
              ))}
            </div>
          </div>

          {/* Key Insight */}
          <div className="p-5 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/30 dark:to-purple-950/30 border border-blue-200 dark:border-blue-800 rounded-xl">
            <h4 className="font-bold text-blue-900 dark:text-blue-100 mb-2">
              Key Insight: BFloat16 vs Float16
            </h4>
            <p className="text-sm text-blue-800 dark:text-blue-200 leading-relaxed">
              BFloat16 keeps the same <strong>8-bit exponent</strong> as FP32,
              preserving dynamic range (~10<sup>38</sup>). Float16 uses only 5
              exponent bits, limiting range to ~65504. BFloat16 trades mantissa
              precision (7 vs 10 bits) for range &mdash; ideal for deep learning
              where values span many orders of magnitude.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Comparison Table */}
      <Card className="border-0 shadow-lg">
        <CardHeader>
          <CardTitle>Format Comparison Table</CardTitle>
          <CardDescription>
            Side-by-side comparison of precision, range, and memory usage
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto rounded-xl border">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-muted/50">
                  <th className="text-left py-3 px-4 font-semibold">Format</th>
                  <th className="text-center py-3 px-4 font-semibold">Bits</th>
                  <th className="text-center py-3 px-4 font-semibold text-red-600">Sign</th>
                  <th className="text-center py-3 px-4 font-semibold text-blue-600">Exp</th>
                  <th className="text-center py-3 px-4 font-semibold text-green-600">Mantissa</th>
                  <th className="text-right py-3 px-4 font-semibold">Range</th>
                  <th className="text-center py-3 px-4 font-semibold">Compression</th>
                </tr>
              </thead>
              <tbody>
                {formats.map((fmt) => (
                  <tr
                    key={fmt.name}
                    className="border-t hover:bg-primary/5 transition-colors cursor-default"
                  >
                    <td className="py-3 px-4 font-medium">{fmt.name}</td>
                    <td className="text-center py-3 px-4">
                      <span className="font-mono font-bold">{fmt.bits}</span>
                    </td>
                    <td className="text-center py-3 px-4 text-red-600 font-mono">
                      {fmt.sign_bits || "-"}
                    </td>
                    <td className="text-center py-3 px-4 text-blue-600 font-mono">
                      {fmt.exponent_bits || "-"}
                    </td>
                    <td className="text-center py-3 px-4 text-green-600 font-mono">
                      {fmt.mantissa_bits || "-"}
                    </td>
                    <td className="text-right py-3 px-4 font-mono text-xs">
                      [{fmt.min_value.toExponential(1)}, {fmt.max_value.toExponential(1)}]
                    </td>
                    <td className="text-center py-3 px-4">
                      <span className="inline-flex items-center rounded-full bg-green-100 dark:bg-green-900/30 px-2.5 py-1 text-xs font-bold text-green-800 dark:text-green-200">
                        {(32 / fmt.bits).toFixed(0)}x
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Descriptions */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3">
            {formats.map((fmt) => (
              <div
                key={fmt.name}
                className="p-3 rounded-lg border hover:border-primary/30 hover:bg-primary/5 transition-all cursor-default"
              >
                <span className="text-xs font-bold">{fmt.name}</span>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {fmt.description}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
