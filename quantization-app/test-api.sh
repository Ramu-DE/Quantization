#!/bin/bash
# Test API endpoints

API_URL="http://localhost:8000"

echo "Testing API endpoints..."
echo ""

echo "1. Health check:"
curl -s "$API_URL/health" | python -m json.tool 2>/dev/null || curl -s "$API_URL/health"
echo ""
echo ""

echo "2. Root endpoint:"
curl -s "$API_URL/" | python -m json.tool 2>/dev/null || curl -s "$API_URL/"
echo ""
echo ""

echo "3. API docs available at: $API_URL/docs"
echo ""

echo "4. Testing quantization endpoint (sample data):"
cat << 'EOF' | curl -s -X POST "$API_URL/api/quantize/symmetric" \
  -H "Content-Type: application/json" \
  -d @- | python -m json.tool 2>/dev/null | head -30
{
  "weights": [0.1, -0.2, 0.3, -0.4, 0.5],
  "bits": 8
}
EOF
