# 🚀 Quick Start Guide

## ⚡ **Get Running in 5 Minutes**

### **Prerequisites**
- Node.js 20+
- Python 3.11+
- UV package manager

### **1. Install UV (if not installed)**

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### **2. Install Dependencies**

```bash
# Frontend
cd quantization-app
npm install

# Backend
cd backend
uv sync
```

### **3. Run the Application**

**Option A: Two Terminals (Recommended)**

Terminal 1 - Backend:
```bash
cd quantization-app/backend
uv run uvicorn main:app --reload
```

Terminal 2 - Frontend:
```bash
cd quantization-app
npm run dev
```

**Option B: Using npm scripts**

Terminal 1:
```bash
npm run backend
```

Terminal 2:
```bash
npm run dev
```

### **4. Access the Application**

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

---

## 📁 **Missing Files to Create**

Create these 4 files with the code from `IMPLEMENTATION_STATUS.md`:

1. **`src/components/dashboard/Dashboard.tsx`**
2. **`src/components/visualizers/WeightDistribution.tsx`**
3. **`src/components/visualizers/QuantizationComparison.tsx`**
4. **`src/components/visualizers/InteractivePlayground.tsx`**

All the code for these files is provided in `IMPLEMENTATION_STATUS.md` - just copy and paste!

---

## ✅ **Verification Checklist**

After starting the application:

- [ ] Frontend loads at localhost:3000
- [ ] Backend API docs at localhost:8000/docs
- [ ] "Weight Distribution" tab visible
- [ ] "Load ResNet50 Weights" button works
- [ ] Charts render after loading weights
- [ ] "Quantization Comparison" tab shows comparison table
- [ ] "Interactive Playground" sliders are responsive

---

## 🐛 **Troubleshooting**

### **Backend won't start**

**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Fix:**
```bash
cd backend
uv sync --reinstall
```

### **CORS Error in Browser**

**Error:** `Access to fetch at 'http://localhost:8000' from origin 'http://localhost:3000' has been blocked by CORS`

**Fix:** Check that `backend/main.py` has:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### **Frontend Build Errors**

**Error:** `Module not found: Can't resolve '@/components/dashboard/Dashboard'`

**Fix:** The visualizer components haven't been created yet. See "Missing Files to Create" above.

### **Weights Won't Load**

**Error:** API returns 500 error

**Fix:** Torch/TorchVision may need to download ResNet50 model first:
```bash
python -c "import torchvision.models as models; models.resnet50(weights=models.ResNet50_Weights.DEFAULT)"
```

---

## 🎨 **Features Overview**

### **Tab 1: Weight Distribution**
- Load ResNet50 FC layer weights (2M+ parameters)
- View histogram of weight values
- See cumulative distribution
- Display statistics (min, max, mean, std)

### **Tab 2: Quantization Comparison**
- Compare 3 quantization methods side-by-side
- View MSE, RSS, scale, zero-point for each
- Automatically identify best method
- Based on AMD quantization notebook

### **Tab 3: Interactive Playground**
- Adjust test values with sliders
- See real-time quantization effects
- Modify clip ranges
- Visualize error metrics

---

## 📊 **API Endpoints Reference**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/quantize/symmetric` | POST | Symmetric INT8 quantization |
| `/api/quantize/asymmetric` | POST | Asymmetric INT8 quantization |
| `/api/quantize/clipped` | POST | Clipped quantization with range |
| `/api/weights/resnet50?sample=10000` | GET | Load ResNet50 weights (optional sample) |
| `/api/weights/distribution` | POST | Compute histogram & cumulative |

Test in API docs: http://localhost:8000/docs

---

## 🔧 **Development Tips**

### **Hot Reload**
- Frontend: Auto-reloads on file changes
- Backend: Auto-reloads with `--reload` flag

### **Check Backend Status**
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

### **View Logs**
- Frontend: Check browser console (F12)
- Backend: Check terminal running uvicorn

### **Reset State**
- Refresh browser page (Ctrl+R)
- State is managed by Zustand (in-memory)

---

## 🚢 **Production Deployment**

### **Frontend (Vercel)**
```bash
vercel deploy
```

### **Backend (Docker)**
```dockerfile
FROM python:3.11-slim
RUN pip install uv
COPY backend /app
WORKDIR /app
RUN uv sync
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Environment Variables**
```bash
# Frontend (.env.local)
NEXT_PUBLIC_API_URL=https://your-backend.com

# Backend
PYTHONUNBUFFERED=1
```

---

## 📚 **Additional Resources**

- **Next.js 15 Docs:** https://nextjs.org/docs
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **UV Docs:** https://docs.astral.sh/uv
- **Shadcn/ui:** https://ui.shadcn.com
- **Recharts:** https://recharts.org

---

## 🎉 **You're All Set!**

The application is ready to run. Just create the 4 missing visualizer components (code provided in `IMPLEMENTATION_STATUS.md`) and you'll have a fully functional quantization visualizer!

**Happy quantizing! 🚀**
