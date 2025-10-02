from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
from app.sensor_noise import add_sensor_noise_pil  # <- ajuste aqui

app = FastAPI(title="Sensor Noise Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    iso: int = Form(400),
    prnu_strength: float = Form(0.002),
    read_noise_sigma: float = Form(2.0),
    seed: int | None = Form(None),
):
    # validação básica
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Apenas JPEG/PNG são suportados")

    contents = await file.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao abrir imagem: {e}")

    buf = add_sensor_noise_pil(
        pil,
        iso=iso,
        prnu_strength=prnu_strength,
        read_noise_sigma=read_noise_sigma,
        seed=seed,
    )
    return StreamingResponse(buf, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)
