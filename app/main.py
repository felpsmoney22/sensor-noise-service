from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
from app.sensor_noise import add_sensor_noise_pil

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

    # 🔽 novos parâmetros de saída
    out_format: str = Form("PNG"),             # "PNG" ou "JPEG"
    png_compress_level: int = Form(6),         # 0..9 (0 = sem compressão, 9 = máximo)
    png_optimize: bool = Form(False),          # True/False
    jpeg_quality: int = Form(95),              # usado só se out_format="JPEG"
    jpeg_progressive: bool = Form(False),      # idem
    preserve_exif: bool = Form(True),          # tenta manter EXIF
    preserve_icc: bool = Form(True),           # tenta manter ICC
):
    # validação básica do MIME
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Apenas JPEG/PNG são suportados")

    contents = await file.read()
    try:
        pil_in = Image.open(io.BytesIO(contents))
        # Captura metadados da origem (se existirem)
        exif_bytes = pil_in.info.get("exif") if preserve_exif else None
        icc_bytes = pil_in.info.get("icc_profile") if preserve_icc else None

        # Processa (gera PIL RGB)
        out_img = add_sensor_noise_pil(
            pil_in, iso=iso, prnu_strength=prnu_strength,
            read_noise_sigma=read_noise_sigma, seed=seed
        )

        # Salva no formato escolhido, preservando EXIF/ICC quando possível
        buf = io.BytesIO()
        fmt = out_format.strip().upper()

        if fmt == "PNG":
            save_kwargs = {
                "format": "PNG",
                "compress_level": int(png_compress_level),
                "optimize": bool(png_optimize),
            }
            if icc_bytes:
                save_kwargs["icc_profile"] = icc_bytes
            if exif_bytes:
                # Pillow salva EXIF no chunk eXIf quando fornecido
                save_kwargs["exif"] = exif_bytes

            out_img.save(buf, **save_kwargs)
            media_type = "image/png"

        elif fmt == "JPEG":
            save_kwargs = {
                "format": "JPEG",
                "quality": int(jpeg_quality),
                "progressive": bool(jpeg_progressive),
                "subsampling": "4:2:0",
            }
            if icc_bytes:
                save_kwargs["icc_profile"] = icc_bytes
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes

            # JPEG exige modo "RGB"
            out_img.convert("RGB").save(buf, **save_kwargs)
            media_type = "image/jpeg"

        else:
            raise HTTPException(status_code=400, detail="Formato de saída inválido. Use PNG ou JPEG.")

        buf.seek(0)
        return StreamingResponse(buf, media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem: {e}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)
