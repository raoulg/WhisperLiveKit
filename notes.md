example usage of output file
```
docker run -p 8000:8000 \
  -v ./data:/app/output \
  --name wlk wlk \
  --language nl \
  --output-file /app/output/transcription.txt
```
