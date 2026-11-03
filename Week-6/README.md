# Week 6: Docker + Cog

## Task 1: Dockerized Speech apps

Both apps use Azure Speech. Create a `.env` file in each folder (it is git-ignored):

```
SPEECH_KEY=<your Azure Speech key>
SPEECH_REGION=<your region>
```

### Speech Studio website (port 5001)

```bash
cd Task-1-speech-website
docker build -t speech-website .
docker run --rm -p 5001:5001 --env-file .env speech-website
```

Open http://localhost:5001

### Voice calculator (port 5002)

```bash
cd Task-1-voice-calculator
docker build -t voice-calculator .
docker run --rm -p 5002:5002 --env-file .env voice-calculator
```

Open http://localhost:5002

## Task 2: Cat classifier with Cog

Pre-trained ResNet50 (torchvision) served by Cog. It returns the best ImageNet class,
whether it is a cat kind (tabby, tiger cat, Persian, Siamese, Egyptian cat) and the top 5.

### 1. Build and run the Cog container

```bash
cd Task-2-cat-classifier
cog build -t cat-classifier
docker run -d --rm --name cat-classifier -p 5100:5000 cat-classifier
```

The model weights are downloaded at start-up (about 40 seconds). Wait for it:

```bash
docker logs cat-classifier 2>&1 | grep -E "now accepting|ERROR"
```

On Apple silicon the Cog base image is linux/amd64 only. If the pull fails, run
`docker pull --platform linux/amd64 r8.im/cog-base:python3.11-torch2.4.1` first.

### 2. Test the inference endpoint

`GET http://localhost:5100/health-check` returns status READY.

`POST http://localhost:5100/predictions` with a JSON body (Postman: Body, raw, JSON):

```json
{"input": {"image": "data:image/jpeg;base64,<BASE64 OF THE IMAGE>"}}
```

From the terminal:

```bash
B64=$(base64 < cat.jpg | tr -d '\n')
curl -s -X POST http://localhost:5100/predictions \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"image\": \"data:image/jpeg;base64,$B64\"}}"
```

### 3. Live prediction logs from inside the container

```bash
docker exec -it cat-classifier tail -F /tmp/predictions.log
```

### 4. Website (upload an image, press Classify)

```bash
cd website
pip install flask requests
python app.py
```

Open http://localhost:5200. The site forwards the image to the Cog container
(`COG_URL`, default `http://localhost:5100/predictions`).

### 5. Stop

```bash
docker stop cat-classifier
```