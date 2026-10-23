# Hosted demo (#24)

The goal: evaluators open a URL instead of installing Ollama, and one full run on Gemini is committed as evidence. This needs a person with a Gemini API key, plus a cloud account for option B.

**Read this first.** The API has no login. Anyone who has the URL can start runs, and every run spends your Gemini quota. So:

- Use a key from a Google project that has a **daily quota cap** (AI Studio → the key's project → Quotas), or a budget alert.
- Share the URL only with the reviewers. Take it down after the review.
- The key lives only in the environment of the machine that runs the container. Never put it in the image, the repo or a chat message.

## Option A: a temporary URL from a laptop (review day, 5 minutes)

Run the cloud profile on any laptop with Docker, then open a temporary public tunnel to it. You need no cloud account. The URL lives only while the tunnel runs.

```
# PowerShell: $env:GEMINI_API_KEY="..."   (bash: export GEMINI_API_KEY=...)
docker compose --profile cloud up --build -d
curl http://localhost:8000/system          # "provider": "gemini", "model_ready": true
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` prints a `https://<random>.trycloudflare.com` address. Send that address to the reviewers. Press Ctrl+C to stop the tunnel, then run `docker compose --profile cloud down`.

## Option B: a small cloud VM (stays up for days)

One CPU VM is enough. Gemini does the model work, and the VM only runs the API, Discovery's HTTP calls and the LaTeX build. These are Google Cloud commands; the gcloud CLI is already installed on the dev laptop.

```
gcloud compute instances create researchgenie-demo --machine-type=e2-medium \
  --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=20GB --tags=rg-demo
gcloud compute firewall-rules create rg-demo-8000 --allow=tcp:8000 --target-tags=rg-demo \
  --source-ranges=<reviewer IP or campus range>/32
gcloud compute ssh researchgenie-demo
```

On the VM:

```
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin git
git clone https://github.com/Ayush-3103-AI/ResearchGenie.git && cd ResearchGenie
read -s GEMINI_API_KEY && export GEMINI_API_KEY      # paste the key; it is not echoed
sudo -E docker compose --profile cloud up --build -d
curl localhost:8000/system
```

The demo URL is `http://<VM external IP>:8000`. Keep `--source-ranges` narrow, because there is no login. Delete the VM after the review: `gcloud compute instances delete researchgenie-demo`.

## The first full Gemini run (the evidence #24 asks for)

On either option, run this once:

```
researchgenie run -q "What are the effects of intermittent fasting on cognitive performance?" --json
```

Use the CLI inside the container (`docker compose exec app-cloud researchgenie run ...`), or the web UI. Then:

1. Copy the run folder's `final/`, `metadata.json` and `events.jsonl` to `docs/research-papers/gemini-first-run/`.
2. In `TILL_NOW.md`, replace the "Gemini never observed" limitation with the measured result: words, draft_status, the QA score and the time taken.
3. Close #24 with a link to the folder.
