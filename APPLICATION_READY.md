# Parameter Golf — Application Templates (COPY-PASTE READY)

## FORM 1: Compute Credits (RunPod — FREE $1M pool)
**URL:** https://openai.com/index/parameter-golf/#credit-form
**USE YOUR OPENAI/CHATGPT EMAIL**

### Fields to fill:

**Name:** Andres Silva

**Email:** (your OpenAI/ChatGPT account email)

**GitHub:** Blasmerit

**Organization:** CAUM Systems LLC

**Compute Level:** 8xH100 SXM

**Justification (copy this):**
```
First pure State-Space Model (Mamba-2) submission to Parameter Golf.

OpenAI explicitly requested SSM entries in the "Requests for PRs" section.
No pure SSM has been submitted yet — only a hybrid (Hymba at 1.1828 BPB).

Our submission uses Mamba-2 SSD blocks (10 layers, 512d, ~17M params)
with the baseline's training infrastructure (Muon optimizer, int8+zlib).
Code is complete and tested — we just need GPU access to run 3 seeds.

This explores whether selective state spaces can compete with transformers
in the parameter-constrained regime — a data point the community needs.

CAUM Systems — AI agent behavioral monitoring (caum.systems)
```

---

## FORM 2: Participant Form (Optional but recommended — OpenAI recruiting)
**URL:** https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf

### Fields:

**Name:** Andres Silva

**Email:** (same as above)

**GitHub:** https://github.com/Blasmerit

**Organization:** CAUM Systems LLC

**Background/Bio:**
```
Founder of CAUM Systems — building passive behavioral monitoring for AI agents.
Published research on AI agent waste detection (99K sessions, 2.6M steps).
Motor v10.31.0 uses SBERT embeddings and LZ complexity for real-time
behavioral regime classification. 2 USPTO patents filed (CIP-1, CIP-2).

Entering Parameter Golf with the first pure SSM (Mamba-2) submission.
Compression and sequence analysis are core competencies of our product.
```

---

## AFTER CREDITS ARRIVE — Run in 3 commands:

```bash
# 1. Deploy RunPod pod (use official template):
#    https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
#    Select: 8xH100 SXM, enable SSH

# 2. SSH into pod, then:
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
bash records/track_10min_16mb/2026-03-25_SSM_Mamba2_CaumSystems/setup.sh

# 3. Run 3 seeds (~15 min each):
for SEED in 1337 42 2025; do
    SEED=$SEED bash records/track_10min_16mb/2026-03-25_SSM_Mamba2_CaumSystems/run.sh
done
```

## AFTER RESULTS — Submit PR:

```bash
# Fork openai/parameter-golf on GitHub
# Push your results branch
# Create PR adding only our records/ folder
gh pr create --title "First Pure SSM (Mamba-2) Submission — CAUM Systems" \
  --body "First pure state-space model entry. Mamba-2 SSD, 10 layers, no attention.
OpenAI requested SSM submissions in the Requests for PRs section.

val_bpb: [FILL WITH RESULT]
Architecture: Mamba-2 SSD (10L, 512d, d_state=64, expand=2)
Seeds: 1337, 42, 2025
Artifact: ~12MB (int8+zlib)

CAUM Systems — https://caum.systems"
```
