# Nuevos Cambios para Test Local — 2026-03-26

## RESUMEN: 3 mejoras implementadas en `experiments/train_gpt_depthrecur.py`

Todas ya están en el código. Para el test local solo necesitas cambiar env vars.

---

## CAMBIO 1: LoRA Rank 16 (antes 8)

**Qué hace:** Duplica la capacidad de especialización por capa. Con depth recurrence (4 bloques × 3 repeticiones), cada posición ahora tiene adaptadores más expresivos.

**Impacto:** +639K params (22.4M → 23.1M). Artifact 8.48 MB (antes ~10.5 MB — BAJA porque ahora las capas del medio usan int6 y las críticas int8).

**Para activar:**
```bash
# Ya es el default nuevo, pero si quieres ser explícito:
export LORA_RANK=16
```

**Para comparar con rank 8:**
```bash
export LORA_RANK=8
```

---

## CAMBIO 2: Progressive Sequence Length

**Qué hace:** Empieza entrenando con secuencias cortas (más steps/segundo) y progresivamente sube a la longitud final. El modelo aprende gramática rápido con seq cortas, luego contexto largo.

**Schedule default:**
```
0% - 15% del training:   seq_len=512   (4× más steps/segundo)
15% - 45% del training:  seq_len=1024  (2× más steps/segundo)
45% - 100% del training: seq_len=2048  (calidad final)
```

**Para activar:**
```bash
export PROG_SEQ_ENABLED=1  # ya es default
```

**Para desactivar (baseline comparison):**
```bash
export PROG_SEQ_ENABLED=0
```

**NOTA IMPORTANTE para RTX 4070:** Como torch.compile está deshabilitado en tu setup local (SDPA fallback), no hay costo de recompilación. En H100 con torch.compile habrá 3 recompilaciones (~30s cada una) pero el ahorro de steps compensa con creces.

**NOTA sobre batch tokens:** Con seq_len=512, cada step procesa el mismo número de tokens (TRAIN_BATCH_TOKENS no cambia), pero en más secuencias más cortas. El loader maneja esto automáticamente.

---

## CAMBIO 3: Mixed-Precision Quantization

**Qué hace:** Las capas de entrada (0,1) y salida (10,11) se cuantizan a int8 (más precisión). Las capas del medio (2-9) se cuantizan a int6 (menor tamaño). Las capas críticas tienen más impacto en BPB.

**No requiere env var — es automático en el código de cuantización.**

**Esto solo afecta el artifact final, NO el entrenamiento.** Así que no verás diferencia en BPB durante training. La mejora viene en la evaluación post-cuantización.

---

## TEST SUGERIDO: A/B Comparison

### Test A: Baseline (como el test de 20K que acabas de correr)
```bash
export LORA_RANK=8
export PROG_SEQ_ENABLED=0
python experiments/train_gpt_depthrecur.py  # o como lo estés corriendo
```

### Test B: Con todos los cambios nuevos
```bash
export LORA_RANK=16
export PROG_SEQ_ENABLED=1
python experiments/train_gpt_depthrecur.py
```

### Qué reportar:
1. **BPB cada 2K steps** (como antes)
2. **Tiempo total y ms/step** — el prog_seq debería dar más steps en el mismo tiempo
3. **Steps totales alcanzados** — con prog_seq deberías llegar a más steps que 20K en los mismos 36 min
4. **Loss final** — esperamos mejor que 1.5472

### Qué buscar:
- Si prog_seq funciona, verás en el log: `prog_seq:change 512->1024 step:XXX` y luego `prog_seq:change 1024->2048 step:XXX`
- Los primeros steps (seq_len=512) deberían ser ~4× más rápidos (~27ms/step vs 108ms/step)
- El BPB puede empezar más alto con seq_len=512 (normal — contexto corto) pero debería compensar cuando sube a 2048

---

## IMPORTANTE: No toques nada más

- El CAUM regime TTT ya está integrado (se activa solo durante TTT evaluation, no durante training)
- Deep supervision ya está con `DEEP_SUP_ENABLED=1` (default)
- El bug fix de Muon gradient scaling ya está aplicado
- LoRA en K,V ya está activo automáticamente

---

## SI ALGO FALLA

1. **Out of memory con rank 16:** Baja a `LORA_RANK=12` como punto medio
2. **prog_seq rompe algo:** `PROG_SEQ_ENABLED=0` para desactivar
3. **Loss explota en los primeros steps:** Normal si seq_len=512 da loss más alto al inicio. Espera hasta que suba a 2048 para comparar
