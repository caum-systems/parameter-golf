# Tests Finales Antes de H100

## CONTEXTO
- Baseline (LoRA-8, no prog_seq): **1.5472 BPB** en 20K steps, 108 ms/step
- V2 (LoRA-16 + prog_seq 512→1024→2048): **1.5601 BPB** en 20K, 93.5 ms/step
- V2 fue peor en BPB pero 13% más rápido. El deficit viene del seq=512.

## CAMBIOS HECHOS DESDE V2
1. **prog_seq default cambiado**: `1024:0.25,2048:1.0` (eliminamos seq=512)
2. **byte_weighted_loss**: KILLED (default OFF). Matemáticamente inválido — standard CE ya minimiza BPB.

## 3 TESTS A CORRER (10K steps suficiente para comparar)

### Test C: LoRA-16 sin prog_seq (aislar efecto del rank)
```bash
export LORA_RANK=16
export PROG_SEQ_ENABLED=0
# Correr 10K steps
```
**Pregunta**: ¿LoRA-16 mejora o empeora vs baseline LoRA-8?

### Test D: LoRA-8 con prog_seq suave (aislar efecto del schedule)
```bash
export LORA_RANK=8
export PROG_SEQ_ENABLED=1
export PROG_SEQ_PHASES="1024:0.25,2048:1.0"
# Correr 10K steps
```
**Pregunta**: ¿prog_seq 1024→2048 (sin el dañino 512) mejora la velocidad sin perder BPB?

### Test E: El candidato final (ambos cambios si C y D son buenos)
```bash
export LORA_RANK=16
export PROG_SEQ_ENABLED=1
export PROG_SEQ_PHASES="1024:0.25,2048:1.0"
# Correr 10K steps
```

## COMPARAR CON BASELINE A 10K STEPS
Del test anterior, baseline a 10K steps fue ~1.65 BPB. Usa eso como referencia.

## CRITERIO DE DECISIÓN
- Si Test C (LoRA-16 solo) es MEJOR que baseline → usar LoRA-16 en H100
- Si Test D (prog_seq suave) es ≤ +0.005 BPB del baseline → usar prog_seq en H100 (por el speed gain)
- Si ambos mejoran → Test E es el candidato final
- Si ambos empeoran → ir con baseline LoRA-8 en H100 (safe bet)

## NOTA SOBRE H100 vs LOCAL
En H100 el entrenamiento es TIME-LIMITED (10 minutos), no step-limited:
- prog_seq 1024→2048 da ~25% más steps en los primeros 2.5 minutos
- Con batch de 786K tokens (vs ~4K local), el efecto puede ser diferente
- LoRA-16 debería beneficiarse MÁS a escala (más datos para especializar)

## DESPUÉS DE ESTOS TESTS
Si hay ganador claro → estamos listos para H100.
Si no hay ganador → ir con baseline (LoRA-8, no prog_seq) que ya probó 1.5472.
