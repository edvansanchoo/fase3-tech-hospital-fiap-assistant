# Hospital FIAP Assistant

Tech Challenge project for FIAP POS Tech — Fase 3 (IA para Devs): a modular clinical assistant combining fine-tuned LLaMA 2, LangChain pipelines, LangGraph workflows, RAG over hospital protocols, guardrails, and audit logging. See docs in parent folder for full spec.

## Fine-tuning (Colab / GPU)

LoRA training is expected on **Google Colab** or a local CUDA GPU. The script loads 4-bit quantized base model weights, applies PEFT LoRA (rank 16, alpha 32), and writes the adapter to `artifacts/lora_adapter/` plus `artifacts/training_metrics.json`.

```bash
python fine_tuning/prepare_dataset.py --pubmedqa data/synthetic/pubmedqa_sample.jsonl
python fine_tuning/train.py                  # full 3-epoch run (GPU required)
python fine_tuning/train.py --max-steps 2    # short GPU smoke test
```

On **CPU-only** machines the script skips training, validates JSONL loading, and still writes a metrics stub (empty `epochs` array) so CI and local dev do not require a GPU.

## Streamlit demo

Interactive UI with LangChain assistant, LangGraph workflow, and live audit log tail. On CPU-only machines the app sets `USE_MOCK_LLM=1` automatically.

```bash
streamlit run streamlit_app/app.py
```
