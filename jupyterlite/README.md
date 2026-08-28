# Hospital FIAP Assistant — Versão JupyterLite

Demo do **Tech Challenge Fase 3** que roda no navegador via [JupyterLite](https://jupyter.org/try-jupyter/lab/), sem instalar Python localmente.

## O que esta versão faz

Reproduz os conceitos das disciplinas da Fase 3 em **Python puro** (sem `torch`, `LangChain` ou `LangGraph`):

| Seção do notebook | Aula / disciplina |
|-------------------|-------------------|
| Preparação de dataset | Fine-tuning — Preparando os dados |
| Simulação LoRA + curva de loss | Fine-tuning — Fine-tuning de LLMs |
| Busca em protocolos | Fine-tuning — RAG para documentos |
| Chain clínica (prontuário + RAG + LLM mock) | LangChain na prática |
| Fluxo com ramos condicionais | LangGraph |
| Guardrails e log de auditoria | Tech Challenge — Segurança |

## Como executar no JupyterLite

1. Abra **https://jupyter.org/try-jupyter/lab/**
2. No painel esquerdo, clique no ícone **Upload** (seta para cima)
3. Envie **dois arquivos** desta pasta:
   - `hospital_fiap_assistant_lite.ipynb`
   - `lite_core.py`
4. Abra o notebook e execute: **Run → Run All Cells**

> Os dois arquivos precisam estar na **mesma pasta** no JupyterLite.

## Limitações

| Recurso | JupyterLite | Projeto completo (local/Colab) |
|---------|-------------|--------------------------------|
| Fine-tuning LoRA real | Não (simulado) | Sim (GPU) |
| LLaMA 7B + adapter | Não | Sim |
| LangChain / LangGraph nativos | Não (reimplementado) | Sim |
| ChromaDB / embeddings | Não (RAG por keywords) | Sim |
| Streamlit | Não | Sim |
| SQLite prontuário | Sim (em memória) | Sim |
| Guardrails + logs | Sim | Sim |

Para treino real e entrega final, use o repositório principal com Colab (`notebooks/02_fine_tuning_colab.ipynb`) ou ambiente local conforme o [README principal](../README.md).

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `hospital_fiap_assistant_lite.ipynb` | Notebook principal — execute este |
| `lite_core.py` | Módulo com dados embutidos e lógica |
| `generate_lite_core.py` | Script para regenerar `lite_core.py` a partir do projeto |
| `_embedded_data.json` | Dados intermediários (gerado automaticamente) |

## Regenerar após mudanças no projeto

Se você alterar protocolos ou dataset no projeto principal:

```bash
cd hospital-fiap-assistant
python jupyterlite/generate_lite_core.py
```

Isso atualiza `lite_core.py` com os protocolos e amostras mais recentes.
