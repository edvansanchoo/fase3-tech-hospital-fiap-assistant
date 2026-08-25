# Hospital FIAP Assistant

Assistente clínico modular para o **FIAP POS Tech — Tech Challenge Fase 3 (IA para Devs)**. O projeto combina fine-tuning LoRA em LLaMA 2 7B, pipelines LangChain com ferramentas estruturadas e RAG, fluxo clínico automatizado com LangGraph, guardrails de segurança, logging de auditoria em JSONL e interfaces CLI, Streamlit e Jupyter notebooks.

Especificação completa: [`docs/superpowers/specs/2026-08-24-hospital-fiap-assistant-design.md`](../docs/superpowers/specs/2026-08-24-hospital-fiap-assistant-design.md)

---

## Problema

Médicos e equipes de plantão precisam de apoio rápido para consultar **protocolos internos** e **contexto do paciente**, com rastreabilidade das fontes e validação humana obrigatória em condutas clínicas. O desafio da Fase 3 exige demonstrar domínio de fine-tuning, orquestração com LangChain/LangGraph e entrega reprodutível — sem dados reais de pacientes.

**Solução:** pipeline que une LLM fine-tuned (estilo e formato de resposta), RAG sobre protocolos sintéticos (conteúdo factual), ferramentas SQL sobre prontuários fictícios, guardrails que bloqueiam prescrição direta e log append-only para auditoria.

---

## Arquitetura

```
Entrada (patient_id + pergunta)
  → LangChain: consultar prontuário (SQLite) + buscar protocolo (ChromaDB RAG)
  → LLM fine-tuned (LoRA) gera resposta contextualizada
  → Guardrails (prescrição, fontes, validação humana)
  → Log JSONL (logs/interactions.jsonl)

LangGraph (fluxo clínico):
  triagem → verificar_exames → [alerta | prontuário] → agente_protocolo
         → agente_auditoria → registrar_log → END
```

| Componente | Responsabilidade |
|------------|------------------|
| `fine_tuning/` | Preparação de dataset, treino LoRA, avaliação ROUGE-L |
| `assistant/` | LLM loader, tools (SQL + RAG), chains, CLI |
| `langgraph_flows/` | StateGraph com nós especializados e roteamento condicional |
| `security/` | Guardrails e logger de auditoria |
| `data/synthetic/` | Protocolos MD, seed SQLite (5 pacientes demo) |
| `streamlit_app/` | Demo visual para vídeo |
| `notebooks/` | Exploração, Colab, avaliação, demos |

**Papel fine-tuning vs RAG:** o fine-tune ensina estilo clínico, citação de fontes e frase de validação humana; o RAG recupera trechos dos protocolos para explicabilidade.

---

## Pré-requisitos

| Requisito | Detalhe |
|-----------|---------|
| **Python** | 3.10 ou superior |
| **GPU** | Opcional para inferência local; **recomendada** para treino LoRA (Colab T4/A100 ou CUDA local) |
| **HF_TOKEN** | Token Hugging Face com acesso ao modelo base (ex.: `meta-llama/Llama-2-7b-hf`) — necessário para download do peso base em treino/inferência real |
| **Disco** | ~8 GB para modelo 7B em 4-bit + embeddings + ChromaDB |

---

## Instalação

```bash
cd hospital-fiap-assistant
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
cp .env.example .env
# Edite .env: HF_TOKEN, paths, BASE_MODEL_ID conforme necessário
```

---

## Passo a passo (reprodução local)

### 1. Seed do banco SQLite (5 pacientes demo)

```bash
python data/synthetic/seed_db.py
```

Gera `data/synthetic/hospital.db` com PAC-001 a PAC-005.

### 2. Índice vetorial RAG (ChromaDB)

```bash
python -c "from assistant.tools import build_vector_store; build_vector_store('data/synthetic/protocols', 'artifacts/chroma_db')"
```

Indexa os protocolos em `data/synthetic/protocols/*.md`.

### 3. Preparar dataset de fine-tuning

```bash
python fine_tuning/prepare_dataset.py --pubmedqa data/synthetic/pubmedqa_sample.jsonl
```

Saída: `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`.

### 4. Fine-tuning LoRA (Colab ou GPU local)

Treino completo em **Google Colab** (notebook `notebooks/02_fine_tuning_colab.ipynb`) ou via script:

```bash
python fine_tuning/train.py                  # 3 épocas (GPU)
python fine_tuning/train.py --max-steps 2    # smoke test GPU
```

Artefatos: `artifacts/lora_adapter/`, `artifacts/training_metrics.json`.

Em **CPU**, o script valida o carregamento do JSONL e grava stub de métricas sem treinar.

### 5. Avaliação (opcional)

```bash
USE_MOCK_LLM=1 python fine_tuning/evaluate.py --max-samples 2   # CPU smoke
python fine_tuning/evaluate.py --max-samples 20                   # GPU + adapter
```

Saída: `artifacts/evaluation_report.json`.

### 6. CLI — Assistente LangChain

```bash
USE_MOCK_LLM=1 python -m assistant.cli --patient PAC-001 --query "Protocolo para febre e tosse?"
```

Com GPU e adapter treinado, omita `USE_MOCK_LLM=1`.

### 7. CLI — Fluxo LangGraph

```bash
USE_MOCK_LLM=1 python -m langgraph_flows.clinical_workflow --patient PAC-002 --query "Conduta atual?"
```

PAC-002 possui exame pendente → ramo `agente_alerta`.

### 8. Streamlit (demo para vídeo)

```bash
streamlit run streamlit_app/app.py
```

Abas: Assistente LangChain | Fluxo LangGraph | Logs (últimas 10 linhas de `logs/interactions.jsonl`).

### 9. Notebooks

| Notebook | Conteúdo |
|----------|----------|
| `01_dataset_exploration.ipynb` | Contagens e amostras do dataset processado |
| `02_fine_tuning_colab.ipynb` | Treino LoRA no Colab |
| `03_model_evaluation.ipynb` | ROUGE-L, checklist, curvas de loss |
| `04_assistant_demo.ipynb` | `run_assistant` para PAC-001 |
| `05_langgraph_flow_demo.ipynb` | `run_workflow` para os 5 pacientes |

---

## Variáveis de ambiente

Copie `.env.example` para `.env` e ajuste conforme o ambiente.

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `HF_TOKEN` | Token Hugging Face para download de modelos | *(seu token)* |
| `USE_MOCK_LLM` | `1` = respostas mock sem GPU (dev/CI) | `1` |
| `BASE_MODEL_ID` | Modelo base Hugging Face (swappable) | `meta-llama/Llama-2-7b-hf` |
| `LORA_ADAPTER_PATH` | Caminho do adapter LoRA treinado | `./artifacts/lora_adapter` |
| `EMBEDDING_MODEL` | Modelo de embeddings para RAG | `sentence-transformers/all-MiniLM-L6-v2` |
| `DATABASE_URL` | URL SQLAlchemy do SQLite sintético | `sqlite:///data/synthetic/hospital.db` |
| `LOG_PATH` | Arquivo de log de interações (append-only) | `./logs/interactions.jsonl` |
| `CHROMA_PATH` | Diretório persistente do ChromaDB | `./artifacts/chroma_db` |

---

## Troca de modelo (`BASE_MODEL_ID`)

O carregador em `assistant/llm_loader.py` lê `BASE_MODEL_ID` do ambiente. Altere no `.env` sem reescrever chains ou nós do grafo:

```env
BASE_MODEL_ID=mistralai/Mistral-7B-v0.1
# ou
BASE_MODEL_ID=meta-llama/Llama-3.2-3B
```

Após trocar o modelo base, **re-treine o adapter LoRA** com `fine_tuning/train.py` — adapters são específicos ao modelo base. Confirme que seu `HF_TOKEN` tem acesso ao repositório escolhido.

---

## Desenvolvimento em CPU (`USE_MOCK_LLM`)

Para máquinas sem GPU ou CI:

```bash
export USE_MOCK_LLM=1          # Linux/macOS
set USE_MOCK_LLM=1             # Windows CMD
$env:USE_MOCK_LLM="1"          # Windows PowerShell
```

Com `USE_MOCK_LLM=1`:

- `assistant/llm_loader.py` retorna respostas template com citação de protocolo e frase de validação humana
- Testes pytest, CLI, LangGraph e Streamlit funcionam sem download do LLaMA 7B
- `fine_tuning/evaluate.py` gera comparação mock em `artifacts/evaluation_report.json`

O Streamlit define `USE_MOCK_LLM=1` automaticamente quando não detecta CUDA.

---

## Pacientes demo

| ID | Caso | Fluxo destacado |
|----|------|-----------------|
| PAC-001 | Febre + tosse (3 dias) | RAG + sugestão de conduta |
| PAC-002 | Dor abdominal, TC pendente | Ramo de alerta LangGraph |
| PAC-003 | Follow-up oncologia mama (Fase 1) | Contexto + protocolo oncológico |
| PAC-004 | Polifarmácia (warfarina + amoxicilina) | Alerta + guardrail de prescrição |
| PAC-005 | Consulta geral | Resposta com dados limitados |

---

## Roteiro do vídeo demo (≤ 15 min)

| Tempo | Conteúdo |
|-------|----------|
| 0–2 min | Contexto hospitalar e arquitetura geral |
| 2–5 min | Dataset e fine-tuning (notebook Colab; loss; adapter salvo) |
| 5–8 min | Assistente LangChain respondendo com contexto do paciente |
| 8–11 min | Fluxo LangGraph (exame pendente → alerta → sugestão) |
| 11–14 min | Logs, fontes citadas, mensagem de validação humana |
| 14–15 min | Conclusão e limitações |

---

## Checklist de entregáveis

### Repositório Git

- [x] Pipeline de fine-tuning (prep + LoRA + avaliação)
- [x] Assistente médico LangChain (pipeline completo)
- [x] Grafo de fluxo clínico LangGraph
- [x] Dataset sintético/anônimo no repositório
- [x] README completo (instalação, execução, env vars, arquitetura)
- [x] Código Python modular
- [x] Verificação E2E (`pytest tests/ -v` + smoke CLI/LangGraph com `USE_MOCK_LLM=1`)

### Relatório técnico

- [x] Template em `reports/relatorio_tecnico.md` (seções 1–6)
- [ ] Métricas finais preenchidas após treino completo em GPU
- [ ] Link do vídeo demo

### Vídeo (≤ 15 min)

- [ ] Treinamento e comportamento do LLM customizado
- [ ] Execução automatizada do fluxo LangGraph
- [ ] Q&A clínico contextualizado
- [ ] Demonstração de logs e validação de respostas

### Interfaces

- [x] CLI: `assistant.cli`, `langgraph_flows.clinical_workflow`
- [x] Streamlit: demo visual
- [x] Notebooks: 01–05 (exploração, Colab, avaliação, demos)

---

## Testes

```bash
pytest tests/ -v
```

Testes usam `USE_MOCK_LLM=1` e não exigem GPU.

### Verificação E2E (smoke)

```bash
USE_MOCK_LLM=1 python -m assistant.cli --patient PAC-001 --query "Protocolo para febre?"
USE_MOCK_LLM=1 python -m langgraph_flows.clinical_workflow --patient PAC-002 --query "Conduta?"
```

PAC-002 aciona o ramo `agente_alerta` (exame pendente).

---

## Estrutura do repositório

```
hospital-fiap-assistant/
├── assistant/           # LLM, tools, chains, CLI
├── langgraph_flows/     # StateGraph clínico
├── fine_tuning/         # Dataset, train, evaluate
├── security/            # Guardrails + logger
├── data/synthetic/      # Protocolos + seed SQLite
├── streamlit_app/       # UI demo
├── notebooks/           # 01–05
├── reports/             # relatorio_tecnico.md
├── artifacts/           # lora_adapter, chroma_db, métricas (gitignored)
├── logs/                # interactions.jsonl (gitignored)
└── tests/
```

---

## Licença e dados

Apenas dados **sintéticos e fictícios** estão versionados. Não use dados reais de pacientes. O assistente **não substitui** decisão médica — todas as condutas exigem validação do médico responsável.
