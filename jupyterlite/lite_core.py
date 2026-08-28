"""Hospital FIAP Assistant — versão JupyterLite (Python puro).

Compatível com https://jupyter.org/try-jupyter/lab/
Sem torch, LangChain ou LangGraph — reproduz os conceitos das aulas da Fase 3.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable

PROTOCOLS: dict[str, str] = {
    "protocolo_anemia.md": "# Protocolo Interno — Anemia\n\n## Seção 1 — Definição\nHemoglobina abaixo dos limites de referência por sexo e idade. Classificar microcítica, normocítica ou macrocítica conforme VCM.\n\n## Seção 2 — Avaliação inicial\nHistória de sangramento, dieta, medicamentos (AAS, AINEs), doença renal crônica e sintomas como fadiga, dispneia e palidez.\n\n## Seção 3 — Exames\nHemograma completo, reticulócitos, ferritina, saturação de transferrina, vitamina B12 e folato conforme padrão morfológico. Endoscopia digestiva alta e colonoscopia em anemia ferropriva sem causa aparente em adultos.\n\n## Seção 4 — Conduta\nTratar causa de base; reposição de ferro oral ou endovenoso conforme gravidade e tolerância. Transfusão reservada para anemia sintomática grave ou instabilidade hemodinâmica. Requer validação do médico responsável.\n",
    "protocolo_asma.md": "# Protocolo Interno — Asma\n\n## Seção 1 — Definição\nDoença inflamatória crônica das vias aéreas com obstrução reversível e hiperresponsividade brônquica. Confirmar história de sibilância recorrente e variabilidade dos sintomas.\n\n## Seção 2 — Gravidade\nClassificar crise leve, moderada ou grave conforme frequência respiratória, SpO2, uso de musculatura acessória e capacidade de falar frases completas.\n\n## Seção 3 — Tratamento de crise\nOxigenoterapia se SpO2 < 94%. Beta-2 agonista de curta ação inalatório repetido conforme resposta. Corticoide sistêmico em crise moderada a grave.\n\n## Seção 4 — Conduta de manutenção\nRevisar técnica inalatória e adesão ao corticoide inalatório. Plano de ação por escrito para crise. Requer validação do médico responsável.\n",
    "protocolo_avc_suspeita.md": "# Protocolo Interno — Suspeita de AVC\n\n## Seção 1 — Reconhecimento rápido\nUtilizar escala FAST ou NIHSS na triagem. Início dos sintomas com horário preciso é crítico para elegibilidade a trombólise.\n\n## Seção 2 — Sinais de alerta\nDéficit motor ou sensitivo súbito, afasia, alteração visual ou equilíbrio, cefaleia thunderclap. Ativar protocolo de emergência imediata.\n\n## Seção 3 — Exames urgentes\nTC de crânio sem contraste para excluir hemorragia. Glicemia capilar antes de trombólise. Coagulograma conforme protocolo de unidade.\n\n## Seção 4 — Conduta\nEncaminhamento imediato à unidade com capacidade de AVC; não administrar anti-hipertensivos de rotina sem orientação especializada. Requer validação do médico responsável.\n",
    "protocolo_diabetes.md": "# Protocolo Interno — Diabetes Mellitus Tipo 2\n\n## Seção 1 — Escopo\nManejo ambulatorial de diabetes tipo 2 em adultos estáveis, sem cetoacidose ou estado hiperglicêmico hiperosmolar.\n\n## Seção 2 — Avaliação\nRegistrar glicemia capilar ou HbA1c recente, tempo de diagnóstico, medicações em uso e sintomas de hipo ou hiperglicemia. Examinar pés para úlceras ou perda de sensibilidade.\n\n## Seção 3 — Metas\nMeta individualizada de HbA1c: geralmente < 7% em adultos sem comorbidades graves; flexibilizar em idosos frágeis ou alto risco de hipoglicemia.\n\n## Seção 4 — Conduta\nReforçar dieta, atividade física e adesão medicamentosa. Ajustar esquema conforme perfil glicêmico e função renal. Encaminhar endocrinologia se falha terapêutica repetida. Requer validação do médico responsável.\n",
    "protocolo_dor_cabeca.md": "# Protocolo Interno — Cefaleia\n\n## Seção 1 — Classificação inicial\nDiferenciar cefaleia primária (tensional, enxaqueca) de secundária (infecção, hipertensão intracraniana, AVC). Início súbito e intensidade máxima em segundos sugere alerta.\n\n## Seção 2 — Sinais de alerta (red flags)\nCefaleia thunderclap, déficit neurológico focal, febre com rigidez de nuca, papiledema, trauma recente ou imunossupressão — encaminhar emergência.\n\n## Seção 3 — Avaliação ambulatorial\nCaracterizar localização, duração, frequência, fatores desencadeantes e resposta a analgésicos prévios. Escala de dor e impacto funcional.\n\n## Seção 4 — Conduta\nAnalgésicos simples para cefaleia tensional episódica; triptanos considerados em enxaqueca confirmada sem contraindicações. Orientar diário de cefaleia e retorno se piora ou red flags. Requer validação do médico responsável.\n",
    "protocolo_dor_toracica.md": "# Protocolo Interno — Dor Torácica\n\n## Seção 1 — Priorização\nTratar como emergência até prova em contrário. ECG em até 10 minutos na chegada para suspeita de síndrome coronariana aguda.\n\n## Seção 2 — Diagnósticos diferenciais\nIsquemia miocárdica, embolia pulmonar, dissecção aórtica, pneumotórax, pericardite e causas esofágicas ou musculoesqueléticas.\n\n## Seção 3 — Avaliação\nHistória de irradiação, sudorese, dispneia e fatores de risco cardiovascular. Exame físico com ausculta cardíaca e pulmonar, pulsos periféricos simétricos.\n\n## Seção 4 — Conduta\nOxigenoterapia se hipoxemia; monitorização contínua. AAS e anticoagulação conforme protocolo de SCA após exclusão de contraindicações. Requer validação do médico responsável.\n",
    "protocolo_febre_v2.md": "# Protocolo Interno — Febre e Tosse v2\n\n## Seção 1 — Definição e escopo\nEste protocolo orienta a avaliação de pacientes adultos com febre e tosse de curta duração (até 7 dias), sem comorbidades graves conhecidas. Aplica-se a atendimentos ambulatoriais e pronto-socorro de baixa complexidade.\n\n## Seção 2 — Sinais de alerta\nInvestigar imediatamente: SpO2 < 94%, frequência respiratória > 24 irpm, confusão mental, hipotensão, dor torácica pleurítica ou hemoptise. Encaminhar para avaliação de emergência se presentes.\n\n## Seção 3.1 — Avaliação inicial\nPaciente adulto com febre e tosse por até 5 dias, sem comorbidades: avaliar SpO2, frequência respiratória, ausculta pulmonar e sinais de desidratação. Registrar temperatura, duração dos sintomas e contato com casos respiratórios.\n\n## Seção 3.2 — Exames complementares\nHemograma e radiografia de tórax quando suspeita de pneumonia ou persistência de sintomas além de 5 dias. Teste rápido para influenza ou COVID-19 conforme sazonalidade e epidemiologia local.\n\n## Seção 3.3 — Conduta\nConsiderar suporte clínico ambulatorial se estável: hidratação, antitérmicos e orientação de retorno em 48–72 h se piora. Evitar antibiótico de rotina em quadro viral típico. Requer validação do médico responsável.\n\n## Seção 4 — Critérios de alta\nAlta ambulatorial permitida com SpO2 ≥ 94%, ausência de sinais de gravidade e orientação por escrito sobre sinais de retorno imediato. Requer validação do médico responsável.\n",
    "protocolo_gastroenterite.md": "# Protocolo Interno — Gastroenterite Aguda\n\n## Seção 1 — Etiologia provável\nQuadro autolimitado viral na maioria dos casos. Investigar alimentos suspeitos, contatos doentes e viagens recentes.\n\n## Seção 2 — Avaliação de desidratação\nPeso, turgor cutâneo, mucosas, diurese e pressão arterial. Sinais de choque ou sangue nas fezes indicam reavaliação urgente.\n\n## Seção 3 — Exames\nGeralmente clínico. Eletrólitos e hemograma se desidratação moderada a grave, idosos ou comorbidades.\n\n## Seção 4 — Conduta\nReidratação oral preferencial; antieméticos se vômitos impedem ingesta. Evitar antibiótico de rotina. Orientar dieta progressiva e sinais de retorno. Requer validação do médico responsável.\n",
    "protocolo_hipertensao.md": "# Protocolo Interno — Hipertensão Arterial\n\n## Seção 1 — Definição\nConsidera-se hipertensão estágio 1 PA ≥ 140/90 mmHg em duas medições separadas. Avaliar em repouso, com manguito adequado e técnica padronizada.\n\n## Seção 2 — Avaliação inicial\nHistória familiar, tabagismo, diabetes, dislipidemia e lesão de órgão-alvo. Exame físico com ausculta cardíaca, fundoscopia quando disponível e pesquisa de sopros carotídeos.\n\n## Seção 3 — Exames complementares\nEletrocardiograma, creatinina, potássio, glicemia de jejum e lipidograma na primeira avaliação ou quando não realizados nos últimos 12 meses.\n\n## Seção 4 — Conduta\nModificações de estilo de vida: redução de sódio, atividade física regular, controle ponderal. Iniciar ou ajustar anti-hipertensivo conforme diretriz institucional e comorbidades. Requer validação do médico responsável.\n",
    "protocolo_insuficiencia_cardiaca.md": "# Protocolo Interno — Insuficiência Cardíaca\n\n## Seção 1 — Classificação\nInsuficiência cardíaca com fração de ejeção reduzida (ICFEr), intermediária (ICFEi) ou preservada (ICFEp). Sintomas: dispneia, edema, fadiga e intolerância ao exercício.\n\n## Seção 2 — Descompensação aguda\nPeso corporal, balanço hídrico, estertores pulmonares, turgência jugular e edema de membros inferiores. BNP ou NT-proBNP auxiliam quando diagnóstico incerto.\n\n## Seção 3 — Exames\nEcocardiograma para FEVE; eletrólitos, função renal e ECG. Radiografia de tórax para congestão pulmonar.\n\n## Seção 4 — Conduta\nRestrição de sódio e controle de ingesta hídrica em descompensados. Diurético de alça e otimização de terapia modificadora de prognóstico em ICFEr. Requer validação do médico responsável.\n",
    "protocolo_interacao_medicamentosa.md": "# Protocolo Interno — Interação Medicamentosa\n\n## Seção 1 — Objetivo\nOrientar identificação e manejo inicial de interações medicamentosas clinicamente relevantes em pacientes polimedicados, com ênfase em anticoagulantes e antibióticos.\n\n## Seção 2 — Triagem de risco\nRevisar lista completa de medicamentos ativos, incluindo automedicação e fitoterápicos. Pacientes idosos, insuficiência renal ou hepática demandam atenção redobrada.\n\n## Seção 3.1 — Warfarina e amoxicilina\nA amoxicilina pode alterar flora intestinal e reduzir produção de vitamina K, potencializando efeito anticoagulante da warfarina. Monitorar INR em 3–5 dias após início ou suspensão do antibiótico. Sinais de sangramento: equimoses extensas, hematúria, melena.\n\n## Seção 3.2 — Avaliação laboratorial\nVerificar função renal, hemograma e INR recente antes de ajustes. Repetir INR se interação confirmada ou sintomas hemorrágicos leves.\n\n## Seção 3.3 — Conduta\nConsiderar alternativa antibiótica de menor interação quando clinicamente adequado; caso mantida amoxicilina, reforçar orientação sobre sinais de sangramento e agendar controle de INR. Requer validação do médico responsável.\n\n## Seção 4 — Alerta e registro\nRegistrar alerta de interação no prontuário e comunicar prescrição concorrente. Requer validação do médico responsável.\n",
    "protocolo_ivu.md": "# Protocolo Interno — Infecção do Trato Urinário\n\n## Seção 1 — Apresentações\nCistite não complicada: disúria, polaciúria, urgência sem febre alta. Pielonefrite: febre, dor lombar, náuseas — requer avaliação mais ampla.\n\n## Seção 2 — População não complicada\nMulheres jovens sem comorbidades, gestantes e homens excluem-se do fluxo simplificado de cistite.\n\n## Seção 3 — Diagnóstico\nUrina tipo I e urocultura quando recorrência, falha terapêutica ou suspeita de pielonefrite. Teste de gravidez em mulheres em idade fértil.\n\n## Seção 4 — Conduta\nAntibiótico empírico de curta duração para cistite não complicada; ajustar conforme antibiograma. Hidratação e analgesia sintomática. Requer validação do médico responsável.\n",
    "protocolo_oncologia_mama.md": "# Protocolo Interno — Oncologia de Mama e Seguimento\n\n## Seção 1 — Contexto e vínculo Fase 1\nEste protocolo complementa o conhecimento de classificação de câncer de mama desenvolvido na Fase 1 (dataset Wisconsin — características citológicas e estratificação de risco). Utiliza-se aqui para conduzir seguimento clínico, recall de exames e priorização de imagem, não para diagnóstico automatizado.\n\n## Seção 2 — População-alvo\nPacientes em acompanhamento oncológico de mama: pós-tratamento curativo, vigilância de remissão ou investigação de achados mamográficos suspeitos (BI-RADS 3–5). Histórico de cirurgia, quimioterapia e radioterapia deve constar no prontuário.\n\n## Seção 3.1 — Priorização de recall\nClassificar recalls conforme urgência: BI-RADS 4–5 ou nódulo palpável novo → prioridade alta (retorno em até 7 dias); BI-RADS 3 → prioridade intermediária (retorno em 6 meses); achados estáveis documentados → rotina anual. Alinhar com critérios de risco da Fase 1 quando disponível score de estratificação.\n\n## Seção 3.2 — Exames de imagem\nMamografia bilateral anual em pacientes assintomáticas em seguimento. Ultrassonografia complementar para mamas densas ou achados mamográficos inconclusos. Ressonância magnética reservada para alto risco hereditário ou discordância clínico-radiológica.\n\n## Seção 3.3 — Mamografia pendente\nSe mamografia constar como pendente no prontuário, acionar fluxo de alerta antes de sugerir conduta clínica. Não emitir parecer definitivo sobre estágio ou recidiva sem resultado de imagem atualizado.\n\n## Seção 4 — Conduta clínica\nRevisar último laudo mamográfico, comparar com exames prévios e registrar plano de seguimento. Encaminhar ao serviço de imagem para agendamento prioritário quando recall de alta urgência. Requer validação do médico responsável.\n\n## Seção 5 — Documentação\nRegistrar BI-RADS, data do último exame, próximo exame programado e decisão de recall no prontuário. Requer validação do médico responsável.\n",
    "protocolo_pneumonia.md": "# Protocolo Interno — Pneumonia Adquirida na Comunidade\n\n## Seção 1 — Diagnóstico clínico\nFebre, tosse produtiva, dor pleurítica e estertores localizados. Confirmar com radiografia de tórax quando disponível.\n\n## Seção 2 — Estratificação de gravidade\nUtilizar escala CURB-65 ou PSI para decidir internação versus tratamento ambulatorial. Idade avançada, hipotensão e confusão aumentam risco.\n\n## Seção 3 — Exames\nHemograma, radiografia de tórax, gasometria se hipoxemia. Hemoculturas em internados ou casos graves.\n\n## Seção 4 — Conduta\nAntibioticoterapia empírica conforme diretriz local e fatores de risco. Suporte ventilatório e hidratação conforme necessidade. Requer validação do médico responsável.\n",
    "protocolo_trauma_leve.md": "# Protocolo Interno — Trauma Leve\n\n## Seção 1 — Definição\nLesões contusas ou lacerações superficiais sem comprometimento de via aérea, hemodinâmica instável ou déficit neurológico focal.\n\n## Seção 2 — Avaliação primária\nABCDE simplificado: via aérea pérvia, ventilação adequada, circulação estável, escala de coma de Glasgow e exposição controlada da lesão.\n\n## Seção 3 — Imagem\nRadiografias locais conforme mecanismo e dor. TC de crânio se mecanismo de alto impacto ou anticoagulação, conforme critérios institucionais.\n\n## Seção 4 — Conduta\nAnalgesia, limpeza e sutura de feridas quando indicado. Imobilização de entorses leves. Orientação de cuidados domiciliares e retorno se piora. Requer validação do médico responsável.\n"
}

DATASET_SAMPLES: dict[str, list[dict[str, Any]]] = {
    "train": [
        {
            "instruction": "Segundo protocolo_dor_cabeca, o que orienta a seção 'Avaliação ambulatorial'?",
            "input": "Protocolo: Protocolo Interno — Cefaleia",
            "output": "Segundo protocolo_dor_cabeca, Seção 3 — Avaliação ambulatorial: Caracterizar localização, duração, frequência, fatores desencadeantes e resposta a analgésicos prévios. Escala de dor e impacto funcional. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_dor_cabeca, o que orienta a seção 'Avaliação ambulatorial'?\n### Contexto:\nProtocolo: Protocolo Interno — Cefaleia\n\n### Resposta:\nSegundo protocolo_dor_cabeca, Seção 3 — Avaliação ambulatorial: Caracterizar localização, duração, frequência, fatores desencadeantes e resposta a analgésicos prévios. Escala de dor e impacto funcional. Requer validação do médico responsável."
        },
        {
            "instruction": "Segundo protocolo_pneumonia, o que orienta a seção 'Exames'?",
            "input": "Protocolo: Protocolo Interno — Pneumonia Adquirida na Comunidade",
            "output": "Segundo protocolo_pneumonia, Seção 3 — Exames: Hemograma, radiografia de tórax, gasometria se hipoxemia. Hemoculturas em internados ou casos graves. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_pneumonia, o que orienta a seção 'Exames'?\n### Contexto:\nProtocolo: Protocolo Interno — Pneumonia Adquirida na Comunidade\n\n### Resposta:\nSegundo protocolo_pneumonia, Seção 3 — Exames: Hemograma, radiografia de tórax, gasometria se hipoxemia. Hemoculturas em internados ou casos graves. Requer validação do médico responsável."
        },
        {
            "instruction": "Segundo protocolo_gastroenterite, o que orienta a seção 'Exames'?",
            "input": "Protocolo: Protocolo Interno — Gastroenterite Aguda",
            "output": "Segundo protocolo_gastroenterite, Seção 3 — Exames: Geralmente clínico. Eletrólitos e hemograma se desidratação moderada a grave, idosos ou comorbidades. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_gastroenterite, o que orienta a seção 'Exames'?\n### Contexto:\nProtocolo: Protocolo Interno — Gastroenterite Aguda\n\n### Resposta:\nSegundo protocolo_gastroenterite, Seção 3 — Exames: Geralmente clínico. Eletrólitos e hemograma se desidratação moderada a grave, idosos ou comorbidades. Requer validação do médico responsável."
        },
        {
            "instruction": "Segundo protocolo_anemia, o que orienta a seção 'Definição'?",
            "input": "Protocolo: Protocolo Interno — Anemia",
            "output": "Segundo protocolo_anemia, Seção 1 — Definição: Hemoglobina abaixo dos limites de referência por sexo e idade. Classificar microcítica, normocítica ou macrocítica conforme VCM. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_anemia, o que orienta a seção 'Definição'?\n### Contexto:\nProtocolo: Protocolo Interno — Anemia\n\n### Resposta:\nSegundo protocolo_anemia, Seção 1 — Definição: Hemoglobina abaixo dos limites de referência por sexo e idade. Classificar microcítica, normocítica ou macrocítica conforme VCM. Requer validação do médico responsável."
        },
        {
            "instruction": "Como abordar paciente com insuficiencia cardiaca?",
            "input": "Paciente dispneia paroxistica noturna.",
            "output": "Segundo evidencia clinica sintetica: otimizar diuretico e reavaliar congestao. Citar protocolo interno quando aplicavel. Requer validacao do medico responsavel.",
            "text": "### Instrução:\n### Pergunta:\nComo abordar paciente com insuficiencia cardiaca?\n### Contexto:\nPaciente dispneia paroxistica noturna.\n\n### Resposta:\nSegundo evidencia clinica sintetica: otimizar diuretico e reavaliar congestao. Citar protocolo interno quando aplicavel. Requer validacao do medico responsavel."
        }
    ],
    "val": [
        {
            "instruction": "Segundo protocolo_ivu, o que orienta a seção 'População não complicada'?",
            "input": "Protocolo: Protocolo Interno — Infecção do Trato Urinário",
            "output": "Segundo protocolo_ivu, Seção 2 — População não complicada: Mulheres jovens sem comorbidades, gestantes e homens excluem-se do fluxo simplificado de cistite. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_ivu, o que orienta a seção 'População não complicada'?\n### Contexto:\nProtocolo: Protocolo Interno — Infecção do Trato Urinário\n\n### Resposta:\nSegundo protocolo_ivu, Seção 2 — População não complicada: Mulheres jovens sem comorbidades, gestantes e homens excluem-se do fluxo simplificado de cistite. Requer validação do médico responsável."
        },
        {
            "instruction": "Segundo protocolo_febre_v2, o que orienta a seção 'Exames complementares'?",
            "input": "Protocolo: Protocolo Interno — Febre e Tosse v2",
            "output": "Segundo protocolo_febre_v2, Seção 3.2 — Exames complementares: Hemograma e radiografia de tórax quando suspeita de pneumonia ou persistência de sintomas além de 5 dias. Teste rápido para influenza ou COVID-19 conforme sazonalidade e epidemiologia local. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_febre_v2, o que orienta a seção 'Exames complementares'?\n### Contexto:\nProtocolo: Protocolo Interno — Febre e Tosse v2\n\n### Resposta:\nSegundo protocolo_febre_v2, Seção 3.2 — Exames complementares: Hemograma e radiografia de tórax quando suspeita de pneumonia ou persistência de sintomas além de 5 dias. Teste rápido para influenza ou COVID-19 conforme sazonalidade e epidemiologia local. Requer validação do médico responsável."
        },
        {
            "instruction": "Segundo protocolo_diabetes, o que orienta a seção 'Metas'?",
            "input": "Protocolo: Protocolo Interno — Diabetes Mellitus Tipo 2",
            "output": "Segundo protocolo_diabetes, Seção 3 — Metas: Meta individualizada de HbA1c: geralmente < 7% em adultos sem comorbidades graves; flexibilizar em idosos frágeis ou alto risco de hipoglicemia. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_diabetes, o que orienta a seção 'Metas'?\n### Contexto:\nProtocolo: Protocolo Interno — Diabetes Mellitus Tipo 2\n\n### Resposta:\nSegundo protocolo_diabetes, Seção 3 — Metas: Meta individualizada de HbA1c: geralmente < 7% em adultos sem comorbidades graves; flexibilizar em idosos frágeis ou alto risco de hipoglicemia. Requer validação do médico responsável."
        },
        {
            "instruction": "Quais sinais de alarme em cefaleia tensional?",
            "input": "Paciente dor holocraniana leve, sem sinais de alarme.",
            "output": "Segundo evidencia clinica sintetica: analgesia sintomatica e orientacao de retorno. Citar protocolo interno quando aplicavel. Requer validacao do medico responsavel.",
            "text": "### Instrução:\n### Pergunta:\nQuais sinais de alarme em cefaleia tensional?\n### Contexto:\nPaciente dor holocraniana leve, sem sinais de alarme.\n\n### Resposta:\nSegundo evidencia clinica sintetica: analgesia sintomatica e orientacao de retorno. Citar protocolo interno quando aplicavel. Requer validacao do medico responsavel."
        },
        {
            "instruction": "Quais exames considerar em interacao medicamentosa?",
            "input": "Paciente uso de AINE e anticoagulante.",
            "output": "Segundo evidencia clinica sintetica: revisar risco hemorragico e alternativas. Citar protocolo interno quando aplicavel. Requer validacao do medico responsavel.",
            "text": "### Instrução:\n### Pergunta:\nQuais exames considerar em interacao medicamentosa?\n### Contexto:\nPaciente uso de AINE e anticoagulante.\n\n### Resposta:\nSegundo evidencia clinica sintetica: revisar risco hemorragico e alternativas. Citar protocolo interno quando aplicavel. Requer validacao do medico responsavel."
        }
    ],
    "test": [
        {
            "instruction": "Segundo protocolo_dor_cabeca, o que orienta a seção 'Conduta'?",
            "input": "Protocolo: Protocolo Interno — Cefaleia",
            "output": "Segundo protocolo_dor_cabeca, Seção 4 — Conduta: Analgésicos simples para cefaleia tensional episódica; triptanos considerados em enxaqueca confirmada sem contraindicações. Orientar diário de cefaleia e retorno se piora ou red flags. Requer validação do médico responsável. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_dor_cabeca, o que orienta a seção 'Conduta'?\n### Contexto:\nProtocolo: Protocolo Interno — Cefaleia\n\n### Resposta:\nSegundo protocolo_dor_cabeca, Seção 4 — Conduta: Analgésicos simples para cefaleia tensional episódica; triptanos considerados em enxaqueca confirmada sem contraindicações. Orientar diário de cefaleia e retorno se piora ou red flags. Requer validação do médico responsável. Requer validação do médico responsável."
        },
        {
            "instruction": "Segundo protocolo_oncologia_mama, o que orienta a seção 'Documentação'?",
            "input": "Protocolo: Protocolo Interno — Oncologia de Mama e Seguimento",
            "output": "Segundo protocolo_oncologia_mama, Seção 5 — Documentação: Registrar BI-RADS, data do último exame, próximo exame programado e decisão de recall no prontuário. Requer validação do médico responsável. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_oncologia_mama, o que orienta a seção 'Documentação'?\n### Contexto:\nProtocolo: Protocolo Interno — Oncologia de Mama e Seguimento\n\n### Resposta:\nSegundo protocolo_oncologia_mama, Seção 5 — Documentação: Registrar BI-RADS, data do último exame, próximo exame programado e decisão de recall no prontuário. Requer validação do médico responsável. Requer validação do médico responsável."
        },
        {
            "instruction": "Segundo protocolo_hipertensao, o que orienta a seção 'Exames complementares'?",
            "input": "Protocolo: Protocolo Interno — Hipertensão Arterial",
            "output": "Segundo protocolo_hipertensao, Seção 3 — Exames complementares: Eletrocardiograma, creatinina, potássio, glicemia de jejum e lipidograma na primeira avaliação ou quando não realizados nos últimos 12 meses. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_hipertensao, o que orienta a seção 'Exames complementares'?\n### Contexto:\nProtocolo: Protocolo Interno — Hipertensão Arterial\n\n### Resposta:\nSegundo protocolo_hipertensao, Seção 3 — Exames complementares: Eletrocardiograma, creatinina, potássio, glicemia de jejum e lipidograma na primeira avaliação ou quando não realizados nos últimos 12 meses. Requer validação do médico responsável."
        },
        {
            "instruction": "Quando internar paciente com oncologia de mama?",
            "input": "Paciente nodulo palpavel BI-RADS 4.",
            "output": "Segundo evidencia clinica sintetica: encaminhar para core biopsy e discussao multidisciplinar. Citar protocolo interno quando aplicavel. Requer validacao do medico responsavel.",
            "text": "### Instrução:\n### Pergunta:\nQuando internar paciente com oncologia de mama?\n### Contexto:\nPaciente nodulo palpavel BI-RADS 4.\n\n### Resposta:\nSegundo evidencia clinica sintetica: encaminhar para core biopsy e discussao multidisciplinar. Citar protocolo interno quando aplicavel. Requer validacao do medico responsavel."
        },
        {
            "instruction": "Segundo protocolo_trauma_leve, o que orienta a seção 'Definição'?",
            "input": "Protocolo: Protocolo Interno — Trauma Leve",
            "output": "Segundo protocolo_trauma_leve, Seção 1 — Definição: Lesões contusas ou lacerações superficiais sem comprometimento de via aérea, hemodinâmica instável ou déficit neurológico focal. Requer validação do médico responsável.",
            "text": "### Instrução:\n### Pergunta:\nSegundo protocolo_trauma_leve, o que orienta a seção 'Definição'?\n### Contexto:\nProtocolo: Protocolo Interno — Trauma Leve\n\n### Resposta:\nSegundo protocolo_trauma_leve, Seção 1 — Definição: Lesões contusas ou lacerações superficiais sem comprometimento de via aérea, hemodinâmica instável ou déficit neurológico focal. Requer validação do médico responsável."
        }
    ]
}

PATIENTS = [
    ("PAC-001", "Adulto 45a", "febre, tosse 3 dias", "estável"),
    ("PAC-002", "Adulto 62a", "dor abdominal", "estável"),
    ("PAC-003", "Adulto 52a", "follow-up oncologia mama", "estável"),
    ("PAC-004", "Adulto 70a", "polifarmácia", "estável"),
    ("PAC-005", "Adulto 30a", "consulta geral", "estável"),
]

EXAMS = [
    ("PAC-001", "hemograma", "concluido", "2026-01-10"),
    ("PAC-002", "tomografia_abdominal", "pendente", "2026-01-20"),
    ("PAC-003", "mamografia", "pendente", "2026-01-18"),
    ("PAC-004", "função_renal", "concluido", "2026-01-12"),
    ("PAC-005", "consulta_geral", "concluido", "2026-01-15"),
]

PRESCRIPTIONS = [
    ("PAC-004", "warfarina 5mg", "ativo"),
    ("PAC-004", "amoxicilina 500mg", "ativo"),
]

SYSTEM_PROMPT = """Você é assistente de apoio clínico do Hospital FIAP.
- NÃO substitui o médico.
- NUNCA prescreva medicamentos ou doses finais.
- SEMPRE cite a fonte (protocolo, seção, registro).
- SEMPRE indique "Requer validação do médico responsável" em condutas.
- Se dados insuficientes, diga explicitamente."""

HUMAN_VALIDATION_PHRASE = "validação do médico responsável"
PRESCRIPTION_PATTERNS = [r"\bprescrev", r"\d+\s*mg", r"\d+\s*ml", r"\d+/\d+h"]

AUDIT_LOG: list[dict[str, Any]] = []


def anonymize_text(text: str) -> str:
    text = re.sub(r"\d{3}\.\d{3}\.\d{3}-\d{2}", "[CPF]", text)
    text = re.sub(r"\d{2}/\d{2}/\d{4}", "[DATA]", text)
    text = re.sub(
        r"(?:Sr\.|Sra\.|Dr\.|Dra\.)\s+[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*",
        "[NOME]",
        text,
    )
    return text


def format_instruction_sample(instruction: str, input_text: str, output: str) -> dict[str, str]:
    text = (
        f"### Instrução:\n### Pergunta:\n{instruction}\n"
        f"### Contexto:\n{input_text}\n\n### Resposta:\n{output}"
    )
    return {"instruction": instruction, "input": input_text, "output": output, "text": text}


def generate_protocol_pairs() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for filename, content in sorted(PROTOCOLS.items()):
        protocol_name = filename.replace(".md", "")
        title = content.splitlines()[0].lstrip("# ").strip() if content else protocol_name
        sections = re.split(r"^##\s+", content, flags=re.MULTILINE)
        for section in sections[1:]:
            header_end = section.find("\n")
            if header_end == -1:
                continue
            header = section[:header_end].strip()
            body = section[header_end:].strip()
            if len(body) < 20:
                continue
            instruction = f"Segundo {protocol_name}, o que orienta a seção '{header}'?"
            input_text = f"Protocolo: {title}"
            output = (
                f"Segundo {protocol_name}, {header}: {body[:280].strip()} "
                f"Requer validação do médico responsável."
            )
            records.append(format_instruction_sample(instruction, input_text, output))
    return records


def prepare_dataset_demo() -> dict[str, list[dict[str, str]]]:
    records = generate_protocol_pairs()
    total = len(records)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }


def simulate_lora_training(epochs: int = 3, steps_per_epoch: int = 10) -> dict[str, Any]:
    import random

    random.seed(42)
    loss_history: list[dict[str, float]] = []
    loss = 2.4
    step = 0
    for epoch in range(1, epochs + 1):
        for _ in range(steps_per_epoch):
            step += 1
            loss = max(0.35, loss * 0.92 + random.uniform(-0.03, 0.05))
            loss_history.append({"step": step, "epoch": epoch, "loss": round(loss, 4)})

    return {
        "base_model": "meta-llama/Llama-2-7b-hf (simulado)",
        "method": "LoRA rank=16, alpha=32, 4-bit (conceitual)",
        "epochs": epochs,
        "final_loss": loss_history[-1]["loss"],
        "loss_history": loss_history,
        "adapter_path": "artifacts/lora_adapter/ (simulado)",
        "note": "No JupyterLite não há GPU/torch. Treino real: use Colab ou máquina local.",
    }


def create_in_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE pacientes (patient_id TEXT PRIMARY KEY, perfil TEXT, queixa TEXT, status TEXT);
        CREATE TABLE exames (patient_id TEXT, tipo TEXT, status TEXT, data TEXT);
        CREATE TABLE prescricoes (patient_id TEXT, medicamento TEXT, status TEXT);
        """
    )
    cur.executemany("INSERT INTO pacientes VALUES (?,?,?,?)", PATIENTS)
    cur.executemany("INSERT INTO exames VALUES (?,?,?,?)", EXAMS)
    cur.executemany("INSERT INTO prescricoes VALUES (?,?,?)", PRESCRIPTIONS)
    conn.commit()
    return conn


_DB: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    global _DB
    if _DB is None:
        _DB = create_in_memory_db()
    return _DB


def consultar_prontuario(patient_id: str) -> str:
    conn = get_db()
    patient = conn.execute(
        "SELECT patient_id, perfil, queixa, status FROM pacientes WHERE patient_id = ?",
        (patient_id,),
    ).fetchone()
    if patient is None:
        return f"Paciente {patient_id} não encontrado."
    exams = conn.execute(
        "SELECT tipo, status, data FROM exames WHERE patient_id = ? ORDER BY data",
        (patient_id,),
    ).fetchall()
    prescriptions = conn.execute(
        "SELECT medicamento, status FROM prescricoes WHERE patient_id = ?",
        (patient_id,),
    ).fetchall()

    lines = [
        f"Prontuário — {patient['patient_id']}",
        f"Perfil: {patient['perfil']}",
        f"Queixa: {patient['queixa']}",
        f"Status: {patient['status']}",
    ]
    if exams:
        lines.append("Exames:")
        for exam in exams:
            lines.append(f"  - {exam['tipo']} ({exam['status']}, {exam['data']})")
    else:
        lines.append("Exames: nenhum registrado.")
    if prescriptions:
        lines.append("Prescrições ativas:")
        for rx in prescriptions:
            lines.append(f"  - {rx['medicamento']} ({rx['status']})")
    else:
        lines.append("Prescrições: nenhuma ativa.")
    return "\n".join(lines)


def verificar_exames_pendentes(patient_id: str) -> list[dict[str, str]]:
    conn = get_db()
    rows = conn.execute(
        """
        SELECT tipo, status, data FROM exames
        WHERE patient_id = ? AND status = 'pendente' ORDER BY data
        """,
        (patient_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def buscar_protocolo(query: str, k: int = 3) -> list[dict[str, Any]]:
    keywords = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    if not keywords:
        keywords = [query.lower()]

    scored: list[dict[str, Any]] = []
    for filename, content in PROTOCOLS.items():
        haystack = f"{filename} {content}".lower()
        matches = sum(1 for kw in keywords if kw in haystack)
        if matches == 0:
            continue
        scored.append({
            "content": content[:1200],
            "source": filename,
            "score": round(matches / len(keywords), 4),
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:k]


class MockLLM:
    def __init__(self, model_id: str = "meta-llama/Llama-2-7b-hf (mock)") -> None:
        self.model_id = model_id

    def _extract_sources(self, prompt: str) -> list[str]:
        sources: list[str] = []
        for match in re.finditer(r"([\w\-]+\.md)", prompt, flags=re.IGNORECASE):
            if match.group(1) not in sources:
                sources.append(match.group(1))
        return sources[:3] or ["protocolo_febre_v2.md"]

    def invoke(self, prompt: str) -> str:
        sources = self._extract_sources(prompt)
        refs = ", ".join(f"{s}, seção 3.1" for s in sources)
        return (
            f"Segundo {refs}: avaliar sinais de gravidade, monitorar evolução clínica "
            f"e seguir protocolo interno do Hospital FIAP. "
            f"Conduta sugerida com base nos registros e protocolos disponíveis. "
            f"Requer validação do médico responsável."
        )


def validate_response(text: str, sources: list[str]) -> dict[str, Any]:
    flags: list[str] = []
    lower = text.lower()
    for pat in PRESCRIPTION_PATTERNS:
        if re.search(pat, lower):
            flags.append("prescription")
    if not sources:
        flags.append("missing_source")
    source_markers = ("protocolo", "secção", "seção", "registro", ".md")
    if not any(m in lower for m in source_markers) and not sources:
        flags.append("missing_source")
    requires_human = any(k in lower for k in ("conduta", "recomend", "protocolo", "seguir"))
    if requires_human and HUMAN_VALIDATION_PHRASE not in lower:
        flags.append("missing_human_validation")
    sanitized = text
    if "prescription" in flags:
        sanitized = "Não posso prescrever medicamentos. Requer validação do médico responsável."
    return {
        "valid": "prescription" not in flags,
        "requires_human_validation": requires_human or "missing_human_validation" in flags,
        "flags": flags,
        "sanitized_text": sanitized,
    }


def log_interaction(entry: dict[str, Any]) -> dict[str, Any]:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    AUDIT_LOG.append(record)
    return record


def build_prompt(patient_id: str, query: str, chart: str, protocol_chunks: list[dict]) -> str:
    protocol_context = "\n\n".join(
        f"### Protocolo {i}: {c['source']} (score={c.get('score', 'n/a')})\n{c['content']}"
        for i, c in enumerate(protocol_chunks, 1)
    ) or "Nenhum protocolo relevante encontrado."
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Paciente: {patient_id}\n\n"
        f"### Prontuário:\n{chart}\n\n"
        f"### Pergunta:\n{query}\n\n"
        f"### Protocolos relevantes:\n{protocol_context}\n\n"
        "### Resposta:"
    )


def run_assistant(patient_id: str, query: str, llm: MockLLM | None = None) -> dict[str, Any]:
    chart = consultar_prontuario(patient_id)
    protocol_chunks = buscar_protocolo(query)
    sources = [c["source"] for c in protocol_chunks]
    prompt = build_prompt(patient_id, query, chart, protocol_chunks)
    model = llm or MockLLM()
    raw = model.invoke(prompt)
    validation = validate_response(raw, sources)
    response = validation["sanitized_text"]
    log_interaction({
        "patient_id": patient_id,
        "query": query,
        "response": response,
        "sources": sources,
        "flags": validation["flags"],
        "valid": validation["valid"],
        "requires_human_validation": validation["requires_human_validation"],
        "mode": "langchain_chain",
    })
    return {
        "response": response,
        "sources": sources,
        "requires_human_validation": validation["requires_human_validation"],
        "flags": validation["flags"],
        "valid": validation["valid"],
        "prompt_preview": prompt[:500] + "...",
    }


ClinicalState = dict[str, Any]


def _append_path(state: ClinicalState, node: str) -> list[str]:
    return [*state.get("graph_path", []), node]


def node_triagem(state: ClinicalState) -> ClinicalState:
    query_lower = state["query"].lower()
    if any(w in query_lower for w in ("conduta", "protocolo", "tratamento")):
        triage_type = "conduta"
    elif any(w in query_lower for w in ("exame", "pendente", "alerta")):
        triage_type = "alerta"
    else:
        triage_type = "informacao"
    return {**state, "triage_type": triage_type, "graph_path": _append_path(state, "triagem")}


def node_verificar_exames(state: ClinicalState) -> ClinicalState:
    pending = verificar_exames_pendentes(state["patient_id"])
    return {**state, "pending_exams": pending, "graph_path": _append_path(state, "verificar_exames")}


def route_after_exams(state: ClinicalState) -> str:
    return "alerta" if state.get("pending_exams") else "prontuario"


def node_agente_alerta(state: ClinicalState) -> ClinicalState:
    alerts = [
        f"ALERTA: Exame pendente — {e['tipo']} (status={e['status']}, data={e['data']})."
        for e in state.get("pending_exams", [])
    ]
    chart = consultar_prontuario(state["patient_id"])
    return {**state, "alerts": alerts, "chart": chart, "graph_path": _append_path(state, "agente_alerta")}


def node_agente_prontuario(state: ClinicalState) -> ClinicalState:
    chart = consultar_prontuario(state["patient_id"])
    return {**state, "chart": chart, "graph_path": _append_path(state, "agente_prontuario")}


def node_agente_protocolo(state: ClinicalState) -> ClinicalState:
    chart = state.get("chart") or consultar_prontuario(state["patient_id"])
    protocol_chunks = buscar_protocolo(state["query"])
    sources = [c["source"] for c in protocol_chunks]
    prompt = build_prompt(state["patient_id"], state["query"], chart, protocol_chunks)
    suggestion = MockLLM().invoke(prompt)
    return {
        **state,
        "retrieved_protocols": protocol_chunks,
        "sources": sources,
        "suggestion": suggestion,
        "graph_path": _append_path(state, "agente_protocolo"),
    }


def node_agente_auditoria(state: ClinicalState) -> ClinicalState:
    validation = validate_response(state.get("suggestion", ""), state.get("sources", []))
    return {
        **state,
        "suggestion": validation["sanitized_text"],
        "requires_human_validation": validation["requires_human_validation"],
        "audit_flags": validation["flags"],
        "graph_path": _append_path(state, "agente_auditoria"),
    }


def node_registrar_log(state: ClinicalState) -> ClinicalState:
    graph_path = _append_path(state, "registrar_log")
    log_entry = log_interaction({
        "patient_id": state["patient_id"],
        "query": state["query"],
        "response": state.get("suggestion", ""),
        "sources": state.get("sources", []),
        "alerts": state.get("alerts", []),
        "requires_human_validation": state.get("requires_human_validation", False),
        "graph_path": graph_path,
        "flags": state.get("audit_flags", []),
        "mode": "langgraph_workflow",
    })
    return {**state, "log_entry": log_entry, "graph_path": graph_path}


WORKFLOW_NODES: dict[str, Callable[[ClinicalState], ClinicalState]] = {
    "triagem": node_triagem,
    "verificar_exames": node_verificar_exames,
    "agente_alerta": node_agente_alerta,
    "agente_prontuario": node_agente_prontuario,
    "agente_protocolo": node_agente_protocolo,
    "agente_auditoria": node_agente_auditoria,
    "registrar_log": node_registrar_log,
}


def run_workflow(patient_id: str, query: str) -> ClinicalState:
    state: ClinicalState = {
        "patient_id": patient_id,
        "query": query,
        "pending_exams": [],
        "alerts": [],
        "sources": [],
        "graph_path": [],
    }
    state = WORKFLOW_NODES["triagem"](state)
    state = WORKFLOW_NODES["verificar_exames"](state)
    branch = route_after_exams(state)
    state = WORKFLOW_NODES["agente_alerta" if branch == "alerta" else "agente_prontuario"](state)
    state = WORKFLOW_NODES["agente_protocolo"](state)
    state = WORKFLOW_NODES["agente_auditoria"](state)
    state = WORKFLOW_NODES["registrar_log"](state)
    return state
