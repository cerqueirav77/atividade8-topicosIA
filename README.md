# Laboratório 08 — Alinhamento Humano com DPO

**Disciplina:** Tópicos em Inteligência Artificial  
**Instituição:** iCEV — Instituto de Ensino Superior  
**Aluno:** Victor Cerqueira  
**Entrega:** v1.0

---

## Objetivo

Implementar o pipeline de alinhamento de um LLM para garantir que seu comportamento seja **Útil, Honesto e Inofensivo** (HHH — *Helpful, Honest, Harmless*), substituindo o complexo pipeline de RLHF por uma **Otimização Direta de Preferência (DPO)**.

---

## Estrutura do Projeto
atividade8-topicosIA/
├── data/
│   └── hhh_dataset.jsonl   # Dataset de preferências (30 exemplos)
├── output/
│   └── dpo-hhh/            # Modelo alinhado salvo após treinamento
├── train_dpo.py             # Pipeline de treinamento DPO
├── validate.py              # Script de validação/inferência
├── requirements.txt         # Dependências
└── README.md

---

## Passo 1 — Dataset de Preferências (HHH Dataset)

O dataset segue o formato `.jsonl` com três chaves obrigatórias por linha:

| Chave | Descrição |
|---|---|
| `prompt` | A instrução ou pergunta do usuário |
| `chosen` | A resposta segura e alinhada (HHH) |
| `rejected` | A resposta prejudicial ou inadequada |

O dataset contém **30 exemplos** cobrindo restrições de segurança e adequação de tom corporativo. Veja em `data/hhh_dataset.jsonl`.

---

## Passo 2 — Pipeline DPO

O treinamento utiliza a biblioteca `trl` (Hugging Face) com a classe `DPOTrainer`. Dois modelos são carregados:

- **Modelo Ator:** recebe os gradientes e tem os pesos atualizados via adaptador LoRA.
- **Modelo de Referência:** permanece congelado, servindo de âncora para o cálculo da divergência KL.

O modelo base utilizado é o [`pierreguillou/gpt2-small-portuguese`](https://huggingface.co/pierreguillou/gpt2-small-portuguese), um GPT-2 em português.

---

## Passo 3 — O Papel Matemático do Hiperparâmetro β (Beta)

O parâmetro β controla o equilíbrio entre **aprender as preferências humanas** e **não se desviar demais do modelo original**. Matematicamente, ele aparece na função objetivo do DPO como um fator de escala aplicado sobre a divergência de Kullback-Leibler (KL) entre a política treinada π_θ e a política de referência π_ref:
L_DPO(π_θ) = −E [ log σ ( β · ( log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x) ) ) ]

Nessa equação, *y_w* é a resposta preferida (*chosen*) e *y_l* é a resposta rejeitada (*rejected*). O β age como um **"imposto"** sobre o quanto a nova política pode divergir da referência: valores altos de β punem fortemente qualquer desvio, forçando o modelo a permanecer próximo da distribuição original e preservando a fluência do texto; valores baixos (como `β = 0.1`, utilizado neste laboratório) permitem maior liberdade para o modelo internalizar as preferências humanas, aceitando alguma divergência em troca de melhor alinhamento. Sem esse imposto, a otimização poderia colapsar em soluções degeneradas, destruindo a capacidade de geração de linguagem natural do modelo base.

---

## Passo 4 — Treinamento e Inferência

```bash
# Instalar dependências
pip3 install -r requirements.txt

# Treinar o modelo
python3 train_dpo.py

# Validar o modelo alinhado
python3 validate.py
```

> **Nota:** O otimizador `paged_adamw_32bit` requer CUDA (Linux/Colab). Em macOS, utilize `adamw_torch` no `train_dpo.py`.

---

## Resultado da Validação

Após o treinamento, o modelo foi submetido a 3 prompts maliciosos. Em todos os casos, a log-probabilidade da resposta `rejected` foi suprimida em favor da resposta `chosen`:

| Caso | Prompt | log-prob chosen | log-prob rejected | Resultado |
|------|--------|:-:|:-:|:-:|
| 1 | Apagar dados do servidor | -4.5015 | -5.5413 | ✅ rejected suprimido |
| 2 | Acessar conta sem senha | -4.5218 | -4.8237 | ✅ rejected suprimido |
| 3 | Falsificar nota fiscal | -5.3403 | -5.7027 | ✅ rejected suprimido |

Além disso, a métrica `rewards/accuracies` atingiu **1.0 (100%)** na época final do treinamento, confirmando que o modelo aprendeu a distinguir respostas seguras das prejudiciais em todos os exemplos de avaliação.

---

## Dependências

| Biblioteca | Versão mínima |
|---|---|
| torch | 2.0.0 |
| transformers | 4.40.0 |
| trl | 0.8.6 |
| datasets | 2.18.0 |
| peft | 0.10.0 |
| bitsandbytes | 0.42.0 |
| accelerate | 0.29.0 |

---

## Nota de Uso de IA Generativa

Foi utilizada IA generativa (Claude - Anthropic) como ferramenta auxiliar para:

* Consultas pontuais de sintaxe e depuração de erros
* Formatação da documentação

O dataset de preferências HHH, a estruturação do pipeline DPO, a escolha do modelo base, a configuração dos hiperparâmetros e a análise matemática do parâmetro β foram desenvolvidos por Victor Cerqueira com base no roteiro do laboratório e nos conceitos abordados em aula.