# Eval Report — phase1-text-baseline v3

- **Run ID:** `fd50bbda0212`
- **Started:** 2026-05-19T22:24:51.998679+00:00
- **Finished:** 2026-05-19T22:38:08.049831+00:00
- **Queries:** 39

## Configuration

- `agentic`: `True`
- `agentic_max_subqueries`: `4`
- `agentic_model`: `gemma3:4b`
- `agentic_provider`: `ollama`
- `cascade`: `False`
- `cascade_threshold`: `None`
- `contextualize`: `False`
- `contextualize_model`: `None`
- `contextualize_num_ctx`: `None`
- `contextualize_provider`: `None`
- `embedding_dim`: `1024`
- `embedding_model`: `bge-m3`
- `extract_figures`: `False`
- `extract_tables`: `False`
- `generate`: `True`
- `generator_model`: `gemma3:4b`
- `generator_provider`: `ollama`
- `judge`: `True`
- `judge_model`: `gemma3:4b`
- `judge_n_samples`: `1`
- `judge_num_ctx`: `16384`
- `judge_provider`: `ollama`
- `paper_id_filter`: `True`
- `paper_ids`: `['2604.22753v1', '2604.27742v1', '2604.27883v1', '2604.28144v1', '2604.28149v1', '2604.28159v1', '2604.28169v1', '2604.28173v1', '2604.28175v1', '2604.28176v1', '2604.28177v1', '2604.28180v1', '2604.28181v1', '2604.28182v1', '2604.28186v1', '2604.28190v1', '2604.28192v1', '2604.28193v1', '2604.28196v1', '2604.28197v1']`
- `query_expansion`: `False`
- `query_expansion_mode`: `None`
- `query_expansion_model`: `None`
- `query_expansion_n`: `None`
- `refusal_score_threshold`: `None`
- `region_number_boost`: `False`
- `rerank`: `False`
- `rerank_input_size`: `None`
- `rerank_length_norm`: `False`
- `rerank_length_penalty`: `None`
- `rerank_length_threshold`: `None`
- `rerank_model`: `None`
- `retriever`: `pipeline`
- `router`: `False`
- `top_k`: `10`
- `visual_device`: `None`
- `visual_model`: `None`
- `vlm_caption_model`: `None`
- `vlm_caption_provider`: `None`

## Retrieval (in-corpus queries)

| Metric | Value |
|---|---|
| nDCG@5 (macro) | 0.0342 |
| recall@10 (macro) | 0.0968 |
| MRR (macro) | 0.0296 |
| n in-corpus queries | 31 |

## Generation

| Metric | Value |
|---|---|
| citation grounding (mean) | 1.0000 |
| faithfulness (LLM judge) (mean) | 0.7538 |
| answer relevance (LLM judge) (mean) | 0.6308 |
| context precision (LLM judge) (mean) | 0.8179 |
| answer correctness vs expected_facts (LLM judge) (mean) | 0.7613 |
| total tokens in | 145865 |
| total tokens out | 10932 |

## Latency

| Metric | Value (ms) |
|---|---|
| p50 | 20812 |
| p95 | 24125 |
| mean | 20411.5 |
| n | 39 |

## Per-Query Results

| query_id | category | nDCG@5 | recall@10 | MRR | latency (ms) | cite. | faith. | answ.rel. | ctx.prec. | ans.corr. |
|---|---|---|---|---|---|---|---|---|---|---|
| `q1_inter_basin` | factual | 0.000 | 0.000 | 0.000 | 25765 | — | 0.950 | 0.700 | 1.000 | 0.750 |
| `q2_bic_ranking` | factual | 0.000 | 0.000 | 0.000 | 22109 | 1.000 | 0.950 | 0.900 | 1.000 | 1.000 |
| `q3_approximation_options` | factual | 0.000 | 0.000 | 0.000 | 21655 | 1.000 | 0.950 | 0.900 | 0.600 | 0.000 |
| `q4_target_region` | multi_hop | 0.000 | 0.000 | 0.000 | 24125 | 1.000 | 1.000 | 0.900 | 0.800 | 1.000 |
| `q5_oc_weather` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16891 | 1.000 | 0.000 | 0.000 | 0.650 | — |
| `q6_basin_definition` | factual | 0.000 | 0.000 | 0.000 | 26250 | — | 0.950 | 0.800 | 1.000 | 0.800 |
| `q7_posterior_mixture` | factual | 0.000 | 0.000 | 0.000 | 21452 | 1.000 | 0.950 | 0.900 | 0.850 | 0.200 |
| `q8_benchmark_size` | factual | 0.000 | 0.000 | 0.000 | 19765 | 1.000 | 1.000 | 0.900 | 0.800 | 1.000 |
| `q9_baselines` | factual | 0.000 | 0.000 | 0.000 | 19717 | 1.000 | 0.950 | 0.900 | 0.850 | 0.800 |
| `q10_ablation_terms` | factual | 0.000 | 0.000 | 0.000 | 22766 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
| `q11_budget_levels` | factual | 0.000 | 0.000 | 0.000 | 22625 | 1.000 | 0.950 | 0.900 | 1.000 | 0.800 |
| `q12_multibasin_vs_vopt` | multi_hop | 0.000 | 0.000 | 0.000 | 20202 | 1.000 | 0.950 | 0.800 | 0.800 | 0.600 |
| `q13_figure1_lrbsz` | figure | 0.000 | 0.000 | 0.000 | 22327 | — | 0.950 | 0.900 | 0.850 | 1.000 |
| `q14_final_acquisition` | equation | 0.000 | 0.000 | 0.000 | 20266 | — | 1.000 | 0.300 | 1.000 | 1.000 |
| `q15_oc_gpt4` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16061 | 1.000 | 0.000 | 0.000 | 0.600 | — |
| `q16_aw_pinn_what` | factual | 0.000 | 0.000 | 0.000 | 20954 | — | 1.000 | 0.900 | 0.800 | 0.800 |
| `q17_oc_pinn_rl` | out_of_corpus | 0.000 | 0.000 | 0.000 | 15952 | 1.000 | 0.000 | 0.000 | 0.800 | — |
| `q18_synth_computer_count` | factual | 0.000 | 0.000 | 0.000 | 20562 | 1.000 | 0.950 | 0.800 | 0.950 | 1.000 |
| `q19_oc_synth_energy` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16047 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q20_exploration_hacking` | factual | 0.000 | 0.000 | 0.000 | 20827 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
| `q21_oc_llama_finetune` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16625 | 1.000 | 0.000 | 0.000 | 0.800 | — |
| `q22_mase_definition` | factual | 0.431 | 1.000 | 0.250 | 20469 | 1.000 | 0.950 | 0.900 | 0.800 | 1.000 |
| `q23_oc_eurozone` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16296 | — | 0.000 | 0.000 | 0.100 | — |
| `q24_lc_fig4_gradients` | figure | 0.000 | 0.000 | 0.000 | 19750 | 1.000 | 0.850 | 0.900 | 0.850 | 0.750 |
| `q25_lc_tab1_loss_properties` | table | 0.000 | 0.000 | 0.000 | 21344 | 1.000 | 0.950 | 0.900 | 0.600 | 0.800 |
| `q26_fd_tab4_imagenet256` | table | 0.000 | 0.000 | 0.000 | 21217 | — | 0.950 | 0.500 | 0.950 | 0.600 |
| `q27_fd_fig2_pipeline` | figure | 0.000 | 0.000 | 0.000 | 20094 | — | 0.950 | 0.900 | 1.000 | 0.800 |
| `q28_aegis_fig4_tasks` | figure | 0.000 | 0.000 | 0.000 | 20718 | 1.000 | 1.000 | 0.900 | 0.850 | 1.000 |
| `q29_aegis_tab8_janus_rank` | table | 0.000 | 0.000 | 0.000 | 20781 | — | 0.950 | 0.000 | 0.850 | 0.000 |
| `q30_a4mer_tab1_pose_datasets` | table | 0.000 | 0.000 | 0.000 | 19750 | — | 0.950 | 0.100 | 0.600 | 0.000 |
| `q31_hermes_fig1_unification` | figure | 0.000 | 0.000 | 0.000 | 21047 | 1.000 | 0.850 | 0.900 | 0.650 | 0.800 |
| `q32_pgp_fig1_methods` | figure | 0.000 | 0.000 | 0.000 | 21483 | 1.000 | 0.950 | 0.900 | 0.850 | 0.600 |
| `q33_oc_aegis_quantum` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16719 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q34_oc_lc_nas` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16968 | 1.000 | 0.000 | 0.000 | 0.950 | — |
| `q35_qd_fig6_rejection` | figure | 0.000 | 0.000 | 0.000 | 20812 | 1.000 | 0.850 | 1.000 | 0.850 | 1.000 |
| `q36_qd_fig7_layer_structure` | figure | 0.000 | 0.000 | 0.000 | 21422 | 1.000 | 0.950 | 0.600 | 0.850 | 0.900 |
| `q37_eh_fig1_outcomes` | figure | 0.000 | 0.000 | 0.000 | 21327 | 1.000 | 0.950 | 0.900 | 0.800 | 0.800 |
| `q38_eh_fig2_main_categories` | figure | 0.631 | 1.000 | 0.500 | 21657 | 1.000 | 0.950 | 1.000 | 0.850 | 0.800 |
| `q39_eh_fig2_terminal_vs_instrumental` | figure | 0.000 | 1.000 | 0.167 | 21250 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
