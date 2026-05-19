# Eval Report — phase1-text-baseline v3

- **Run ID:** `325375af3043`
- **Started:** 2026-05-19T22:04:50.737410+00:00
- **Finished:** 2026-05-19T22:19:24.077049+00:00
- **Queries:** 39

## Configuration

- `agentic`: `False`
- `agentic_max_subqueries`: `None`
- `agentic_model`: `None`
- `agentic_provider`: `None`
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
| nDCG@5 (macro) | 0.0264 |
| recall@10 (macro) | 0.1129 |
| MRR (macro) | 0.0328 |
| n in-corpus queries | 31 |

## Generation

| Metric | Value |
|---|---|
| citation grounding (mean) | 1.0000 |
| faithfulness (LLM judge) (mean) | 0.7454 |
| answer relevance (LLM judge) (mean) | 0.6179 |
| context precision (LLM judge) (mean) | 0.8077 |
| answer correctness vs expected_facts (LLM judge) (mean) | 0.7626 |
| total tokens in | 146619 |
| total tokens out | 16733 |

## Latency

| Metric | Value (ms) |
|---|---|
| p50 | 20311 |
| p95 | 24937 |
| mean | 22392.6 |
| n | 39 |

## Per-Query Results

| query_id | category | nDCG@5 | recall@10 | MRR | latency (ms) | cite. | faith. | answ.rel. | ctx.prec. | ans.corr. |
|---|---|---|---|---|---|---|---|---|---|---|
| `q1_inter_basin` | factual | 0.000 | 0.000 | 0.000 | 22608 | — | 0.950 | 0.700 | 0.600 | 0.200 |
| `q2_bic_ranking` | factual | 0.000 | 0.000 | 0.000 | 20484 | — | 0.950 | 0.700 | 0.600 | 0.600 |
| `q3_approximation_options` | factual | 0.000 | 0.000 | 0.000 | 24047 | 1.000 | 0.950 | 0.900 | 0.600 | 0.600 |
| `q4_target_region` | multi_hop | 0.000 | 0.000 | 0.000 | 22592 | 1.000 | 1.000 | 0.900 | 0.800 | 0.800 |
| `q5_oc_weather` | out_of_corpus | 0.000 | 0.000 | 0.000 | 18688 | 1.000 | 0.000 | 0.000 | 0.650 | — |
| `q6_basin_definition` | factual | 0.000 | 0.000 | 0.000 | 28640 | — | 0.950 | 0.700 | 1.000 | 0.800 |
| `q7_posterior_mixture` | factual | 0.000 | 0.000 | 0.000 | 22609 | 1.000 | 0.950 | 0.900 | 0.600 | 0.800 |
| `q8_benchmark_size` | factual | 0.000 | 0.000 | 0.000 | 20311 | 1.000 | 1.000 | 0.900 | 0.800 | 1.000 |
| `q9_baselines` | factual | 0.000 | 0.000 | 0.000 | 19375 | 1.000 | 1.000 | 1.000 | 0.850 | 1.000 |
| `q10_ablation_terms` | factual | 0.000 | 0.000 | 0.000 | 20438 | — | 0.950 | 0.900 | 1.000 | 1.000 |
| `q11_budget_levels` | factual | 0.000 | 0.000 | 0.000 | 22733 | 1.000 | 0.950 | 0.900 | 0.850 | 0.800 |
| `q12_multibasin_vs_vopt` | multi_hop | 0.000 | 0.000 | 0.000 | 24937 | 1.000 | 1.000 | 0.900 | 0.800 | 0.800 |
| `q13_figure1_lrbsz` | figure | 0.000 | 0.000 | 0.000 | 22297 | 1.000 | 0.950 | 0.700 | 0.950 | 0.670 |
| `q14_final_acquisition` | equation | 0.000 | 0.000 | 0.000 | 20250 | — | 1.000 | 0.900 | 0.850 | 1.000 |
| `q15_oc_gpt4` | out_of_corpus | 0.000 | 0.000 | 0.000 | 15859 | — | 0.000 | 0.000 | 0.650 | — |
| `q16_aw_pinn_what` | factual | 0.000 | 0.000 | 0.000 | 23155 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
| `q17_oc_pinn_rl` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16327 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q18_synth_computer_count` | factual | 0.431 | 1.000 | 0.250 | 18875 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
| `q19_oc_synth_energy` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16266 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q20_exploration_hacking` | factual | 0.000 | 0.000 | 0.000 | 19718 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
| `q21_oc_llama_finetune` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16703 | 1.000 | 0.000 | 0.000 | 0.950 | — |
| `q22_mase_definition` | factual | 0.000 | 1.000 | 0.100 | 21905 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
| `q23_oc_eurozone` | out_of_corpus | 0.000 | 0.000 | 0.000 | 15359 | — | 0.000 | 0.000 | 0.100 | — |
| `q24_lc_fig4_gradients` | figure | 0.000 | 0.000 | 0.000 | 19077 | 1.000 | 0.850 | 0.100 | 0.800 | 0.000 |
| `q25_lc_tab1_loss_properties` | table | 0.000 | 0.000 | 0.000 | 20890 | 1.000 | 0.950 | 0.900 | 0.800 | 0.800 |
| `q26_fd_tab4_imagenet256` | table | 0.000 | 0.000 | 0.000 | 19297 | — | 0.950 | 0.900 | 0.950 | 1.000 |
| `q27_fd_fig2_pipeline` | figure | 0.000 | 0.000 | 0.000 | 20250 | — | 1.000 | 0.900 | 0.850 | 0.800 |
| `q28_aegis_fig4_tasks` | figure | 0.000 | 0.000 | 0.000 | 19609 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
| `q29_aegis_tab8_janus_rank` | table | 0.000 | 0.000 | 0.000 | 21530 | — | 0.950 | 0.000 | 0.850 | 0.000 |
| `q30_a4mer_tab1_pose_datasets` | table | 0.000 | 0.000 | 0.000 | 104672 | — | 0.620 | 0.000 | 0.800 | 0.000 |
| `q31_hermes_fig1_unification` | figure | 0.387 | 0.500 | 0.500 | 20562 | 1.000 | 0.950 | 0.900 | 0.850 | 0.800 |
| `q32_pgp_fig1_methods` | figure | 0.000 | 0.000 | 0.000 | 19702 | — | 0.850 | 0.500 | 0.800 | 0.670 |
| `q33_oc_aegis_quantum` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16344 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q34_oc_lc_nas` | out_of_corpus | 0.000 | 0.000 | 0.000 | 16202 | 1.000 | 0.000 | 0.000 | 0.800 | — |
| `q35_qd_fig6_rejection` | figure | 0.000 | 0.000 | 0.000 | 20562 | 1.000 | 0.850 | 0.900 | 0.800 | 1.000 |
| `q36_qd_fig7_layer_structure` | figure | 0.000 | 0.000 | 0.000 | 20390 | 1.000 | 0.850 | 0.800 | 0.950 | 0.800 |
| `q37_eh_fig1_outcomes` | figure | 0.000 | 0.000 | 0.000 | 19157 | 1.000 | 0.950 | 0.900 | 0.950 | 0.800 |
| `q38_eh_fig2_main_categories` | figure | 0.000 | 0.000 | 0.000 | 21015 | 1.000 | 0.950 | 0.900 | 0.800 | 1.000 |
| `q39_eh_fig2_terminal_vs_instrumental` | figure | 0.000 | 1.000 | 0.167 | 19875 | 1.000 | 1.000 | 0.900 | 0.850 | 0.900 |
