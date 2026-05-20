# Eval Report — phase1-text-baseline v3

- **Run ID:** `54d579118679`
- **Started:** 2026-05-20T09:52:58.996464+00:00
- **Finished:** 2026-05-20T10:42:44.381900+00:00
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
- `extract_figures`: `True`
- `extract_tables`: `True`
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
- `use_docling`: `True`
- `visual_device`: `None`
- `visual_model`: `None`
- `vlm_caption_model`: `None`
- `vlm_caption_provider`: `None`

## Retrieval (in-corpus queries)

| Metric | Value |
|---|---|
| nDCG@5 (macro) | 0.0389 |
| recall@10 (macro) | 0.1290 |
| MRR (macro) | 0.0479 |
| n in-corpus queries | 31 |

## Generation

| Metric | Value |
|---|---|
| citation grounding (mean) | 1.0000 |
| faithfulness (LLM judge) (mean) | 0.7487 |
| answer relevance (LLM judge) (mean) | 0.6321 |
| context precision (LLM judge) (mean) | 0.7974 |
| answer correctness vs expected_facts (LLM judge) (mean) | 0.7113 |
| total tokens in | 142127 |
| total tokens out | 11256 |

## Latency

| Metric | Value (ms) |
|---|---|
| p50 | 77984 |
| p95 | 94422 |
| mean | 76548.1 |
| n | 39 |

## Per-Query Results

| query_id | category | nDCG@5 | recall@10 | MRR | latency (ms) | cite. | faith. | answ.rel. | ctx.prec. | ans.corr. |
|---|---|---|---|---|---|---|---|---|---|---|
| `q1_inter_basin` | factual | 0.000 | 0.000 | 0.000 | 94422 | — | 0.950 | 0.800 | 0.600 | 0.250 |
| `q2_bic_ranking` | factual | 0.000 | 0.000 | 0.000 | 76421 | — | 0.950 | 0.600 | 0.600 | 0.800 |
| `q3_approximation_options` | factual | 0.000 | 0.000 | 0.000 | 67719 | — | 0.950 | 0.800 | 0.600 | 0.700 |
| `q4_target_region` | multi_hop | 0.000 | 0.000 | 0.000 | 96984 | 1.000 | 1.000 | 0.900 | 0.800 | 1.000 |
| `q5_oc_weather` | out_of_corpus | 0.000 | 0.000 | 0.000 | 36405 | 1.000 | 0.000 | 0.000 | 0.200 | — |
| `q6_basin_definition` | factual | 0.000 | 0.000 | 0.000 | 131890 | — | 0.950 | 0.700 | 1.000 | 0.000 |
| `q7_posterior_mixture` | factual | 0.000 | 0.000 | 0.000 | 78547 | 1.000 | 0.950 | 0.900 | 0.600 | 0.600 |
| `q8_benchmark_size` | factual | 0.000 | 0.000 | 0.000 | 68452 | 1.000 | 0.950 | 0.900 | 0.800 | 1.000 |
| `q9_baselines` | factual | 0.000 | 0.000 | 0.000 | 64484 | 1.000 | 0.950 | 0.900 | 0.850 | 0.800 |
| `q10_ablation_terms` | factual | 0.000 | 0.000 | 0.000 | 80000 | — | 0.950 | 0.200 | 1.000 | 0.000 |
| `q11_budget_levels` | factual | 0.000 | 0.000 | 0.000 | 91359 | 1.000 | 0.950 | 0.900 | 0.800 | 0.700 |
| `q12_multibasin_vs_vopt` | multi_hop | 0.000 | 0.000 | 0.000 | 88420 | 1.000 | 1.000 | 0.900 | 0.800 | 0.800 |
| `q13_figure1_lrbsz` | figure | 0.000 | 0.000 | 0.000 | 80625 | — | 0.950 | 0.800 | 0.850 | 0.750 |
| `q14_final_acquisition` | equation | 0.000 | 0.000 | 0.000 | 92484 | 1.000 | 0.950 | 0.600 | 0.800 | 0.670 |
| `q15_oc_gpt4` | out_of_corpus | 0.000 | 0.000 | 0.000 | 47750 | — | 0.000 | 0.000 | 0.700 | — |
| `q16_aw_pinn_what` | factual | 0.000 | 0.000 | 0.000 | 93140 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
| `q17_oc_pinn_rl` | out_of_corpus | 0.000 | 0.000 | 0.000 | 65172 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q18_synth_computer_count` | factual | 0.431 | 1.000 | 0.250 | 65219 | 1.000 | 0.950 | 0.900 | 0.800 | 1.000 |
| `q19_oc_synth_energy` | out_of_corpus | 0.000 | 0.000 | 0.000 | 60718 | 1.000 | 0.000 | 0.000 | 0.800 | — |
| `q20_exploration_hacking` | factual | 0.000 | 0.000 | 0.000 | 86906 | 1.000 | 1.000 | 0.900 | 0.850 | 0.900 |
| `q21_oc_llama_finetune` | out_of_corpus | 0.000 | 0.000 | 0.000 | 62718 | 1.000 | 0.000 | 0.000 | 0.950 | — |
| `q22_mase_definition` | factual | 0.000 | 1.000 | 0.111 | 85562 | 1.000 | 0.950 | 1.000 | 0.800 | 1.000 |
| `q23_oc_eurozone` | out_of_corpus | 0.000 | 0.000 | 0.000 | 53062 | — | 0.000 | 0.000 | 0.200 | — |
| `q24_lc_fig4_gradients` | figure | 0.000 | 0.000 | 0.000 | 77984 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
| `q25_lc_tab1_loss_properties` | table | 0.387 | 0.500 | 0.500 | 81000 | 1.000 | 0.950 | 0.900 | 0.800 | 0.800 |
| `q26_fd_tab4_imagenet256` | table | 0.000 | 0.000 | 0.000 | 91719 | 1.000 | 0.950 | 0.500 | 1.000 | 0.330 |
| `q27_fd_fig2_pipeline` | figure | 0.000 | 0.000 | 0.000 | 67483 | — | 0.950 | 0.900 | 0.850 | 0.200 |
| `q28_aegis_fig4_tasks` | figure | 0.000 | 0.000 | 0.000 | 76000 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
| `q29_aegis_tab8_janus_rank` | table | 0.000 | 0.000 | 0.000 | 78547 | — | 0.950 | 0.000 | 0.850 | 0.000 |
| `q30_a4mer_tab1_pose_datasets` | table | 0.000 | 0.000 | 0.000 | 91625 | 1.000 | 0.950 | 0.600 | 0.800 | 0.000 |
| `q31_hermes_fig1_unification` | figure | 0.387 | 0.500 | 0.500 | 76594 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
| `q32_pgp_fig1_methods` | figure | 0.000 | 0.000 | 0.000 | 79468 | 1.000 | 0.850 | 0.900 | 0.800 | 1.000 |
| `q33_oc_aegis_quantum` | out_of_corpus | 0.000 | 0.000 | 0.000 | 57577 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q34_oc_lc_nas` | out_of_corpus | 0.000 | 0.000 | 0.000 | 54172 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q35_qd_fig6_rejection` | figure | 0.000 | 0.000 | 0.000 | 70092 | 1.000 | 0.850 | 1.000 | 0.850 | 1.000 |
| `q36_qd_fig7_layer_structure` | figure | 0.000 | 0.000 | 0.000 | 81359 | 1.000 | 0.750 | 0.750 | 0.950 | 0.750 |
| `q37_eh_fig1_outcomes` | figure | 0.000 | 0.000 | 0.000 | 74172 | 1.000 | 0.950 | 1.000 | 0.950 | 1.000 |
| `q38_eh_fig2_main_categories` | figure | 0.000 | 0.000 | 0.000 | 84750 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
| `q39_eh_fig2_terminal_vs_instrumental` | figure | 0.000 | 1.000 | 0.125 | 74375 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
