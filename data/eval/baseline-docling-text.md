# Eval Report — phase1-text-baseline v3

- **Run ID:** `e7643a0c1085`
- **Started:** 2026-05-20T11:52:36.101704+00:00
- **Finished:** 2026-05-20T12:46:15.536810+00:00
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
- `use_docling`: `True`
- `visual_device`: `None`
- `visual_model`: `None`
- `vlm_caption_model`: `None`
- `vlm_caption_provider`: `None`

## Retrieval (in-corpus queries)

| Metric | Value |
|---|---|
| nDCG@5 (macro) | 0.0467 |
| recall@10 (macro) | 0.1290 |
| MRR (macro) | 0.0360 |
| n in-corpus queries | 31 |

## Generation

| Metric | Value |
|---|---|
| citation grounding (mean) | 1.0000 |
| faithfulness (LLM judge) (mean) | 0.8110 |
| answer relevance (LLM judge) (mean) | 0.7654 |
| context precision (LLM judge) (mean) | 0.8859 |
| answer correctness vs expected_facts (LLM judge) (mean) | 0.8255 |
| total tokens in | 106852 |
| total tokens out | 11637 |

## Latency

| Metric | Value (ms) |
|---|---|
| p50 | 83484 |
| p95 | 102969 |
| mean | 82549.3 |
| n | 39 |

## Per-Query Results

| query_id | category | nDCG@5 | recall@10 | MRR | latency (ms) | cite. | faith. | answ.rel. | ctx.prec. | ans.corr. |
|---|---|---|---|---|---|---|---|---|---|---|
| `q1_inter_basin` | factual | 0.000 | 0.000 | 0.000 | 92422 | 1.000 | 0.950 | 0.900 | 0.850 | 0.800 |
| `q2_bic_ranking` | factual | 0.000 | 0.000 | 0.000 | 86327 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
| `q3_approximation_options` | factual | 0.000 | 0.000 | 0.000 | 85047 | 1.000 | 0.950 | 0.900 | 0.850 | 1.000 |
| `q4_target_region` | multi_hop | 0.000 | 0.000 | 0.000 | 102969 | 1.000 | 1.000 | 0.900 | 0.850 | 1.000 |
| `q5_oc_weather` | out_of_corpus | 0.000 | 0.000 | 0.000 | 29218 | 1.000 | 1.000 | 1.000 | 0.650 | — |
| `q6_basin_definition` | factual | 0.000 | 0.000 | 0.000 | 73109 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
| `q7_posterior_mixture` | factual | 0.000 | 0.000 | 0.000 | 88437 | 1.000 | 0.950 | 0.900 | 0.850 | 0.600 |
| `q8_benchmark_size` | factual | 0.000 | 0.000 | 0.000 | 94922 | 1.000 | 1.000 | 0.900 | 0.950 | 1.000 |
| `q9_baselines` | factual | 0.000 | 0.000 | 0.000 | 91265 | 1.000 | 0.980 | 0.900 | 0.950 | 0.800 |
| `q10_ablation_terms` | factual | 0.000 | 0.000 | 0.000 | 83297 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
| `q11_budget_levels` | factual | 0.000 | 0.000 | 0.000 | 98015 | 1.000 | 0.950 | 0.900 | 0.950 | 0.800 |
| `q12_multibasin_vs_vopt` | multi_hop | 0.000 | 0.000 | 0.000 | 83375 | 1.000 | 0.950 | 0.900 | 0.850 | 0.800 |
| `q13_figure1_lrbsz` | figure | 0.000 | 0.000 | 0.000 | 105937 | 1.000 | 0.950 | 0.900 | 0.950 | 0.660 |
| `q14_final_acquisition` | equation | 0.000 | 0.000 | 0.000 | 101469 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
| `q15_oc_gpt4` | out_of_corpus | 0.000 | 0.000 | 0.000 | 65186 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q16_aw_pinn_what` | factual | 0.000 | 0.000 | 0.000 | 85969 | 1.000 | 0.950 | 0.900 | 0.950 | 0.800 |
| `q17_oc_pinn_rl` | out_of_corpus | 0.000 | 0.000 | 0.000 | 82109 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q18_synth_computer_count` | factual | 0.631 | 1.000 | 0.500 | 78717 | 1.000 | 0.950 | 0.900 | 0.800 | 1.000 |
| `q19_oc_synth_energy` | out_of_corpus | 0.000 | 0.000 | 0.000 | 95359 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q20_exploration_hacking` | factual | 0.000 | 0.000 | 0.000 | 83484 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
| `q21_oc_llama_finetune` | out_of_corpus | 0.000 | 0.000 | 0.000 | 75405 | 1.000 | 0.000 | 0.000 | 0.850 | — |
| `q22_mase_definition` | factual | 0.000 | 0.000 | 0.000 | 79547 | 1.000 | 0.950 | 1.000 | 0.950 | 1.000 |
| `q23_oc_eurozone` | out_of_corpus | 0.000 | 0.000 | 0.000 | 53375 | 1.000 | 1.000 | 1.000 | 0.650 | — |
| `q24_lc_fig4_gradients` | figure | 0.000 | 0.000 | 0.000 | 94952 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
| `q25_lc_tab1_loss_properties` | table | 0.000 | 0.000 | 0.000 | 100875 | 1.000 | 0.950 | 0.900 | 0.950 | 0.800 |
| `q26_fd_tab4_imagenet256` | table | 0.000 | 0.000 | 0.000 | 85969 | 1.000 | 0.950 | 0.900 | 0.950 | 1.000 |
| `q27_fd_fig2_pipeline` | figure | 0.000 | 0.000 | 0.000 | 107812 | 1.000 | 0.950 | 0.900 | 0.950 | 0.800 |
| `q28_aegis_fig4_tasks` | figure | 0.000 | 0.000 | 0.000 | 75750 | 1.000 | 1.000 | 0.900 | 0.850 | 1.000 |
| `q29_aegis_tab8_janus_rank` | table | 0.000 | 0.000 | 0.000 | 68687 | 1.000 | 0.950 | 0.750 | 0.950 | 0.000 |
| `q30_a4mer_tab1_pose_datasets` | table | 0.000 | 0.000 | 0.000 | 87218 | 1.000 | 0.950 | 0.800 | 0.850 | 0.600 |
| `q31_hermes_fig1_unification` | figure | 0.000 | 0.000 | 0.000 | 70422 | 1.000 | 0.950 | 0.900 | 0.950 | 0.600 |
| `q32_pgp_fig1_methods` | figure | 0.000 | 0.000 | 0.000 | 88547 | 1.000 | 0.950 | 1.000 | 0.850 | 0.800 |
| `q33_oc_aegis_quantum` | out_of_corpus | 0.000 | 0.000 | 0.000 | 59859 | 1.000 | 0.000 | 0.000 | 0.800 | — |
| `q34_oc_lc_nas` | out_of_corpus | 0.000 | 0.000 | 0.000 | 69843 | 1.000 | 0.000 | 0.000 | 0.950 | — |
| `q35_qd_fig6_rejection` | figure | 0.000 | 0.000 | 0.000 | 75890 | 1.000 | 0.950 | 1.000 | 0.950 | 1.000 |
| `q36_qd_fig7_layer_structure` | figure | 0.431 | 1.000 | 0.250 | 80391 | 1.000 | 0.950 | 0.800 | 0.800 | 0.330 |
| `q37_eh_fig1_outcomes` | figure | 0.000 | 0.000 | 0.000 | 73608 | 1.000 | 0.950 | 0.900 | 0.800 | 0.800 |
| `q38_eh_fig2_main_categories` | figure | 0.387 | 1.000 | 0.200 | 89312 | 1.000 | 0.950 | 0.900 | 0.950 | 0.800 |
| `q39_eh_fig2_terminal_vs_instrumental` | figure | 0.000 | 1.000 | 0.167 | 75328 | 1.000 | 0.950 | 0.900 | 0.950 | 0.800 |
