# Governance and risk

This document lists what can go wrong with a ResearchGenie run and how the product stops it. For each risk it names the control, where the control is in the code, the test that proves the control works, and how to roll back. It is for students, supervisors and reviewers.

ResearchGenie never presents AI synthesis as verified fact. Every draft starts with a "for human review" disclaimer. The controls below exist to show the user where to look. They do not replace the user's judgement.

## Rollback, for every risk

Every run keeps all of its work on disk, so nothing is lost and nothing has to be redone from scratch:

- **Each stage saves its result** in the run folder (`01_discovery/` … `04_quality_assurance/result.json`), and `events.jsonl` logs every event. A crash keeps what has been logged so far.
- **Resume** (`researchgenie resume <run>`, `--from STAGE`, or the web **Resume** button) loads the earlier stages from disk and re-runs only the stage that failed, or the stage you name.
- **Versioned final output.** A resume never overwrites results. The existing `final/` folder is kept as `final_v1/`, `final_v2/`, …, so you can compare the versions or go back to one.

Evidence: `tests/orchestrator/test_resume.py` (`test_resuming_a_run_that_failed_in_verification_skips_discovery_and_writing`, `test_an_existing_final_folder_is_kept_as_a_numbered_version`, `test_a_corrupt_stage_result_fails_with_a_message_naming_the_stage`), and `tests/orchestrator/test_pipeline.py::test_run_pipeline_appends_every_event_to_events_jsonl_including_failure`.

## Risks and controls

| Risk | Control in the product | Where | Evidence (tests) | Rollback |
|---|---|---|---|---|
| **Invented citation** (a reference that doesn't exist) | Writing may cite only the selected papers. A citation to anything else is rejected and that section is revised. References are rendered from paper metadata, never written by the model. Verification checks every DOI against Crossref, then OpenAlex. A DOI found in neither is marked **invalid**. A DOI that got no answer is marked **unverifiable**, never invalid. The draft is flagged when any reference fails. | `services/research-writing/writing/modules/section_auditor.py`, `citation_formatter.py`, `services/verification/service.py` | `test_writing_citation_id_normalization.py::test_genuinely_out_of_scope_citation_is_still_correctly_rejected`, `test_writing_section_reliability.py::test_c_invalid_citation_triggers_targeted_revision_not_blind_retry`, `test_verification_service.py::test_run_verification_classifies_verified_invalid_and_unverifiable`, `test_verification_gateway.py::test_a_doi_both_providers_answer_404_for_is_still_invalid`, `test_verification_gateway.py::test_a_real_doi_throttled_once_by_crossref_is_verified_not_unverifiable` | Fix or drop the reference by hand, guided by `validation_report.md`. Then resume from `verification`. |
| **Off-topic paper** | A hard relevance gate screens every candidate against the question before ranking. A thin corpus triggers one re-planned search. The **approval step** shows the selected papers before any writing, and the user can drop papers by number; replacements come from the ranked candidates. The model's choice of databases is validated, and the domain fallback skips databases outside the field. | `services/research-discovery/discovery/relevance.py`, `pipeline.py` (`_approve_corpus`), `source_plan.py` | `test_discovery_relevance.py` (all rows), `test_discovery_replan.py`, `test_discovery_approval.py::test_the_final_draft_never_cites_a_dropped_paper`, `test_discovery_source_plan.py::test_invalid_tool_calls_are_rejected_and_fall_back` | Resume from `discovery` (and drop the paper at the approval step), or start a new run with sharper keywords. |
| **Unsupported claim** | Quality Assurance is deterministic, with no model call. It flags sentences whose cited papers don't support them, and empirical claims with no citation. The evidence table maps each paper → evidence point → section → DOI → verification status, so a reader can check any claim. Writing marks a section it can't ground as unresolved; it is never silently accepted. | `services/quality-assurance/service.py`, `shared/utilities/evidence_table.py`, `services/research-writing/completeness.py` | `test_qa_service.py::test_find_unsupported_claims_flags_low_overlap_and_skips_unmatched`, `test_qa_service.py::test_find_missing_citations_flags_uncited_empirical_claims`, `test_evidence_table.py::test_rows_join_used_points_to_sections_and_verification`, `test_writing_section_reliability.py::test_unrecoverable_section_is_explicitly_marked_unresolved_not_silently_accepted` | Edit the flagged sentences by hand, or resume from `writing` to draft again. |
| **Key leak** (Gemini, OpenAlex, NCBI keys or the contact email showing up in output) | Keys are stored only in `~/.researchgenie/config.json`, outside the repository. They reach the pipeline only as environment variables. The Gemini key travels in a header, never in a URL. Every URL, log line, error, event-log line and Doctor report is passed through `redact()`. The response cache's keys leave out every identity parameter. Printed config views are masked. | `researchgenie/config.py`, `shared/utilities/scholarly_gateway.py` (`redact`), `shared/utilities/response_cache.py` (`key`), `orchestrator/pipeline.py` (event log), `orchestrator/doctor.py` (`write_report`) | `test_scholarly_gateway.py::test_no_key_or_email_leaks_into_errors_tracebacks_or_logs`, `test_pipeline.py::test_no_key_or_email_reaches_the_event_log_or_the_run_folder`, `test_response_cache.py::test_cache_keys_never_contain_a_key_or_email`, `test_config.py::test_masked_summary_never_exposes_the_full_api_key`, `test_scholarly_setup.py::test_an_invalid_key_is_reported_with_the_fix_and_without_the_key` | Revoke the key at the provider, then run `researchgenie config` to enter a new one. |
| **Exhausted provider** (rate limits, daily budget gone) | One gateway paces each host and retries within each provider's published limits. A circuit breaker skips a host that keeps failing for the rest of the run, and the run continues on the other sources. OpenAlex's spent budget is never retried; it is remembered across runs until the reset. A response cache answers repeat requests at no quota. Every skip is a visible event that names the fix (usually a free key). The **Doctor** diagnoses a run that failed this way. | `shared/utilities/scholarly_gateway.py`, `response_cache.py`, `orchestrator/doctor.py` | `test_scholarly_gateway.py::test_not_enough_credits_is_never_retried_and_marks_the_host_exhausted`, `test_scholarly_gateway.py::test_repeated_5xx_across_requests_trips_the_breaker_for_the_rest_of_the_run`, `test_discovery_source_failures.py::test_three_failed_sources_still_leave_a_corpus_from_the_fourth`, `test_response_cache.py::test_exhaustion_and_balance_survive_a_process_restart`, `test_doctor.py::test_each_rule_row[providers_exhausted]` | Add the free key (`researchgenie config`) or wait for the reset, then resume (`researchgenie doctor <run> --yes` does both checks). |
| **Bad output** (incomplete, broken or short draft; failed run) | Draft completeness is checked on the rendered paper text, not on the model's own report. A partial or failed draft is labelled as such in `metadata.json` and on every surface. A run that fails stops at that stage rather than passing on partial data. The Doctor names the cause from a fixed rule table and proposes one fix. Its "no known cause" answer quotes the last error and never guesses. | `services/research-writing/completeness.py`, `orchestrator/pipeline.py` (`PipelineError`), `orchestrator/doctor.py` | `test_completeness.py` (all rows), `test_doctor.py::test_each_rule_row`, `test_doctor.py::test_a_healthy_completed_run_and_an_unknown_failure_are_reported_honestly` | Resume from `writing` (the Doctor proposes this for a partial draft). Earlier finals stay as `final_vN/`. |

## Human checkpoints

### The approval step

After Discovery selects the papers, an interactive run (the CLI or the web page) pauses and shows them before full-text retrieval, analysis and writing. The user approves them or drops papers by number. A dropped paper is replaced from the ranked candidates and is never read or cited afterwards. The decision (who approved, when, what was dropped and what replaced it) is logged in `events.jsonl`. Headless runs, scripts and tests approve automatically, so automation never stalls. The log still records `approved_by: auto`.

Where: `services/research-discovery/discovery/pipeline.py::_approve_corpus`, `orchestrator/run_registry.py::approver`. Evidence: `tests/services/test_discovery_approval.py`.

### The Doctor's confirmation rule

The Doctor diagnoses without asking, because diagnosing only reads the run. It **never applies a fix without confirmation**:

- The interactive CLI asks "Apply this fix?" and the default answer is no.
- `researchgenie doctor --json` applies the fix only with `--yes`.
- The API applies a fix only when the request has `apply: true`. The web page sends that only when the user clicks **Apply fix** after the diagnosis and the proposed fix are on screen.
- A fix that needs the user (entering a key, installing LaTeX, rephrasing the question) is reported as a step and is never run from headless mode or the web.

Every diagnosis writes `doctor_report.json` in the run folder, with the cause, the evidence lines, the proposed fix, whether it was applied, and the outcome. The outcome is updated when the resumed run ends.

Where: `orchestrator/doctor.py`, `researchgenie/headless.py::_doctor`, `orchestrator/api.py::doctor_research_run`. Evidence: `tests/orchestrator/test_doctor.py::test_headless_needs_yes_before_any_fix_is_applied`, `test_the_api_diagnoses_then_applies_only_when_asked`, `test_a_fix_for_you_is_never_run`.

## Automation surfaces

### Exec safety: headless CLI and agent skill

The agent skill (#21, the next round) drives ResearchGenie only through the headless CLI. The CLI already follows the skill's exec-safety rules:

- **Fixed command lines** that an exec policy can allowlist: `researchgenie run|status|result|resume|doctor`. There is no inline code evaluation and no shell. User input travels only as single argv items or as fields that pydantic validates against the research-request contract.
- `run --detach` starts the same fixed argv in a background process, without a shell.
- **Meaningful exit codes** (0 = success, 3 = model unavailable, 4 = providers exhausted, 5 = no relevant papers; see the README) let an agent react without parsing text.
- There are no install-time scripts and no pipe-to-shell steps.

When #21 ships, the skill must declare every binary and environment variable it needs and keep to the commands above. Add its validator result here.

Where: `researchgenie/headless.py`. Evidence: `tests/researchgenie/test_headless.py::test_detach_prints_the_run_id_and_starts_a_fixed_background_command`, `test_each_failure_class_has_its_own_exit_code`, `test_bad_arguments_exit_2`.

### The MCP server is read-only

`researchgenie-mcp` exposes six tools for other MCP clients: list runs, read a final file, search the library, summarize a run, search the literature and verify a DOI.

- No tool writes a run folder, the library or the config, and no tool starts a run. Long runs go through the headless CLI, because MCP request timeouts are short.
- An artifact read must name one of the six text final files exactly, and the resolved path must stay inside the run folder. Any other name or path is refused.
- Searching a library that doesn't exist returns nothing and does not create the file.
- The literature and DOI tools go through the scholarly gateway, so they share its pacing, keys, redaction and response cache. The cache is the only disk write.

Where: `researchgenie/mcp_server.py`, `docs/mcp-server.md`. Evidence: `tests/researchgenie/test_mcp_server.py` (`test_the_six_read_only_tools_return_the_right_data` checks that the run folder and library are unchanged; `test_artifact_reads_outside_the_final_files_are_refused`; `test_no_tool_starts_a_run_or_writes`).

### The Dev Agent changes code only through a human

The Dev Agent works in a throwaway git worktree, writes only the paths in its plan, and runs tests in a subprocess with a timeout. It never commits, pushes or merges: git commands outside an allowlist are refused before a process starts. A human reviews `patch.diff` and applies it.

Where: `dev_agent/agent.py`. Evidence: `tests/dev_agent/test_dev_agent.py::test_a_write_outside_the_planned_paths_is_refused`, `test_no_commit_push_or_merge_happens_in_any_path`.

## Known gaps

- The **model-chosen sources** spike (#16) has not been run on a local model yet (`scripts/spike_tool_calling.py`). Until it has, treat the domain fallback as the tested path.
- The **Doctor's live demo** (stop Ollama mid-run, diagnose, restart, resume to completion) and the **Dev Agent's APA run** (#26, `docs/tickets/26-apa-7.md`) need a machine with Ollama or a Gemini key, and are not recorded yet.
- The agent skill (#21) does not exist yet. Its exec-safety rules above apply when it is built.
