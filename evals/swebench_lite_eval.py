"""
SWEBench: A Benchmark for Evaluating large language models on real world software issues
John Yang, Carlos E. Jimenez, Alex L. Zhang, Kilian Lieret, Joyce Yang, Xindi Wu, Ori Press, Niklas Muennighoff, Gabriel Synnaeve, Karthik R. Narasimhan, Diyi Yang, Sida I. Wang, Ofir Press
https://arxiv.org/abs/2410.03859

SWEBenchLiteEval

A lightweight wrapper around SWE-bench harness to:
- sample SWE-bench Lite instances
- query a sampler for a patch
- apply patch in a container and run tests
- grade with the official harness parser

"""

from datasets import load_dataset
import docker
import json
from pathlib import Path, PurePosixPath
import os
import random
import traceback
import uuid

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.docker_build import build_env_images
from swebench.harness.docker_utils import cleanup_container
from swebench.harness.grading import get_eval_report
from swebench.harness.docker_build import build_container, close_logger, setup_logger
from swebench.harness.docker_utils import copy_to_container, remove_image, exec_run_with_timeout
from swebench.harness.constants import (
    DOCKER_PATCH, DOCKER_USER, DOCKER_WORKDIR, APPLY_PATCH_PASS, APPLY_PATCH_FAIL, UTF8,
    LOG_INSTANCE,
)
from swebench.inference.make_datasets.utils import extract_diff, extract_minimal_patch

from .. import common
from ..custom_types import Eval, EvalResult, SamplerBase, SingleEvalResult


class SWEBenchLiteEval(Eval):
    """Evaluate a sampler on SWE-bench Lite (BM25 13K) test split."""
    def __init__(
        self,
        num_examples: int | None = None,
    ):
        rows = load_dataset(
            "princeton-nlp/SWE-bench_Lite_bm25_13K", split="test")
        examples = [row for row in rows]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self._docker = docker.from_env()
        build_env_images(self._docker, self.examples, force_rebuild=False, max_workers=1)

    def grade_sample(self, row: dict, patch_text: str) -> str:
        instance_id = row.get(KEY_INSTANCE_ID)
        run_id = str(uuid.uuid4())
        namespace = "swebench" if hasattr(os, "uname") and os.uname().sysname == "Linux" else None
        test_spec = make_test_spec(
            row, namespace=namespace, instance_image_tag="latest")
        # Write prediction dict matching harness expectations
        pred = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: "model_name_placeholder",
            KEY_PREDICTION: patch_text or "",
        }
        container = None
        log_dir = RUN_EVALUATION_LOG_DIR / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(instance_id, log_dir /
                              LOG_INSTANCE, add_stdout=True)
        try:
            container = build_container(
                test_spec, self._docker, run_id, logger, nocache=False, force_rebuild=False)
            container.start()
            patch_file = Path(log_dir / "patch.diff")
            patch_file.write_text(patch_text or "", encoding=UTF8)
            logger.info(f"Patch file written to {patch_file}")
            logger.info(f"Patch Text: {patch_text}")
            copy_to_container(container, patch_file,
                              PurePosixPath(DOCKER_PATCH))
            GIT_APPLY_CMDS = [
                "git apply --verbose",
                "git apply --verbose --reject",
                "patch --batch --fuzz=5 -p1 -i",
            ]
            applied = False
            for cmd in GIT_APPLY_CMDS:
                val = container.exec_run(
                    f"{cmd} {DOCKER_PATCH}", workdir=DOCKER_WORKDIR, user=DOCKER_USER)
                if val.exit_code == 0:
                    logger.info(
                        f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}")
                    applied = True
                    break
                else:
                    logger.error(f"Failed to apply patch to container: {cmd}")
                    logger.error(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}")
            if not applied:
                # Treat as incorrect; produce minimal report-like dict
                return False, {"error": "patch_apply_failed", "output": val.output.decode(UTF8)}

            # Write eval script
            eval_file = Path(log_dir / "eval.sh")
            eval_file.write_text(test_spec.eval_script)
            copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))

            # Run with timeout
            TEST_TIMEOUT_S = 60
            test_output, timed_out, total_runtime = exec_run_with_timeout(
                container, "/bin/bash /eval.sh", TEST_TIMEOUT_S)
            test_output_path = log_dir / LOG_TEST_OUTPUT
            test_output_path.write_text(test_output)

            if timed_out:
                test_output_path.write_text(
                    test_output + f"\n\nTimeout error: {TEST_TIMEOUT_S} seconds exceeded.")
                return False, {"error": "timeout"}

            # Grade using the harness' parser
            report = get_eval_report(test_spec=test_spec, prediction=pred,
                                     test_log_path=test_output_path, include_tests_status=True)
            logger.info(f"Report: {report}")
            resolved = bool(report[instance_id]["resolved"]
                            ) if instance_id in report else False
            # Persist the report JSON (mirrors official behavior)
            (log_dir / LOG_REPORT).write_text(json.dumps(report, indent=4), encoding=UTF8)
            return resolved, report
        except Exception:
            logger.error(traceback.format_exc())
            return False, {"error": "exception"}
        finally:
            cleanup_container(self._docker, container, logger)
            remove_image(self._docker, test_spec.instance_image_key, logger)
            close_logger(logger)

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            issue = row.get("text", "")
            system_message = issue.split("\n", 1)[0]
            user_message = issue.split("\n", 1)[1]
            prompt_messages = [
                sampler._pack_message(content=system_message, role="system"),
                sampler._pack_message(content=user_message, role="user"),
            ]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            model_patch = extract_diff(response_text)
            minimal_patch = extract_minimal_patch(model_patch)
            print(f"[PROMPT]\n{prompt_messages}")
            print(f"[RESPONSE_TEXT]\n{response_text}")
            print(f"[MODEL_PATCH]\n{minimal_patch}")
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list
            resolved, report = self.grade_sample(row, minimal_patch)

            score = score = 1 if resolved else 0

            # Create HTML for each sample result
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["patch"],
                extracted_answer=response_text,
            )
            convo = actual_queried_prompt_messages + \
                [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)

        # Run evaluation and collect results
        results = common.map_with_progress(fn, self.examples)

        return common.aggregate_results(results)
