from .evals.mmlu_tr_eval import MMLUTrEval
from .evals.swebench_lite_eval import SWEBenchLiteEval
from .evals.browsecomp_eval import BrowseCompEval
from .evals.drop_eval import DropEval
from .evals.gpqa_eval import GPQAEval
from .evals.healthbench_eval import HealthBenchEval
from .evals.healthbench_meta_eval import HealthBenchMetaEval
from .evals.math_eval import MathEval
from .evals.mgsm_eval import MGSMEval
from .evals.mmlu_eval import MMLUEval
from .evals.humaneval_eval import HumanEval
from .evals.simpleqa_eval import SimpleQAEval


class EvalBuilder:
    """Builds evaluation instances based on the selected evaluation name."""
    SUPPORTED_EVALS = [
        "mmlu",
        "math",
        "gpqa",
        "mgsm",
        "drop",
        "humaneval",
        "simpleqa",
        "browsecomp",
        "healthbench",
        "healthbench_hard",
        "healthbench_consensus",
        "healthbench_meta",
        "swebench_verified",
        "mmlu_tr",
    ]

    def __init__(self, args, equality_checker, grading_sampler):
        self.args = args
        self.equality_checker = equality_checker
        self.grading_sampler = grading_sampler

    def build(self, eval_name):
        args = self.args
        equality_checker = self.equality_checker
        grading_sampler = self.grading_sampler
        debug_mode = args.debug
        num_examples = (
            args.examples if args.examples is not None else (
                5 if debug_mode else None)
        )
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples)
            case "math":
                return MathEval(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    n_repeats=1 if debug_mode else args.n_repeats or 10,
                )
            case "gpqa":
                return GPQAEval(
                    n_repeats=1 if debug_mode else args.n_repeats or 10,
                    num_examples=num_examples,
                )
            case "mgsm":
                return MGSMEval(
                    num_examples_per_lang=10 if debug_mode else num_examples or 250
                )
            case "drop":
                return DropEval(
                    num_examples=10 if debug_mode else num_examples,
                    train_samples_per_prompt=3,
                )
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case "simpleqa":
                return SimpleQAEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "browsecomp":
                return BrowseCompEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "healthbench":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name=None,
                )
            case "healthbench_hard":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name="hard",
                )
            case "healthbench_consensus":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name="consensus",
                )
            case "healthbench_meta":
                return HealthBenchMetaEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                )
            case "swebench_lite":
                return SWEBenchLiteEval(
                    num_examples=num_examples,
                )
            case "mmlu_tr":
                return MMLUTrEval(num_examples)
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")
