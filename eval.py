import logging
import time

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from agent import Agent
from config import (
    GRADER_MAX_TOKENS,
    GRADER_MODEL,
    ModelConfig,
    OPENROUTER_API_KEY,
    OPTIMIZER_MAX_TOKENS,
    OPTIMIZER_MODEL,
)
from dataset import BrowseCompDataset
from models import BrowseCompJudge, LLMJudgeAnswer
from utils import create_isolated_workspace, create_model_cli_parser, start_cleanup_watchdog

logger = logging.getLogger(__name__)

LM_PRICING = {
    "openrouter/x-ai/grok-4.1-fast": {"input": 0.20, "output": 0.50},
    "openrouter/deepseek/deepseek-v3.2": {"input": 0.24, "output": 0.38},
    "openrouter/google/gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    "openrouter/google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "openrouter/openai/gpt-oss-120b": {"input": 0.039, "output": 0.19},
    "openrouter/qwen/qwen3-235b-a22b-2507": {"input": 0.071, "output": 0.463},
    "openrouter/openai/gpt-5.2": {"input": 1.75, "output": 14.0, "cached_input": 0.175},
}


class MultiAgentResearchSystem(dspy.Module):
    def __init__(self, config: ModelConfig | None = None, *, log_dir: str = "logs/eval"):
        super().__init__()
        self.agent = Agent(config=config, work_dir="memory_eval/default", log_dir=log_dir)

    def forward(self, problem: str) -> dspy.Prediction:
        work_dir = create_isolated_workspace()
        agent = self.agent.deepcopy()
        agent.reset_workspace(work_dir)

        start_time = time.perf_counter()
        prediction = agent(problem)
        prediction.report = prediction.answer
        prediction.elapsed_seconds = time.perf_counter() - start_time
        prediction.tool_cost_usd = agent.search_tool.total_cost_usd + agent.fetch_tool.total_cost_usd
        return prediction


class BrowseCompEvaluator:
    QUERY_TIMEOUT_SECONDS = 600

    def __init__(self, args):
        self.args = args
        self.reflection_lm = None
        self.grader_lm = dspy.LM(
            model=GRADER_MODEL, temperature=1.0, max_tokens=GRADER_MAX_TOKENS, api_key=OPENROUTER_API_KEY
        )
        self.judge = dspy.ChainOfThought(BrowseCompJudge)

        if args.optimize:
            self.reflection_lm = dspy.LM(
                model=OPTIMIZER_MODEL, temperature=1.0, max_tokens=OPTIMIZER_MAX_TOKENS, api_key=OPENROUTER_API_KEY
            )

    def calculate_lm_cost(self, usage: dict) -> float:
        total_cost = 0.0
        for model_name, stats in usage.items():
            pricing = LM_PRICING[model_name]
            prompt_tokens = stats["prompt_tokens"]
            completion_tokens = stats["completion_tokens"]
            prompt_details = stats.get("prompt_tokens_details") or {}
            cached_tokens = prompt_details.get("cached_tokens", 0)
            non_cached_input = prompt_tokens - cached_tokens

            input_cost = (non_cached_input / 1_000_000) * pricing["input"]
            cached_cost = (cached_tokens / 1_000_000) * pricing.get("cached_input", pricing["input"])
            output_cost = (completion_tokens / 1_000_000) * pricing["output"]
            total_cost += input_cost + cached_cost + output_cost
        return total_cost

    def grade_prediction(self, example: dspy.Example, pred: dspy.Prediction) -> LLMJudgeAnswer:
        if pred.elapsed_seconds > self.QUERY_TIMEOUT_SECONDS:
            return LLMJudgeAnswer(
                is_correct=False,
                extracted_answer="TIMEOUT - NOT GRADED",
                reasoning=f"Exceeded {self.QUERY_TIMEOUT_SECONDS}s limit ({int(pred.elapsed_seconds)}s)",
            )
        try:
            with dspy.context(lm=self.grader_lm):
                result = self.judge(question=example.problem, report=pred.report, correct_answer=example.answer)
            return result.answer
        except Exception as e:
            logger.error(f"Grading error: {e}")
            return LLMJudgeAnswer(is_correct=False, extracted_answer="Error", reasoning="Grading failed")

    def metric(self, example, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
        llm_grading = self.grade_prediction(example, pred)
        accuracy = 1.0 if llm_grading.is_correct else 0.0

        usage = pred.get_lm_usage() or {}
        total_cost = self.calculate_lm_cost(usage) + pred.tool_cost_usd
        composite_score = accuracy / (1 + total_cost)

        pred.metrics = {
            "accuracy": accuracy,
            "composite_score": composite_score,
            "total_cost_usd": total_cost,
            "elapsed_seconds": pred.elapsed_seconds,
        }

        feedback = (
            f"Score: {composite_score:.4f} | Accuracy: {accuracy:.0f}/1 | Cost: ${total_cost:.4f}\n"
            f"Agent: {pred.answer}\nGrader: {llm_grading.extracted_answer}\n"
            f"Truth: {example.answer}\nReason: {llm_grading.reasoning}"
        )
        return ScoreWithFeedback(score=composite_score, feedback=feedback)

    def optimize_with_gepa(self, program: MultiAgentResearchSystem, train: list, val: list) -> MultiAgentResearchSystem:
        optimizer = GEPA(
            metric=self.metric,
            reflection_lm=self.reflection_lm,
            max_full_evals=self.args.max_full_evals,
            num_threads=self.args.num_threads,
            track_stats=True,
            track_best_outputs=True,
            enable_tool_optimization=True,
            component_selector="all",
        )
        return optimizer.compile(student=program, trainset=train, valset=val)

    def run(self, program: MultiAgentResearchSystem, examples: list) -> tuple:
        predictions_dict = {}

        def metric_with_capture(example, pred, trace=None):
            result = self.metric(example, pred, trace)
            predictions_dict[example.problem] = pred
            return result

        evaluator = dspy.Evaluate(
            devset=examples,
            metric=metric_with_capture,
            num_threads=self.args.num_threads,
            display_progress=True,
            display_table=5,
            max_errors=10,
        )
        result = evaluator(program)
        predictions = self._extract_predictions(predictions_dict, examples)
        return result, predictions

    def _extract_predictions(self, predictions_dict: dict, examples: list) -> list:
        predictions = []
        for i, ex in enumerate(examples):
            pred = predictions_dict.get(ex.problem)
            if pred is None:
                logger.warning(f"Missing prediction for example {i}")
                pred = dspy.Prediction(answer="ERROR", report="ERROR")
                pred.metrics = {"accuracy": 0.0, "elapsed_seconds": 0, "total_cost_usd": 0}
            predictions.append(pred)
        return predictions


def _parse_args():
    parser = create_model_cli_parser("Run BrowseComp evaluation")
    parser.add_argument("--offline", action="store_true", help="Use local corpus")
    parser.add_argument("--num-examples", type=int, default=8, help="Number of examples")
    parser.add_argument("--num-threads", type=int, default=5, help="Parallel threads")
    parser.add_argument("--optimize", action="store_true", help="Run GEPA optimization")
    parser.add_argument("--max-full-evals", type=int, default=4, help="GEPA iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", default="logs/eval", help="Log directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    print("🔍 BrowseComp Evaluation")
    print("=" * 50)
    print(f"⚖️  Grader: {GRADER_MODEL}")
    print("=" * 50)

    config = ModelConfig(lead=args.lead, sub=args.sub, offline=args.offline)
    print(f"🤖 Models: lead={config.lead}, sub={config.sub}")

    if dspy.settings.lm is None:
        dspy.configure(
            lm=dspy.LM(
                model=config.lead,
                temperature=config.temperature,
                max_tokens=config.lead_max_tokens,
                api_key=OPENROUTER_API_KEY,
            ),
            adapter=ChatAdapter(),
        )
    dspy.settings.configure(track_usage=True)

    evaluator = BrowseCompEvaluator(args)

    if args.offline:
        from browsecompplus import BrowseCompPlusDataset
        dataset = BrowseCompPlusDataset(
            "../BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
            num_examples=args.num_examples,
            seed=args.seed,
        )
        print("📚 Using BrowseComp-Plus (offline)")
    else:
        dataset = BrowseCompDataset(num_examples=args.num_examples, seed=args.seed)

    examples = dataset.load()
    print(f"📚 Loaded {len(examples)} examples")

    program = MultiAgentResearchSystem(config=config, log_dir=args.log_dir)

    if args.optimize:
        train, val = dataset.split(train_size=0.5)
        logger.info("GEPA optimization starting")
        print(f"\n🧬 GEPA Optimization (max_full_evals={args.max_full_evals})")
        print(f"🤖 Reflection: {OPTIMIZER_MODEL}")
        print(f"📊 Train: {len(train)}, Val: {len(val)}")

        optimized_program = evaluator.optimize_with_gepa(program, train, val)
        optimized_program.save("optimized_program.json")
        logger.info("Saved optimized program")

        start_cleanup_watchdog()

        results = optimized_program.detailed_results
        baseline_score = results.val_aggregate_scores[0]
        best_score = results.val_aggregate_scores[results.best_idx]
        baseline_accuracy = sum(1 for s in results.val_subscores[0].values() if s > 0) / len(val)
        best_accuracy = sum(1 for s in results.val_subscores[results.best_idx].values() if s > 0) / len(val)

        print("\n" + "=" * 50)
        print(f"📈 Baseline: accuracy={baseline_accuracy:.0%}, composite={baseline_score:.4f}")
        print(f"📈 Best:     accuracy={best_accuracy:.0%}, composite={best_score:.4f}")
        print(f"📊 Examples: {len(examples)} (train={len(train)}, val={len(val)})")
        print(f"🧬 Candidates: {len(results.candidates)}")
        print(f"🔄 Metric calls: {results.total_metric_calls}")

        baseline_program = results.candidates[0]
        best_program = results.candidates[results.best_idx]

        def get_all_tools(prog):
            tools = {}
            for name, tool in prog.agent.lead_agent.tools.items():
                tools[f"lead.{name}"] = tool
            for tool in prog.agent.subagent_tool._tools:
                tools[f"sub.{tool.name}"] = tool
            return tools

        print("\n📝 Predictor Instructions:")
        for (name, base_pred), (_, opt_pred) in zip(
            baseline_program.named_predictors(), best_program.named_predictors()
        ):
            base_inst = base_pred.signature.instructions
            opt_inst = opt_pred.signature.instructions
            marker = "✨" if base_inst != opt_inst else ""
            print(f"\n{marker} {name}:\n  BASE: {base_inst}\n  OPT:  {opt_inst}")
            logger.info(f"{name} BASE: {base_inst}")
            logger.info(f"{name} OPT: {opt_inst}")

        base_tools = get_all_tools(baseline_program)
        opt_tools = get_all_tools(best_program)
        print("\n🔧 Tool Descriptions:")
        for name, opt_tool in opt_tools.items():
            base_desc = base_tools[name].desc
            opt_desc = opt_tool.desc
            marker = "✨" if base_desc != opt_desc else ""
            print(f"\n{marker} {name}:\n  BASE: {base_desc}\n  OPT:  {opt_desc}")
            logger.info(f"{name} BASE: {base_desc}")
            logger.info(f"{name} OPT: {opt_desc}")
    else:
        print("🚀 Evaluating...")
        result, predictions = evaluator.run(program, examples)
        total_run_cost = sum(p.metrics.get("total_cost_usd", 0) for p in predictions)

        print("\n" + "=" * 50)
        print(f"📈 Score: {result.score:.4f}")
        print(f"💰 Cost: ${total_run_cost:.2f}")
        print(f"📊 Examples: {len(examples)}")
        logger.info(f"Cost: ${total_run_cost:.2f}")

    start_cleanup_watchdog()


if __name__ == "__main__":
    main()
