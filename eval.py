"""
BrowseComp Evaluation Module

Evaluates the multi-agent research system on BrowseComp using DSPy's built-in evaluation framework.
"""

import time
import logging

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from agent import Agent
from models import BrowseCompJudge, LLMJudgeAnswer
from config import (
    ModelConfig,
    LM_PRICING,
    WEBSEARCH_COST_PER_CALL_USD,
    GRADER_MODEL,
    GRADER_MAX_TOKENS,
    OPTIMIZER_MODEL,
    OPTIMIZER_MAX_TOKENS,
    lm_kwargs_for,
)
from dataset import BrowseCompDataset
from utils import (
    create_model_cli_parser,
    start_cleanup_watchdog,
    create_isolated_workspace,
    cleanup_workspace,
)

logger = logging.getLogger(__name__)

class MultiAgentResearchSystem(dspy.Module):
    """
    DSPy program wrapper for Agent to make it compatible with dspy.Evaluate.

    Agent is created as module attribute so GEPA can discover and optimize tools.
    Uses deepcopy() for thread-safe parallel evaluation.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.agent = Agent(config=config, work_dir="memory_eval/default")

    def forward(self, problem: str) -> dspy.Prediction:
        work_dir = create_isolated_workspace()

        try:
            # Use DSPy's built-in deepcopy to preserve optimized tool descriptions
            agent = self.agent.deepcopy()
            agent.reset_workspace(work_dir)

            start = time.perf_counter()
            prediction = agent(problem)
            elapsed = time.perf_counter() - start
            
            prediction.report = prediction.answer
            prediction.elapsed_seconds = elapsed
            prediction.websearch_calls = agent.web_search_tool.call_count
            
            return prediction
        finally:
            cleanup_workspace(work_dir)

class BrowseCompEvaluator:
    """Encapsulates BrowseComp evaluation with proper state management."""
    
    def __init__(self, args):
        self.args = args
        self.reflection_lm = None  # Initialized lazily if optimization requested

        # Initialize grader LM once for all evaluations (major efficiency improvement)
        self.grader_lm = dspy.LM(
            model=GRADER_MODEL,
            temperature=1.0,  # Required for GPT-5 reasoning models
            max_tokens=GRADER_MAX_TOKENS,
            **lm_kwargs_for(GRADER_MODEL),
        )
        self.judge = dspy.ChainOfThought(BrowseCompJudge)

        # Initialize reflection LM for GEPA optimization if needed
        if args.optimize:
            self.reflection_lm = dspy.LM(
                model=OPTIMIZER_MODEL,
                temperature=1.0,  # Higher temp for creative prompt mutations
                max_tokens=OPTIMIZER_MAX_TOKENS,
                **lm_kwargs_for(OPTIMIZER_MODEL),
            )
    
    def calculate_lm_cost(self, usage: dict) -> float:
        """Calculate LM cost with accurate input/output/cached token pricing.
        
        Pricing in LM_PRICING is per 1M tokens (industry standard).
        Formula: (tokens / 1,000,000) * price_per_1M = cost in USD
        """
        total_cost = 0.0
        
        for model_name, stats in usage.items():
            pricing = LM_PRICING.get(model_name, {})
            if not pricing:
                logger.warning(f"No pricing configured for model: {model_name}")
                continue
            
            prompt_tokens = stats.get("prompt_tokens", 0)
            completion_tokens = stats.get("completion_tokens", 0)
            prompt_details = stats.get("prompt_tokens_details") or {}
            cached_tokens = prompt_details.get("cached_tokens", 0)
            non_cached_input = prompt_tokens - cached_tokens
            
            # Pricing is per 1M tokens, so divide by 1,000,000
            input_cost = (non_cached_input / 1_000_000) * pricing.get("input", 0.0)
            cached_cost = (cached_tokens / 1_000_000) * pricing.get("cached_input", pricing.get("input", 0.0))
            output_cost = (completion_tokens / 1_000_000) * pricing.get("output", 0.0)
            
            total_cost += input_cost + cached_cost + output_cost
        
        return total_cost
    
    def grade_prediction(self, example: dspy.Example, pred: dspy.Prediction) -> LLMJudgeAnswer | None:
        """Grade prediction using grader LM."""
        try:
            with dspy.context(lm=self.grader_lm):
                result = self.judge(
                    question=example.problem,
                    report=pred.report,
                    correct_answer=example.answer
                )
            return result.answer
        except Exception as e:
            logger.error(f"Grading error: {e}")
            return None

    def metric(self, example, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
        """Unified metric for dspy.Evaluate and GEPA."""
        grading = self.grade_prediction(example, pred)
        
        accuracy = 1.0 if grading and grading.is_correct else 0.0
        extracted = grading.extracted_answer if grading else "Error"
        reasoning = grading.reasoning if grading else "Grading failed"
        
        usage = pred.get_lm_usage() or {}
        total_tokens = sum(
            s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)
            for s in usage.values()
        )
        total_cost = self.calculate_lm_cost(usage) + pred.websearch_calls * WEBSEARCH_COST_PER_CALL_USD
        
        pred.metrics = {
            "accuracy": accuracy,
            "total_cost_usd": total_cost,
            "elapsed_seconds": pred.elapsed_seconds,
        }
        
        feedback = (
            f"Accuracy: {accuracy:.0f}/1\n"
            f"Expected: {example.answer}\n"
            f"Extracted: {extracted}\n"
            f"Reasoning: {reasoning}\n"
            f"Tokens: {total_tokens:,} | Cost: ${total_cost:.4f} | Time: {pred.elapsed_seconds:.1f}s"
        )
        
        return ScoreWithFeedback(score=accuracy, feedback=feedback)
    
    def optimize_with_gepa(self, program: MultiAgentResearchSystem, train: list) -> MultiAgentResearchSystem:
        """Run GEPA optimization on program."""
        optimizer = GEPA(
            metric=self.metric,
            reflection_lm=self.reflection_lm,
            max_full_evals=self.args.optimize_steps,  # Use explicit steps (can't combine with auto)
            num_threads=self.args.num_threads,
            track_stats=True,
            track_best_outputs=True,
            candidate_selection_strategy='pareto',
            use_merge=True,
            optimize_tool_descriptions=True,  # Optimize tool descriptions alongside signatures
        )
        
        return optimizer.compile(student=program, trainset=train)
    
    def run(self, program: MultiAgentResearchSystem, examples: list) -> tuple:
        """Run evaluation and return (result, predictions)."""
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
        """Extract predictions in correct order, handling missing ones."""
        predictions = []
        for i, ex in enumerate(examples):
            pred = predictions_dict.get(ex.problem)
            if pred is None:
                logger.warning(f"Missing prediction for example {i}, creating placeholder")
                pred = dspy.Prediction(answer="ERROR", report="ERROR")
                pred.metrics = {"accuracy": 0.0, "elapsed_seconds": 0, "total_cost_usd": 0}
            predictions.append(pred)
        return predictions

def _parse_args():
    parser = create_model_cli_parser("Run BrowseComp evaluation")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of dataset examples")
    parser.add_argument("--num-threads", type=int, default=2, help="Parallel evaluation threads")
    parser.add_argument("--optimize", action="store_true", help="Run GEPA optimization")
    parser.add_argument("--optimize-steps", type=int, default=10)
    parser.add_argument("--train-size", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    print("🔍 BrowseComp Evaluation")
    print("=" * 50)
    print(f"⚖️  Grader model: {GRADER_MODEL} (fixed for consistency)")
    print("=" * 50)

    # Build config from CLI args
    config = ModelConfig(lead=args.lead, sub=args.sub)
    print(f"🤖 Models: lead={config.lead}, sub={config.sub}")

    if dspy.settings.lm is None:
        dspy.configure(
            lm=dspy.LM(
                model=config.lead,
                temperature=config.temperature,
                max_tokens=config.lead_max_tokens,
                **lm_kwargs_for(config.lead),
            ),
            adapter=ChatAdapter(),
        )

    dspy.settings.configure(track_usage=True)

    # Initialize evaluator with grader and optimizer LMs
    evaluator = BrowseCompEvaluator(args)

    # Load dataset
    dataset = BrowseCompDataset(num_examples=args.num_examples)
    examples = dataset.load()
    print(f"📚 Loaded {len(examples)} examples")

    # Create agent program
    program = MultiAgentResearchSystem(config=config)

    # GEPA optimization if requested
    if args.optimize:
        print(f"\n🧬 GEPA Optimization ({args.optimize_steps} steps)")
        print(f"🤖 Using reflection model: {OPTIMIZER_MODEL}")
        train, test = dataset.split(train_size=args.train_size)
        print(f"📊 Split: {len(train)} train, {len(test)} test")
        
        program = evaluator.optimize_with_gepa(program, train)
        
        print("\n✨ Optimization complete!")
        print(f"📝 Optimized {len(list(program.named_predictors()))} predictor(s)")
        for name, pred in program.named_predictors():
            instr = getattr(pred.signature, 'instructions', '<no instructions>')
            print(f"  • {name}: {instr[:80]}..." if len(instr) > 80 else f"  • {name}: {instr}")
        
        examples = test  # Evaluate on test set

    # Run evaluation
    print("🚀 Evaluating...")
    result, predictions = evaluator.run(program, examples)

    # Workaround for DSPy/LiteLLM cleanup hang
    start_cleanup_watchdog()

    print("\n" + "=" * 50)
    print(f"📈 Score: {result.score:.4f}")
    print(f"📊 Examples: {len(examples)}")
    if args.optimize:
        print(f"🧬 Optimized with GEPA ({args.optimize_steps} steps)")


if __name__ == "__main__":
    main()
