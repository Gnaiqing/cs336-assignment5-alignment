from vllm import LLM, SamplingParams
from typing import Callable, List
import argparse
import json
from pathlib import Path
import pandas as pd
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn



def evaluate_vllm(
vllm_model: LLM,
reward_fn: Callable[[str, str], dict[str, float]],
prompts: List[str],
answers: List[str], 
eval_sampling_params: SamplingParams,
output_path: str
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    results_list = []
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        response = output.outputs[0].text.strip()
        reward = reward_fn(response, answer)
        output_dict = {
            "prompt": prompt,
            "response": response,
            "answer": answer,
            "reward": reward["reward"],
            "format_reward": reward["format_reward"],
            "answer_reward": reward["answer_reward"],
        }
        results_list.append(output_dict)
    
    results_df = pd.DataFrame(results_list)
    results_df.to_json(output_path, orient="records", lines=True)
    print(f"Evaluated {len(outputs)} prompts and saved results to {output_path}")
    print(f"Average reward: {results_df['reward'].mean()}")
    print(f"Average format reward: {results_df['format_reward'].mean()}")
    print(f"Average answer reward: {results_df['answer_reward'].mean()}")
    print(f"Number of correct answers: {results_df['reward'].sum()}")
    print(f"Number of answers that are formatted correctly but incorrect: {results_df['format_reward'].sum()} - {results_df['reward'].sum()}")
    print(f"Number of incorrectly formatted answers: {len(results_df) - results_df['format_reward'].sum()}")



if __name__ == "__main__":
    # Find project root (directory containing pyproject.toml)
    project_root = Path(__file__).resolve().parent.parent
    while project_root != project_root.parent and not (project_root / "pyproject.toml").exists():
        project_root = project_root.parent
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--prompt-path", type=str, default="cs336_alignment/prompts/r1_zero.txt")
    parser.add_argument("--input-path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--output-path", type=str, default="data/gsm8k/test_results.jsonl")
    args = parser.parse_args()

    # Resolve paths: if relative, make them relative to project root; if absolute, use as-is
    def resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return project_root / path
    
    prompt_path = resolve_path(args.prompt_path)
    input_path = resolve_path(args.input_path)
    output_path = resolve_path(args.output_path)

    model = LLM(model=args.model_name_or_path)
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    df = pd.read_json(input_path, lines=True)
    
    # Split answer column on "####" into reasoning and answer
    df[["reasoning", "answer"]] = df["answer"].str.split("####", n=1, expand=True)
    # Strip whitespace from the answer column
    df["answer"] = df["answer"].str.strip()
    df["reasoning"] = df["reasoning"].str.strip()

    prompts = []
    answers = []
    for _, row in df.iterrows():
        prompts.append(prompt_template.replace(r"{question}", row["question"]))
        answers.append(row["answer"])
    
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True
    evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        answers=answers,
        eval_sampling_params=sampling_params, 
        output_path=output_path
    )