import json
import os
from typing import Tuple


def _parse_gsm8k_answer(ans: str) -> Tuple[str, str]:
    """
    GSM8K 'answer' format typically looks like:
      <step-by-step reasoning...>\n#### <final>
    Returns (reasoning, final_answer).
    """
    if not isinstance(ans, str):
        raise ValueError(f"Expected 'answer' to be a string, got {type(ans)}")

    if "####" in ans:
        reasoning, final = ans.split("####", 1)
        reasoning = reasoning.rstrip()
        final = final.strip()
    else:
        # Fallback: treat entire string as reasoning; try to grab last line/number as final
        reasoning = ans.rstrip()
        final = ans.strip().splitlines()[-1].strip() if ans.strip() else ""

    return reasoning, final


def convert_to_sft_data(prompt_path: str, input_path: str, output_path: str):
    """
    Convert a prompt template and GSM8K-style jsonl input into SFT jsonl output.

    Input (jsonl): each line like {"question": "...", "answer": "...#### 72"}
    Prompt template: a text file containing "{question}" placeholder, and typically ending with
      "Assistant: <think>"

    Output (jsonl): each line like
      {
        "prompt": "<prompt with question filled in>",
        "response": "<reasoning>\n</think>\n<answer>\n<final>\n</answer>"
      }
    """
    # Read prompt template
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_in, n_out = 0, 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            n_in += 1
            try:
                ex = json.loads(line)
                question = ex["question"]
                answer = ex["answer"]

                reasoning, final = _parse_gsm8k_answer(answer)

                prompt = prompt_template.replace("{question}", question)

                # IMPORTANT: prompt often already contains "Assistant: <think>"
                # so response should start with the reasoning content directly.
                response = (
                    f"{reasoning}\n"
                    f"</think>\n"
                    f"<answer>\n"
                    f"{final}\n"
                    f"</answer>"
                )

                out_obj = {"prompt": prompt, "response": response}
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                n_out += 1

            except Exception as e:
                raise RuntimeError(
                    f"Failed processing {input_path}:{lineno}. "
                    f"Line starts with: {line[:120]!r}. Error: {e}"
                ) from e

    return {"input_lines": n_in, "output_lines": n_out, "output_path": output_path}
