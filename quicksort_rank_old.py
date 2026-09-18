import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
RANKING_TEMPERATURE = 0.0


def write_jsonl(filename: str, data: Iterable[Dict[str, Any]]) -> None:
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def load_price_config(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--price-config must be a JSON object keyed by model name.")
    return data


def resolve_pricing(
    model: str,
    price_config: Optional[Dict[str, Dict[str, Any]]] = None,
    input_price_per_1m: Optional[float] = None,
    output_price_per_1m: Optional[float] = None,
    currency: Optional[str] = None,
) -> Dict[str, Any]:
    price_config = price_config or {}
    configured = price_config.get(model, {})
    if configured and not isinstance(configured, dict):
        raise ValueError(f"Price entry for model '{model}' must be an object.")

    config_input = _first_present(
        configured,
        ["input_price_per_1m", "input", "prompt_price_per_1m", "prompt", "input_price"],
    )
    config_output = _first_present(
        configured,
        ["output_price_per_1m", "output", "completion_price_per_1m", "completion", "output_price"],
    )
    config_cache_read = _first_present(
        configured,
        ["cache_read_price_per_1m", "cached_input_price_per_1m", "cache_price_per_1m", "cache_read", "cache"],
    )
    input_price = input_price_per_1m if input_price_per_1m is not None else config_input
    output_price = output_price_per_1m if output_price_per_1m is not None else config_output
    resolved_currency = currency or configured.get("currency", "USD")

    if input_price is None or output_price is None:
        return {
            "configured": False,
            "model": model,
            "currency": resolved_currency,
            "input_price_per_1m": input_price,
            "output_price_per_1m": output_price,
            "cache_read_price_per_1m": config_cache_read,
            "source": "missing",
        }

    source = "cli" if input_price_per_1m is not None or output_price_per_1m is not None else "price_config"
    return {
        "configured": True,
        "model": model,
        "currency": resolved_currency,
        "input_price_per_1m": float(input_price),
        "output_price_per_1m": float(output_price),
        "cache_read_price_per_1m": float(config_cache_read) if config_cache_read is not None else None,
        "source": source,
    }


def add_money_cost(cost: Dict[str, Any], pricing: Dict[str, Any]) -> Dict[str, Any]:
    cost["pricing"] = pricing
    if not pricing.get("configured"):
        cost["estimated_ranking_cost"] = None
        return cost

    cached_input_tokens = int(cost.get("cached_input_tokens", 0) or 0)
    if cached_input_tokens <= 0:
        cached_input_tokens = int(cost.get("comparator_stats", {}).get("cached_input_tokens", 0) or 0)
    cache_price = pricing.get("cache_read_price_per_1m")
    if cached_input_tokens > 0 and cache_price is not None:
        regular_input_tokens = max(int(cost["input_tokens"]) - cached_input_tokens, 0)
        input_cost = regular_input_tokens / 1_000_000 * pricing["input_price_per_1m"]
        cache_read_cost = cached_input_tokens / 1_000_000 * cache_price
        formula = (
            "(input_tokens-cached_input_tokens)/1e6*input_price_per_1m + "
            "cached_input_tokens/1e6*cache_read_price_per_1m + "
            "output_tokens/1e6*output_price_per_1m"
        )
    else:
        regular_input_tokens = int(cost["input_tokens"])
        input_cost = regular_input_tokens / 1_000_000 * pricing["input_price_per_1m"]
        cache_read_cost = 0.0
        formula = "input_tokens/1e6*input_price_per_1m + output_tokens/1e6*output_price_per_1m"
    output_cost = cost["output_tokens"] / 1_000_000 * pricing["output_price_per_1m"]
    cost["estimated_ranking_cost"] = {
        "input_cost": input_cost,
        "cache_read_cost": cache_read_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "currency": pricing["currency"],
        "formula": formula,
    }
    return cost


@dataclass
class ComparatorStats:
    logical_comparisons: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    uncertain_results: int = 0
    failed_results: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    display_originals: int = 0
    display_swaps: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class QuestionItem:
    idx: int
    text: str


class PairwiseCheckpoint:
    def __init__(self, path: Optional[str] = None, presentation_order: Optional[str] = None):
        self.path = path
        self.presentation_order = presentation_order
        self.records: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._lock = Lock()
        if self.path:
            self.load()

    @staticmethod
    def key_for(left_idx: int, right_idx: int) -> Tuple[int, int]:
        return tuple(sorted((left_idx, right_idx)))

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def flip_result(result: int) -> int:
        return (3 - result) if result in (1, 2) else 0

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[Checkpoint Warning] Skip malformed line {line_no}: {self.path}")
                    continue
                record_order = record.get("presentation_order", "fixed")
                if self.presentation_order and record_order != self.presentation_order:
                    continue
                key = self.key_for(int(record["left_idx"]), int(record["right_idx"]))
                self.records[key] = record

    def get(self, left: QuestionItem, right: QuestionItem) -> Optional[Tuple[int, int, int]]:
        key = self.key_for(left.idx, right.idx)
        with self._lock:
            record = self.records.get(key)
        if record is None:
            return None
        expected = {
            int(record["left_idx"]): record.get("left_hash"),
            int(record["right_idx"]): record.get("right_hash"),
        }
        if expected.get(left.idx) != self.text_hash(left.text):
            return None
        if expected.get(right.idx) != self.text_hash(right.text):
            return None
        result = int(record["result"])
        if result not in (1, 2):
            return None
        if left.idx != int(record["left_idx"]):
            result = self.flip_result(result)
        return result, 0, 0

    def append(
        self,
        left: QuestionItem,
        right: QuestionItem,
        result: int,
        prompt_tokens: int,
        completion_tokens: int,
        source: str,
        cached_prompt_tokens: int = 0,
        presentation_order: Optional[str] = None,
        display_order: Optional[str] = None,
        display_swapped: Optional[bool] = None,
        display_votes: Optional[List[int]] = None,
        raw_outputs: Optional[List[str]] = None,
        votes: Optional[List[int]] = None,
        finish_reasons: Optional[List[Optional[str]]] = None,
    ) -> None:
        if not self.path:
            return
        key = self.key_for(left.idx, right.idx)
        record = {
            "left_idx": left.idx,
            "right_idx": right.idx,
            "left_hash": self.text_hash(left.text),
            "right_hash": self.text_hash(right.text),
            "result": result,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "source": source,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if presentation_order is not None:
            record["presentation_order"] = presentation_order
        if display_order is not None:
            record["display_order"] = display_order
        if display_swapped is not None:
            record["display_swapped"] = display_swapped
        if display_votes is not None:
            record["display_votes"] = display_votes
        if raw_outputs is not None:
            record["raw_outputs"] = raw_outputs
        if votes is not None:
            record["votes"] = votes
        if finish_reasons is not None:
            record["finish_reasons"] = finish_reasons
        with self._lock:
            existing = self.records.get(key)
            if existing is not None and int(existing.get("result", 0)) in (1, 2):
                return
            if existing is not None and result not in (1, 2):
                return
            self.records[key] = record
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    def token_totals(self) -> Dict[str, int]:
        with self._lock:
            records = list(self.records.values())
        input_tokens = sum(int(record.get("prompt_tokens", 0)) for record in records)
        output_tokens = sum(int(record.get("completion_tokens", 0)) for record in records)
        cached_input_tokens = sum(int(record.get("cached_prompt_tokens", 0)) for record in records)
        return {
            "records": len(records),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


class DifficultyComparator:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "gpt-4o",
        max_retries: int = 5,
        timeout: int = 120,
        max_output_tokens: int = 16,
        reasoning_effort: Optional[str] = "minimal",
        presentation_order: str = "balanced",
        order_seed: int = 42,
        checkpoint: Optional[PairwiseCheckpoint] = None,
    ):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for LLM QuickSort. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc
        self._openai = openai
        self._client = None
        if hasattr(openai, "OpenAI"):
            client_kwargs = {"api_key": api_key}
            if api_base:
                client_kwargs["base_url"] = api_base
            self._client = openai.OpenAI(**client_kwargs)
        else:
            openai.api_base = api_base
            openai.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.presentation_order = presentation_order
        self.order_seed = order_seed
        self.comparison_cache: Dict[Tuple[str, str], int] = {}
        self.checkpoint = checkpoint
        self.stats = ComparatorStats()
        self._lock = Lock()

    def _bump(self, **kwargs: int) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self.stats, key, getattr(self.stats, key) + value)

    @staticmethod
    def _parse_vote(content: str) -> int:
        text = content.strip().upper()
        if re.search(r"\bQ1\b", text):
            return 1
        if re.search(r"\bQ2\b", text):
            return 2
        return 0

    @staticmethod
    def _majority_vote(votes: Sequence[int]) -> int:
        counts = Counter(v for v in votes if v in (1, 2))
        if not counts:
            return 0
        q1_votes = counts.get(1, 0)
        q2_votes = counts.get(2, 0)
        if q1_votes == q2_votes:
            return 0
        return 1 if q1_votes > q2_votes else 2

    @staticmethod
    def _flip_result(result: int) -> int:
        return (3 - result) if result in (1, 2) else 0

    def _should_swap_display(self, q1_key: str, q2_key: str) -> bool:
        if self.presentation_order == "fixed":
            return False
        digest = hashlib.sha1(f"{self.order_seed}:{q1_key}:{q2_key}".encode("utf-8")).hexdigest()
        return int(digest[-1], 16) % 2 == 1

    @staticmethod
    def _get_usage_value(usage: Any, key: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return int(usage.get(key, 0))
        return int(getattr(usage, key, 0) or 0)

    @staticmethod
    def _get_cached_prompt_tokens(usage: Any) -> int:
        if usage is None:
            return 0

        def get_value(container: Any, key: str) -> Any:
            if container is None:
                return None
            if isinstance(container, dict):
                return container.get(key)
            return getattr(container, key, None)

        for key in ("cached_tokens", "cache_read_tokens", "cached_input_tokens", "prompt_cache_hit_tokens"):
            value = get_value(usage, key)
            if value is not None:
                return int(value or 0)

        for details_key in ("prompt_tokens_details", "prompt_token_details", "input_tokens_details"):
            details = get_value(usage, details_key)
            for key in ("cached_tokens", "cache_read_tokens", "cached_input_tokens"):
                value = get_value(details, key)
                if value is not None:
                    return int(value or 0)
        return 0

    @staticmethod
    def _get_response_choices(response: Any) -> Sequence[Any]:
        if isinstance(response, dict):
            return response["choices"]
        return response.choices

    @staticmethod
    def _get_choice_content(choice: Any) -> str:
        if isinstance(choice, dict):
            message = choice.get("message", {})
            if isinstance(message, dict):
                return message.get("content") or ""
            return getattr(message, "content", "") or ""
        return getattr(choice.message, "content", "") or ""

    @staticmethod
    def _get_choice_finish_reason(choice: Any) -> Optional[str]:
        if isinstance(choice, dict):
            return choice.get("finish_reason")
        return getattr(choice, "finish_reason", None)

    @staticmethod
    def _get_response_usage(response: Any) -> Any:
        if isinstance(response, dict):
            return response.get("usage", {})
        return getattr(response, "usage", None)

    @staticmethod
    def _looks_like_param_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in [
                "unsupported",
                "unrecognized",
                "unknown parameter",
                "invalid parameter",
                "max_completion_tokens",
                "max_tokens",
                "reasoning_effort",
            ]
        )

    def _create_chat_completion(self, messages: List[Dict[str, str]], n: int) -> Any:
        if self._client is not None:
            base_kwargs = {
                "model": self.model,
                "messages": messages,
                "n": n,
                "temperature": RANKING_TEMPERATURE,
                "timeout": self.timeout,
            }
            variants: List[Dict[str, Any]] = []
            if self.reasoning_effort:
                variants.append(
                    {
                        **base_kwargs,
                        "max_completion_tokens": self.max_output_tokens,
                        "reasoning_effort": self.reasoning_effort,
                    }
                )
            variants.append({**base_kwargs, "max_completion_tokens": self.max_output_tokens})
            variants.append({**base_kwargs, "max_tokens": self.max_output_tokens})

            last_exc: Optional[Exception] = None
            for kwargs in variants:
                try:
                    return self._client.chat.completions.create(**kwargs)
                except Exception as exc:
                    last_exc = exc
                    if not self._looks_like_param_error(exc):
                        raise
            raise last_exc if last_exc else RuntimeError("Failed to create chat completion.")
        return self._openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            n=n,
            temperature=RANKING_TEMPERATURE,
            max_tokens=self.max_output_tokens,
            request_timeout=self.timeout,
        )

    def compare_two(self, q1_text: str, q2_text: str, n: int = 1) -> Tuple[int, int, int]:
        """
        Return:
        1 -> q1 is more difficult
        2 -> q2 is more difficult
        0 -> uncertain / roughly equal
        """
        self._bump(logical_comparisons=1)
        result, prompt_tokens, completion_tokens, *_ = self._compare_texts(
            q1_text,
            q2_text,
            n=n,
            q1_key=self.text_key(q1_text),
            q2_key=self.text_key(q2_text),
        )
        return result, prompt_tokens, completion_tokens

    def compare_items(self, q1: QuestionItem, q2: QuestionItem, n: int = 1, source: str = "partition") -> Tuple[int, int, int]:
        """
        Compare two indexed questions and checkpoint the finished pair.
        """
        self._bump(logical_comparisons=1)
        if self.checkpoint:
            restored = self.checkpoint.get(q1, q2)
            if restored is not None:
                self._bump(cache_hits=1)
                return restored

        (
            result,
            prompt_tokens,
            completion_tokens,
            cached_prompt_tokens,
            raw_outputs,
            votes,
            finish_reasons,
            display_order,
            display_swapped,
            display_votes,
        ) = self._compare_texts(q1.text, q2.text, n=n)
        if self.checkpoint and (result in (1, 2) or prompt_tokens > 0 or completion_tokens > 0):
            self.checkpoint.append(
                left=q1,
                right=q2,
                result=result,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                source=source,
                cached_prompt_tokens=cached_prompt_tokens,
                presentation_order=self.presentation_order,
                display_order=display_order,
                display_swapped=display_swapped,
                display_votes=display_votes,
                raw_outputs=raw_outputs,
                votes=votes,
                finish_reasons=finish_reasons,
            )
        return result, prompt_tokens, completion_tokens

    @staticmethod
    def text_key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _compare_texts(
        self,
        q1_text: str,
        q2_text: str,
        n: int = 1,
        q1_key: Optional[str] = None,
        q2_key: Optional[str] = None,
    ) -> Tuple[int, int, int, int, List[str], List[int], List[Optional[str]], str, bool, List[int]]:
        cache_key = tuple(sorted((q1_text, q2_text)))
        cached_res = self.comparison_cache.get(cache_key)
        if cached_res is not None:
            self._bump(cache_hits=1)
            if q1_text == cache_key[0]:
                return cached_res, 0, 0, 0, [], [], [], "cache", False, []
            return self._flip_result(cached_res), 0, 0, 0, [], [], [], "cache", False, []

        q1_key = q1_key or self.text_key(q1_text)
        q2_key = q2_key or self.text_key(q2_text)
        display_swapped = self._should_swap_display(q1_key, q2_key)
        display_q1 = q2_text if display_swapped else q1_text
        display_q2 = q1_text if display_swapped else q2_text
        display_order = "q2_q1" if display_swapped else "q1_q2"

        instruction = (
            "You are a judge for question difficulty.\n"
            "Decide which question is MORE DIFFICULT for a language model to answer correctly.\n"
            "Consider required knowledge, reasoning depth, ambiguity, and distractor strength.\n"
            "Respond with ONLY one token: Q1 or Q2.\n"
        )
        query = f"Q1:\n{display_q1}\n\nQ2:\n{display_q2}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": query},
        ]

        for attempt in range(self.max_retries):
            try:
                response = self._create_chat_completion(messages, n=n)
                usage = self._get_response_usage(response)
                prompt_tokens = self._get_usage_value(usage, "prompt_tokens")
                completion_tokens = self._get_usage_value(usage, "completion_tokens")
                cached_prompt_tokens = self._get_cached_prompt_tokens(usage)
                choices = self._get_response_choices(response)
                raw_outputs = [self._get_choice_content(choice) for choice in choices]
                display_votes = [self._parse_vote(content) for content in raw_outputs]
                votes = [self._flip_result(vote) if display_swapped else vote for vote in display_votes]
                finish_reasons = [self._get_choice_finish_reason(choice) for choice in choices]
                result = self._majority_vote(votes)
                stored_result = result if q1_text == cache_key[0] else self._flip_result(result)
                self.comparison_cache[cache_key] = stored_result

                increments = {
                    "api_calls": 1,
                    "input_tokens": prompt_tokens,
                    "cached_input_tokens": cached_prompt_tokens,
                    "output_tokens": completion_tokens,
                    "display_swaps" if display_swapped else "display_originals": 1,
                }
                if result == 0:
                    increments["uncertain_results"] = 1
                self._bump(**increments)
                return (
                    result,
                    prompt_tokens,
                    completion_tokens,
                    cached_prompt_tokens,
                    raw_outputs,
                    votes,
                    finish_reasons,
                    display_order,
                    display_swapped,
                    display_votes,
                )
            except Exception as exc:
                print(f"[Compare Error] Attempt {attempt + 1}/{self.max_retries}: {exc}")
                time.sleep(10)

        self._bump(failed_results=1, uncertain_results=1)
        return 0, 0, 0, 0, [], [], [], display_order, display_swapped, []


def choose_pivot(
    items: List[QuestionItem],
    comparator: DifficultyComparator,
    compare_n: int,
    rng: random.Random,
    pivot_strategy: str = "random",
    median_k: int = 3,
) -> Tuple[QuestionItem, int, int]:
    strategy = pivot_strategy.lower()
    if strategy in ("random", "rand") or len(items) <= 2:
        return rng.choice(items), 0, 0

    if strategy in ("median3", "median-of-3"):
        median_k = 3
    elif strategy in ("median5", "median-of-5"):
        median_k = 5
    elif strategy not in ("median", "median-k"):
        raise ValueError(f"Unknown pivot strategy: {pivot_strategy}")

    k = min(max(3, median_k), len(items))
    if k % 2 == 0:
        k -= 1
    candidates = rng.sample(items, k)
    scores = {item.idx: 0 for item in candidates}
    total_prompt_tok, total_completion_tok = 0, 0

    for i in range(k):
        for j in range(i + 1, k):
            left = candidates[i]
            right = candidates[j]
            result, p_tok, c_tok = comparator.compare_items(left, right, n=compare_n, source="pivot_selection")
            total_prompt_tok += p_tok
            total_completion_tok += c_tok
            if result == 1:
                scores[left.idx] += 1
                scores[right.idx] -= 1
            elif result == 2:
                scores[left.idx] -= 1
                scores[right.idx] += 1

    ranked = sorted(candidates, key=lambda item: (scores[item.idx], item.idx))
    return ranked[len(ranked) // 2], total_prompt_tok, total_completion_tok


def llm_quicksort(
    items: List[QuestionItem],
    comparator: DifficultyComparator,
    compare_n: int = 1,
    max_workers: int = 8,
    rng: Optional[random.Random] = None,
    pivot_strategy: str = "random",
    median_k: int = 3,
    show_progress: bool = True,
) -> Tuple[List[QuestionItem], int, int]:
    """
    Sort questions from easy to hard using pairwise LLM comparisons.
    """
    if len(items) <= 1:
        return items, 0, 0

    rng = rng or random.Random()
    total_prompt_tok, total_completion_tok = 0, 0
    pivot, p_tok, c_tok = choose_pivot(
        items=items,
        comparator=comparator,
        compare_n=compare_n,
        rng=rng,
        pivot_strategy=pivot_strategy,
        median_k=median_k,
    )
    total_prompt_tok += p_tok
    total_completion_tok += c_tok

    easier: List[QuestionItem] = []
    harder: List[QuestionItem] = []
    same: List[QuestionItem] = []
    others = [item for item in items if item.idx != pivot.idx]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(comparator.compare_items, item, pivot, n=compare_n, source="partition"): item
            for item in others
        }
        iterator = as_completed(future_to_item)
        if show_progress:
            iterator = tqdm(iterator, total=len(others), desc="QuickSort Compare")
        for future in iterator:
            result, p_tok, c_tok = future.result()
            total_prompt_tok += p_tok
            total_completion_tok += c_tok
            item = future_to_item[future]

            if result == 1:
                harder.append(item)
            elif result == 2:
                easier.append(item)
            else:
                same.append(item)

    s_harder, p1, c1 = llm_quicksort(
        harder,
        comparator,
        compare_n=compare_n,
        max_workers=max_workers,
        rng=rng,
        pivot_strategy=pivot_strategy,
        median_k=median_k,
        show_progress=show_progress,
    )
    s_easier, p2, c2 = llm_quicksort(
        easier,
        comparator,
        compare_n=compare_n,
        max_workers=max_workers,
        rng=rng,
        pivot_strategy=pivot_strategy,
        median_k=median_k,
        show_progress=show_progress,
    )
    total_prompt_tok += p1 + p2
    total_completion_tok += c1 + c2

    return s_easier + [pivot] + same + s_harder, total_prompt_tok, total_completion_tok


def dsc_r_quicksort_ranking(
    questions: List[str],
    api_base: str,
    api_key: str,
    model: str = "gpt-4o",
    compare_n: int = 1,
    seed: int = 42,
    max_workers: int = 8,
    pivot_strategy: str = "random",
    median_k: int = 3,
    max_output_tokens: int = 16,
    reasoning_effort: Optional[str] = "minimal",
    presentation_order: str = "balanced",
    show_progress: bool = True,
    checkpoint_path: Optional[str] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    rng = random.Random(seed)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(questions) + 1000))
    indexed_questions = [QuestionItem(idx=idx, text=text) for idx, text in enumerate(questions)]
    checkpoint = PairwiseCheckpoint(checkpoint_path, presentation_order=presentation_order) if checkpoint_path else None
    comparator = DifficultyComparator(
        api_base=api_base,
        api_key=api_key,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        presentation_order=presentation_order,
        order_seed=seed,
        checkpoint=checkpoint,
    )
    sorted_items, p_tok, c_tok = llm_quicksort(
        indexed_questions,
        comparator,
        compare_n=compare_n,
        max_workers=max_workers,
        rng=rng,
        pivot_strategy=pivot_strategy,
        median_k=median_k,
        show_progress=show_progress,
    )
    sorted_indices = [item.idx for item in sorted_items]
    stats = asdict(comparator.stats)
    logical_comparisons = int(stats.get("logical_comparisons", 0) or 0)
    uncertain_results = int(stats.get("uncertain_results", 0) or 0)
    uncertain_rate = uncertain_results / logical_comparisons if logical_comparisons else 0.0
    checkpoint_totals = checkpoint.token_totals() if checkpoint else None
    cumulative_input_tokens = checkpoint_totals["input_tokens"] if checkpoint_totals else p_tok
    cumulative_cached_input_tokens = checkpoint_totals["cached_input_tokens"] if checkpoint_totals else int(stats.get("cached_input_tokens", 0) or 0)
    cumulative_output_tokens = checkpoint_totals["output_tokens"] if checkpoint_totals else c_tok
    ranking_cost: Dict[str, Any] = {
        "input_tokens": cumulative_input_tokens,
        "cached_input_tokens": cumulative_cached_input_tokens,
        "output_tokens": cumulative_output_tokens,
        "total_tokens": cumulative_input_tokens + cumulative_output_tokens,
        "current_run_input_tokens": p_tok,
        "current_run_cached_input_tokens": int(stats.get("cached_input_tokens", 0) or 0),
        "current_run_output_tokens": c_tok,
        "current_run_total_tokens": p_tok + c_tok,
        "comparator_stats": stats,
        "uncertain_rate": uncertain_rate,
        "valid_comparison_rate": 1.0 - uncertain_rate,
        "num_questions": len(questions),
        "logical_comparisons": logical_comparisons,
        "compare_n": compare_n,
        "pivot_strategy": pivot_strategy,
        "median_k": median_k,
        "seed": seed,
        "max_workers": max_workers,
        "model": model,
        "max_output_tokens": max_output_tokens,
        "reasoning_effort": reasoning_effort,
        "presentation_order": presentation_order,
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint_totals,
    }
    return sorted_indices, ranking_cost


def extract_answer_from_raw(item: Dict[str, Any], dataset: str) -> Any:
    if dataset == "GSM8K":
        return item["answer"].split("####")[-1].strip()
    if dataset.startswith("MMLU-Pro"):
        return item.get("answer", item.get("output"))
    return item.get("output", item.get("answer"))


def format_mmlu_pro_question(item: Dict[str, Any]) -> str:
    question = item["question"].strip()
    options = item.get("options") or []
    option_lines = []
    for i, option in enumerate(options):
        label = LETTERS[i] if i < len(LETTERS) else str(i)
        option_lines.append(f"{label}. {option}")
    return question + "\n\nOptions:\n" + "\n".join(option_lines)


def load_dataset(input_path: str, dataset: str, limit: Optional[int] = None) -> Dict[str, List[Any]]:
    questions: List[str] = []
    answers: List[Any] = []
    subjects: List[str] = []
    hard_levels: List[Any] = []
    categories: List[str] = []
    sources: List[str] = []

    problem_field_name = "input" if dataset != "MATH" else "problem"
    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(questions) >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            if dataset.startswith("MMLU-Pro"):
                questions.append(item["input"] if "input" in item else format_mmlu_pro_question(item))
                answers.append(extract_answer_from_raw(item, dataset))
                categories.append(item.get("category", "unknown"))
                sources.append(item.get("src", "unknown"))
            else:
                if problem_field_name not in item:
                    raise KeyError(f"Missing field '{problem_field_name}' at line {line_no}: {input_path}")
                questions.append(item[problem_field_name])
                answers.append(extract_answer_from_raw(item, dataset))
                if dataset == "MATH":
                    subjects.append(item.get("subject", "unknown"))
                    hard_levels.append(item.get("hard_level", item.get("level", 0)))

    return {
        "questions": questions,
        "answers": answers,
        "subjects": subjects,
        "hard_levels": hard_levels,
        "categories": categories,
        "sources": sources,
    }


def build_eval_output(dataset: str, loaded: Dict[str, List[Any]], eval_list: List[float]) -> List[Dict[str, Any]]:
    questions = loaded["questions"]
    answers = loaded["answers"]

    if dataset == "MATH":
        grouped: Dict[str, Dict[str, List[Any]]] = {}
        for i, question in enumerate(questions):
            subject = loaded["subjects"][i]
            grouped.setdefault(subject, {"problem": [], "answer": [], "eval": [], "hard_level": []})
            grouped[subject]["problem"].append(question)
            grouped[subject]["answer"].append(answers[i])
            grouped[subject]["eval"].append(eval_list[i])
            grouped[subject]["hard_level"].append(loaded["hard_levels"][i])
        return [
            {
                "type": subject,
                "questions": info["problem"],
                "answer": info["answer"],
                "eval": info["eval"],
                "hard_level": info["hard_level"],
            }
            for subject, info in grouped.items()
        ]

    output: Dict[str, Any] = {
        "questions": questions,
        "answer": answers,
        "eval": eval_list,
        "completion": [[] for _ in questions],
    }
    if dataset.startswith("MMLU-Pro"):
        output["category"] = loaded["categories"]
        output["src"] = loaded["sources"]
    return [output]


def default_input_path(dataset: str) -> str:
    input_map = {
        "GSM8K": "dataset/GSM8K.jsonl",
        "coin_flip": "dataset/coin_flip.jsonl",
        "last_letter": "dataset/last_letter.jsonl",
        "strategy": "dataset/strategy.jsonl",
        "common": "dataset/common.jsonl",
        "MATH": "dataset/MATH.jsonl",
        "MMLU-Pro": "data/mmlu_pro/test.jsonl",
        "MMLU-Pro-2000": "dataset/MMLU-Pro_2000_seed42.jsonl",
    }
    if dataset not in input_map:
        raise KeyError(f"Unknown dataset: {dataset}. Please provide --input-path.")
    return input_map[dataset]


def default_run_base(args: argparse.Namespace) -> str:
    limit_part = f"limit{args.limit}" if args.limit is not None else "full"
    reasoning_part = f"reasoning{safe_name(str(args.reasoning_effort))}" if args.reasoning_effort else "reasoningnone"
    order_part = f"order{safe_name(str(args.presentation_order))}"
    return "_".join(
        [
            safe_name(args.dataset),
            safe_name(args.model),
            safe_name(args.pivot_strategy),
            f"n{args.compare_n}",
            f"maxout{args.max_output_tokens}",
            reasoning_part,
            order_part,
            f"seed{args.seed}",
            limit_part,
        ]
    )


def resolve_output_paths(args: argparse.Namespace, base: str) -> Dict[str, str]:
    if args.flat_output:
        os.makedirs(args.output_dir, exist_ok=True)
        return {
            "run_dir": args.output_dir,
            "converted_eval": os.path.join(args.output_dir, f"{base}_converted_eval.jsonl"),
            "cost": os.path.join(args.output_dir, f"{base}_quicksort_cost.json"),
            "rank": os.path.join(args.output_dir, f"{base}_quicksort_rank.jsonl"),
            "checkpoint": os.path.join(args.output_dir, f"{base}_pairwise_checkpoint.jsonl"),
        }

    run_dir = os.path.join(args.output_dir, base)
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "converted_eval": os.path.join(run_dir, "converted_eval.jsonl"),
        "cost": os.path.join(run_dir, "quicksort_cost.json"),
        "rank": os.path.join(run_dir, "quicksort_rank.jsonl"),
        "checkpoint": os.path.join(run_dir, "pairwise_checkpoint.jsonl"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank questions by LLM QuickSort difficulty comparison.")
    parser.add_argument("--dataset", default="GSM8K", help="Dataset name, e.g. GSM8K, MATH, MMLU-Pro.")
    parser.add_argument("--input-path", default=None, help="Input JSONL path. Defaults depend on --dataset.")
    parser.add_argument("--output-dir", default=os.path.join("result", "r3sc_old"), help="Directory for ranking outputs.")
    parser.add_argument("--model", default="gpt-4o", help="Comparator model.")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE", "https://aihubmix.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--compare-n", type=int, default=3, help="Number of comparator samples per pair.")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Maximum completion tokens for each Q1/Q2 comparator output.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="minimal",
        choices=["minimal", "low", "medium", "high", "none"],
        help="Reasoning effort hint for reasoning models. Use 'none' to omit the parameter.",
    )
    parser.add_argument(
        "--presentation-order",
        default="balanced",
        choices=["balanced", "fixed"],
        help="Question presentation order. 'balanced' deterministically swaps Q1/Q2 for about half the pairs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--pivot-strategy",
        default="random",
        choices=["random", "median3", "median5", "median", "median-k"],
        help="Pivot strategy for ablation.",
    )
    parser.add_argument("--median-k", type=int, default=3, help="Used when --pivot-strategy median/median-k.")
    parser.add_argument("--limit", type=int, default=None, help="Optional question limit for debugging/ablation.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Pairwise comparison checkpoint JSONL. Defaults to output-dir/run-base_pairwise_checkpoint.jsonl.",
    )
    parser.add_argument(
        "--disable-checkpoint",
        action="store_true",
        help="Disable pairwise checkpointing/resume.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Use the old flat result directory layout instead of creating one folder per run.",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=None,
        help="Comparator model input-token price per 1M tokens. Optional; used for money cost.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help="Comparator model output-token price per 1M tokens. Optional; used for money cost.",
    )
    parser.add_argument(
        "--price-currency",
        default=None,
        help="Currency label for money cost, e.g. USD or CNY. Defaults to config currency or USD.",
    )
    parser.add_argument(
        "--price-config",
        default=None,
        help="Optional JSON file keyed by model name with input/output prices per 1M tokens.",
    )
    parser.add_argument(
        "--max-uncertain-rate",
        type=float,
        default=0.5,
        help="Abort before writing rank files if more than this fraction of pairwise decisions are uncertain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable or --api-key is required.")
    if not 0.0 <= args.max_uncertain_rate <= 1.0:
        raise ValueError("--max-uncertain-rate must be between 0 and 1.")

    input_path = args.input_path or default_input_path(args.dataset)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset file for {args.dataset} not found at {input_path}")

    loaded = load_dataset(input_path, args.dataset, limit=args.limit)
    questions = loaded["questions"]
    print(f"Loaded {len(questions)} questions for dataset: {args.dataset}")

    base = default_run_base(args)
    output_paths = resolve_output_paths(args, base)
    output_filename = output_paths["converted_eval"]
    cost_filename = output_paths["cost"]
    rank_filename = output_paths["rank"]
    print(f"Using run output directory: {output_paths['run_dir']}")
    checkpoint_path = None
    if not args.disable_checkpoint:
        checkpoint_path = args.checkpoint_path or output_paths["checkpoint"]
        print(f"Using pairwise checkpoint: {checkpoint_path}")
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    sorted_idx, cost = dsc_r_quicksort_ranking(
        questions=questions,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        compare_n=args.compare_n,
        seed=args.seed,
        max_workers=args.max_workers,
        pivot_strategy=args.pivot_strategy,
        median_k=args.median_k,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=reasoning_effort,
        presentation_order=args.presentation_order,
        show_progress=not args.no_progress,
        checkpoint_path=checkpoint_path,
    )
    cost["run_base"] = base
    cost["run_dir"] = output_paths["run_dir"]
    pricing = resolve_pricing(
        model=args.model,
        price_config=load_price_config(args.price_config),
        input_price_per_1m=args.input_price_per_1m,
        output_price_per_1m=args.output_price_per_1m,
        currency=args.price_currency,
    )
    cost = add_money_cost(cost, pricing)

    print("\n--- Ranking Complete ---")
    print("Total ranking cost:", cost)

    stats = cost.get("comparator_stats", {})
    api_calls = int(stats.get("api_calls", 0) or 0)
    uncertain_rate = float(cost.get("uncertain_rate", 0.0) or 0.0)
    if api_calls > 0 and uncertain_rate > args.max_uncertain_rate:
        with open(cost_filename, "w", encoding="utf-8") as f:
            json.dump(cost, f, ensure_ascii=False, indent=4)
        print(f"Saved ranking cost to: {cost_filename}")
        raise RuntimeError(
            "Comparator returned too many uncertain pairwise decisions "
            f"({uncertain_rate:.1%} > {args.max_uncertain_rate:.1%}). "
            "Increase --max-output-tokens, try --reasoning-effort none, or use a non-reasoning comparator model."
        )

    eval_list = [0.0] * len(questions)
    num_questions = len(questions)
    for rank, idx in enumerate(sorted_idx):
        eval_list[idx] = rank / max(1, num_questions - 1)

    output_data = build_eval_output(args.dataset, loaded, eval_list)

    write_jsonl(output_filename, output_data)
    print(f"Saved converted output to: {output_filename}")

    with open(cost_filename, "w", encoding="utf-8") as f:
        json.dump(cost, f, ensure_ascii=False, indent=4)
    print(f"Saved ranking cost to: {cost_filename}")

    rank_rows = [
        {"rank": rank, "index": idx, "eval": eval_list[idx]}
        for rank, idx in enumerate(sorted_idx)
    ]
    write_jsonl(rank_filename, rank_rows)
    print(f"Saved sorted ranks to: {rank_filename}")
    print("\nProcess complete.")


if __name__ == "__main__":
    main()
