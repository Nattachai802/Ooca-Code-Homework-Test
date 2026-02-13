from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from models import TriageResult
from tools import TOOL_DISPATCH, TOOL_SCHEMAS

load_dotenv()

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    prompt_path = PROMPTS_DIR / name
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


@dataclass
class ToolTrace:
    """Record of a single tool invocation."""
    tool_name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class AgentResponse:
    """Full agent response including reasoning trace."""
    result: TriageResult
    tool_traces: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TriageAgent:
    MAX_TOOL_ROUNDS = 5

    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.system_prompt = _load_prompt("system_prompt.txt")

    def process_ticket(self, ticket: dict[str, Any]) -> AgentResponse:
        user_message = self._format_ticket(ticket)
        tool_traces: list[ToolTrace] = []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0

        for round_num in range(self.MAX_TOOL_ROUNDS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except RateLimitError as e:
                retry_after = getattr(e, "retry_after", None)
                if retry_after:
                    msg = f"Rate limit reached. Please try again in {retry_after} seconds."
                else:
                    import re
                    match = re.search(r"try again in (\d+\.?\d*s)", str(e))
                    wait_str = match.group(1) if match else "a moment"
                    msg = f"Rate limit reached. Please try again in {wait_str}."
                raise RuntimeError(msg)

            if response.usage:
                total_tokens += response.usage.total_tokens
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens

            choice = response.choices[0]

            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                triage_result = self._parse_response(
                    choice.message.content, ticket["ticket_id"]
                )
                return AgentResponse(
                    result=triage_result,
                    tool_traces=tool_traces,
                    rounds=round_num + 1,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                tool_fn = TOOL_DISPATCH.get(tool_name)
                if tool_fn is None:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        tool_result = tool_fn(**tool_args)
                    except Exception as e:
                        logger.error("Tool %s failed: %s", tool_name, e)
                        tool_result = {"error": str(e)}

                tool_traces.append(ToolTrace(
                    tool_name=tool_name,
                    arguments=tool_args,
                    result=tool_result,
                ))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        raise RuntimeError(
            f"Agent exceeded {self.MAX_TOOL_ROUNDS} tool rounds for ticket {ticket['ticket_id']}"
        )

    def _format_ticket(self, ticket: dict[str, Any]) -> str:
        lines = [
            f"## Support Ticket: {ticket['ticket_id']}",
            f"**Customer Email:** {ticket['customer_email']}",
            f"**Subject:** {ticket.get('subject', 'N/A')}",
            "",
            "### Messages (oldest to newest):",
        ]

        for msg in ticket.get("messages", []):
            timestamp = msg.get("timestamp", "unknown")
            content = msg.get("content", "")
            lines.append(f"\n[{timestamp}]\n{content}")

        return "\n".join(lines)

    def _parse_response(self, content: str | None, ticket_id: str) -> TriageResult:
        if not content:
            raise ValueError(f"Empty response from LLM for ticket {ticket_id}")

        json_str = content.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("\n", 1)[1] if "\n" in json_str else json_str[3:]
            json_str = json_str.rsplit("```", 1)[0]
            json_str = json_str.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            logger.error("Raw response:\n%s", content)
            raise ValueError(f"LLM returned invalid JSON for ticket {ticket_id}: {e}")

        return TriageResult.model_validate(data)