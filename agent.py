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

# กำหนด Path ของโฟลเดอร์ prompts
PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """
    ฟังก์ชันสำหรับอ่านprompt
    """
    prompt_path = PROMPTS_DIR / name
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


@dataclass
class ToolTrace:
    """
    เก็บข้อมูลประวัติการเรียกใช้เครื่องมือในแต่ละครั้ง
    """
    tool_name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class AgentResponse:
    """
    โครงสร้างข้อมูลผลลัพธ์ทั้งหมดจาก Agent รวมถึงประวัติการทำงาน
    """
    result: TriageResult
    tool_traces: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TriageAgent:
    MAX_TOOL_ROUNDS = 5  # จำนวนรอบสูงสุดที่อนุญาตให้ Agent เรียก Tool ได้

    def __init__(self) -> None:
        #สร้าง Client หลัก (OpenAI)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        #สร้าง Client สำรอง (Groq) สำหรับกรณี OpenAI ล่ม
        self.fallback_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.system_prompt = _load_prompt("system_prompt.txt")

    def process_ticket(self, ticket: dict[str, Any]) -> AgentResponse:
        """
        ฟังก์ชันหลักในการประมวลผล Ticket:
        1. รับข้อมูล Ticket
        2. วนลูปให้ AI คิดและเรียก Tool จนกว่าจะได้คำตอบ
        3. ส่งคืนผลลัพธ์ (AgentResponse)
        """
        user_message = self._format_ticket(ticket)
        tool_traces: list[ToolTrace] = []

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0

        # เริ่มต้นลูปการทำงานของ Agent
        for round_num in range(self.MAX_TOOL_ROUNDS):
            try:
                # พยายามเรียก OpenAI 
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except RateLimitError as e:
                # ถ้าเจอ Rate Limit ให้สลับไปใช้ Groq
                logger.warning(f"OpenAI Rate Limit hit: {e}. Switching to fallback provider (Groq).")
                try:
                    # ใช้ Llama-3.1-8b-instant บน Groq แทน
                    fallback_model = "llama-3.1-8b-instant"
                    response = self.fallback_client.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice="auto",
                        parallel_tool_calls=False,
                    )
                except Exception as fallback_error:
                    # ถ้า Fallback ก็ยังพัง ให้แจ้ง Error กลับไป
                    logger.error(f"Fallback provider failed: {fallback_error}")
                    raise RuntimeError(f"Rate limit reached and fallback failed: {e}") from fallback_error

            # เก็บสถิติ Token Usage
            if response.usage:
                total_tokens += response.usage.total_tokens
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens

            choice = response.choices[0]

            # กรณี AI ตอบจบ -> แปลงผลลัพธ์เป็น JSON แล้วจบการทำงาน
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

            # เก็บข้อความตอบกลับของ AI
            messages.append(choice.message)

            # กรณี AI สั่งเรียก Tool 
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # ค้นหาฟังก์ชันจริงจาก dict TOOL_DISPATCH
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

                # ส่งผลลัพธ์ของ Tool กลับไปให้ AI
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        raise RuntimeError(
            f"Agent exceeded {self.MAX_TOOL_ROUNDS} tool rounds for ticket {ticket['ticket_id']}"
        )

    def _format_ticket(self, ticket: dict[str, Any]) -> str:
        """
        แปลงข้อมูล Ticket ให้อยู่ในตูปแบบ Markdown Text เพื่อส่งเข้า Prompt
        """
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
        """
        แปลง String JSON จาก AI ให้กลายเป็น Object TriageResult (Validate ด้วย Pydantic)
        """
        if not content:
            raise ValueError(f"Empty response from LLM for ticket {ticket_id}")

        # ทำความสะอาด String (ตัด Markdown Syntax ```json ... ``` ออก)
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