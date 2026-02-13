from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TicketAnalysis(BaseModel):
    """
    โมเดลสำหรับวิเคราะห์ข้อมูล Ticket เบื้องต้น
    """
    urgency: Literal["critical", "high", "medium", "low"] = Field(
        description="ระดับความเร่งด่วนของปัญหา"
    )
    sentiment: Literal["angry", "frustrated", "neutral", "positive"] = Field(
        description="อารมณ์ของลูกค้าที่ตรวจจับได้จากข้อความ"
    )
    issue_type: str = Field(
        description="หมวดหมู่ของปัญหา เช่น billing, outage, bug, feature_request"
    )
    product_area: str = Field(
        description="ส่วนของ product ที่ได้รับผลกระทบ"
    )
    language: str = Field(
        description="ภาษาที่ใช้ใน ticket (เช่น 'en', 'th')"
    )
    summary: str = Field(
        description="สรุปปัญหาของลูกค้าใน 1 บรรทัด"
    )


class SuggestedAction(BaseModel):
    """
    โมเดลสำหรับสิ่งที่ AI แนะนำให้ทำต่อ (Action)
    """
    action: Literal["auto_respond", "route_to_specialist", "escalate_to_human"] = Field(
        description="การกระทำที่แนะนำ"
    )
    suggested_reply: str = Field(
        description="ร่างข้อความตอบกลับลูกค้า (ต้องมีเสมอ ไม่ว่าจะเลือก action ไหน)"
    )
    reason: str = Field(
        description="เหตุผลประกอบการตัดสินใจ"
    )
    priority_score: int = Field(
        ge=1, le=10,
        description="คะแนนความสำคัญ 1-10 (1=ต่ำสุด, 10=สูงสุด)"
    )
    auto_response: Optional[str] = Field(
        default=None,
        description="ข้อความที่จะส่งตอบกลับอัตโนมัติ (กรณีเลือก action เป็น auto_respond)"
    )
    routing_department: Optional[str] = Field(
        default=None,
        description="แผนกที่จะส่งต่อ ticket ไปให้ (กรณีเลือก action เป็น route_to_specialist)"
    )
    escalation_notes: Optional[str] = Field(
        default=None,
        description="หมายเหตุสำหรับการส่งต่อให้เจ้าหน้าที่คนต่อไป (กรณีเลือก action เป็น escalate_to_human)"
    )


class TriageResult(BaseModel):
    """
    โมเดลผลลัพธ์สุดท้ายของการจัดลำดับความสำคัญ (Triage) ที่รวมการวิเคราะห์และการกระทำเข้าด้วยกัน
    """

    ticket_id: str = Field(description="รหัส Ticket")
    analysis: TicketAnalysis = Field(description="ผลการวิเคราะห์ Ticket")
    action: SuggestedAction = Field(description="Action ที่แนะนำ")
    customer_context: str = Field(
        description="สรุปข้อมูลลูกค้าที่เกี่ยวข้องกับการตัดสินใจนี้"
    )
    kb_articles_used: list[str] = Field(
        default_factory=list,
        description="รายชื่อ ID ของบทความใน Knowledge Base ที่ถูกใช้อ้างอิง"
    )

    @field_validator("customer_context", mode="before")
    @classmethod
    def coerce_customer_context(cls, v):
        """แปลง dict เป็น JSON string ถ้า LLM เผลอส่ง Object มาแทน String"""
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        return v