# เอกสารอธิบายสถาปัตยกรรมระบบ (System Architecture Write-up)

## 1. การตัดสินใจด้านสถาปัตยกรรม (Architecture Decisions)
ระบบถูกออกแบบให้เป็น **AI-driven Triage Agent** ที่ทำงานอัตโนมัติในการวิเคราะห์และจัดหมวดหมู่ตั๋วแจ้งปัญหา (Support Ticket) โดยมีการตัดสินใจสำคัญดังนี้:

*   **Retrieval-Augmented Generation (RAG):**
    *   **การตัดสินใจ:** ใช้ `ChromaDB` เพื่อเก็บและค้นหา Knowledge Base ขององค์กร
    *   **เหตุผล:** LLM ทั่วไปมัก "มโน" (Hallucinate) ข้อมูลหรือจำนโยบายบริษัทไม่ได้แม่นยำ การใช้ RAG ช่วยให้ AI ตอบโดยอ้างอิงจากเอกสารจริงเสมอ ลดความผิดพลาดในการให้ข้อมูล
*   **Tool-Use & Function Calling:**
    *   **การตัดสินใจ:** บังคับให้ AI เรียกใช้ฟังก์ชัน `fetch_customer_data` และ `query_knowledge_base` ก่อนเสมอ
    *   **เหตุผล:** เพื่อให้ AI มีข้อมูลครบทั้ง "บริบทลูกค้า" (เช่น เป็นลูกค้า Enterprise หรือไม่) และ "วิธีแก้ปัญหา" ก่อนที่จะตัดสินใจ Triage (Look before you leap pattern)
*   **Structured Output (Pydantic):**
    *   **การตัดสินใจ:** บังคับ Output ให้อยู่ในรูป JSON ที่ถูกต้องแม่นยำด้วย Pydantic
    *   **เหตุผล:** ระบบ Triage ต้องส่งต่อข้อมูลให้ระบบอื่น (เช่น ระบบ Routing อัตโนมัติ) การได้ Output เป็น Text ธรรมดาจาก AI จะจัดการยากและพังง่าย
*   **Resilient Fallback Mechanism:**
    *   **การตัดสินใจ:** ระบบจะสลับไปใช้ **Groq (Llama-3)** ทันทีเมื่อ **OpenAI** ชน Rate Limit หรือล่ม
    *   **เหตุผล:** ระบบ Support ต้องทำงานตลอดเวลา (High Availability) การมี Model สำรองที่ทำงานได้เหมือนกันแต่คนละ Provider ช่วยลด Downtime ได้อย่างมีประสิทธิภาพ

## 2. สิ่งที่อาจผิดพลาดและการรับมือ (Risk Analysis & Mitigation)

| ความเสี่ยง (Risk) | วิธีการรับมือ (Mitigation) |
| :--- | :--- |
| **API Rate Limiting / Outages** <br> (OpenAI ล่มหรือชนโควต้า) | **Fallback System:** ระบบถูกเขียนให้ดักจับ Error (`RateLimitError`) และสลับไปใช้ Client สำรอง (Groq) ทันทีโดยอัตโนมัติ ทำให้ User ไม่รู้สึกถึงความผิดปกติ |
| **Hallucination** <br> (AI มั่วข้อมูลหรือส่งผิดแผนก) | **Confidence Score & Reason:** ใน Output จะมีคะแนนความมั่นใจ (`priority_score`) และเหตุผลประกอบ ถ้าคะแนนต่ำกว่าเกณฑ์ สามารถตั้งให้ส่งคนตรวจสอบ (Human Review) ก่อนได้ |
| **Tool Execution Failures** <br> (เรียก Database ไม่เจอ / ค้นหาไม่พบ) | **Error Handling in Prompt:** ถ้า Tool พัง ระบบจะส่ง Error Message กลับไปให้ AI รับรู้ เพื่อให้ AI พยายามหาทางออกอื่น หรือแจ้ง User ว่า "ขณะนี้ไม่สามารถดึงข้อมูลได้" แทนที่จะเงียบหายไป |

## 3. การประเมินผลใน Production (Evaluation Strategy)

เพื่อให้มั่นใจว่า Agent ทำงานได้จริง ผมจะใช้วิธีวัดผลดังนี้:

1.  **Golden Set Testing (วัดความแม่นยำ):**
    *   เตรียมตั๋วเก่าจำนวน 100 ใบที่เจ้าหน้าที่เก่งๆ เคยตอบไว้แล้ว (Ground Truth)
    *   ให้ Agent ลองทำใหม่ แล้วเทียบผลลัพธ์ว่าตรงกันกี่ % (Target: >90% Accuracy)
2.  **Escalation Rate Monitoring (วัดประสิทธิภาพ):**
    *   ติดตามดูว่า Agent ส่งต่อให้คน (`escalate_to_human`) บ่อยแค่ไหน
    *   ถ้าส่งให้คนเยอะเกินไป (>40%) แสดงว่า AI ไม่กล้าตัดสินใจ หรือ Knowledge Base ไม่ครอบคลุม
3.  **Fallback Activation Rate (วัดเสถียรภาพ):**
    *   คอยดู Log ว่าระบบสลับไปใช้ Groq บ่อยแค่ไหน ถ้าบ่อยเกินไปอาจต้องพิจารณาอัปเกรด Plan ของ OpenAI
4.  **Customer Sentiment Shift (วัดความพึงพอใจ):**
    *   ในกรณีที่ AI ตอบกลับเอง (Auto-respond) ให้ดูว่าข้อความตอบกลับของลูกค้าหลังจากนั้นมีอารมณ์ดีขึ้นหรือแย่ลง
