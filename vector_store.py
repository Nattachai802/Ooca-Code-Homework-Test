from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

# กำหนด Path ของโฟลเดอร์ data
DATA_DIR = Path(__file__).parent / "data"


class KnowledgeBaseStore:
    COLLECTION_NAME = "knowledge_base"

    def __init__(self) -> None:
        # 1. เชื่อมต่อกับ ChromaDB แบบ Persistent
        self._client = chromadb.PersistentClient(path="./chroma_db")
        
        # 2. สร้างหรือดึง Collection
        # ใช้ hnsw:space = cosine สำหรับการวัดความเหมือนของข้อความ
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._articles: dict[str, dict[str, Any]] = {}
        
        # 3. เริ่มกระบวนการนำเข้าข้อมูล
        self._ingest()

    def _ingest(self) -> None:
        """
        อ่านไฟล์ knowledge_base.json และบันทึกลง ChromaDB ถ้ายังไม่มีข้อมูล
        """
        # เช็คก่อนว่ามีข้อมูลอยู่แล้วหรือยัง เพื่อไม่ให้เสียเวลาทำซ้ำ
        if self._collection.count() > 0:
            logger.info("KB collection already populated, skipping ingestion.")
            return

        kb_path = DATA_DIR / "knowledge_base.json"
        with open(kb_path, encoding="utf-8") as f:
            articles = json.load(f)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        # วนลูปเตรียมข้อมูลแต่ละบทความ
        for article in articles:
            article_id = article["id"]
            self._articles[article_id] = article

            # รวมหัวข้อและเนื้อหาเข้าด้วยกันเพื่อใช้ในการค้นหา
            document_text = f"{article['topic']}\n{article['content']}"

            ids.append(article_id)
            documents.append(document_text)
            # เก็บ Metadata ภาษาอังกฤษ/ไทย ไว้ใช้กรองหรืออ้างอิงภายหลัง
            metadatas.append({
                "topic": article["topic"],
                "category": article.get("category", ""),
                "applies_to_plans": json.dumps(article.get("applies_to_plans", [])),
                "guideline_action": article.get("guideline", {}).get("action", ""),
                "guideline_conditions": article.get("guideline", {}).get("conditions", ""),
            })

        # บันทึกลง Database 
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Ingested %d KB articles into ChromaDB.", len(ids))

    def search(self, query: str, n_results: int = 3) -> list[dict[str, Any]]:
        """
        ค้นหาบทความที่เกี่ยวข้องที่สุดจากคำถาม (Semantic Search)
        params:
            query: คำถามหรือคีย์เวิร์ด
            n_results: จำนวนผลลัพธ์ที่ต้องการ (default 3)
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        matched_articles: list[dict[str, Any]] = []
        if results and results["ids"]:
            # วนลูปแกะผลลัพธ์จาก ChromaDB
            for i, article_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else None

                # ดึงเนื้อหาต้นฉบับจาก Memory หรือประกอบร่างใหม่
                article = self._articles.get(article_id, {})
                matched_articles.append({
                    "id": article_id,
                    "topic": metadata.get("topic", ""),
                    "content": article.get("content", ""),
                    "category": metadata.get("category", ""),
                    "applies_to_plans": json.loads(metadata.get("applies_to_plans", "[]")),
                    "guideline": {
                        "action": metadata.get("guideline_action", ""),
                        "conditions": metadata.get("guideline_conditions", ""),
                    },
                    # คำนวณความเหมือน (Cosine Similarity = 1 - Distance)
                    "relevance_score": round(1 - distance, 4) if distance is not None else None,
                })

        return matched_articles