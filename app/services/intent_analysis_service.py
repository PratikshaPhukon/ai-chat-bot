from enum import Enum
import re


class IntentType(Enum):
    COURSE_COUNT = "course_count"
    COURSE_LIST = "course_list"
    COURSE_RECOMMENDATION = "course_recommendation"
    COURSE_SEARCH = "course_search"
    COURSE_SUMMARY = "course_summary"
    GENERAL_INFO = "general_info"


class IntentAnalysisService:
    """
    Handles classification of user query into specific intents with confidence.
    """

    def analyze_intent(self, query: str) -> tuple[IntentType, float]:
        q = query.lower()

        patterns = {
            IntentType.COURSE_COUNT: [r"\bhow many courses\b", r"\bcount of courses\b"],
            IntentType.COURSE_LIST: [r"\blist.*courses\b", r"\bshow me courses\b"],
            IntentType.COURSE_RECOMMENDATION: [r"\brecommend\b", r"\bsuggest\b"],
            IntentType.COURSE_SEARCH: [r"\bfind courses\b", r"\bcourses related to\b"],
            IntentType.COURSE_SUMMARY: [
                r"\bsummarize\b",
                r"\boverview\b",
                r"\bwhat lessons\b",
            ],
        }

        # Simple scoring
        best_intent = IntentType.GENERAL_INFO
        best_score = 0.0

        for intent, pats in patterns.items():
            for pat in pats:
                if re.search(pat, q):
                    best_score = max(best_score, 0.9)  # strong match
                    best_intent = intent

        if best_score == 0.0:
            # Weak match fallback: check keywords
            if "course" in q:
                best_intent = IntentType.COURSE_SEARCH
                best_score = 0.6

        return best_intent, best_score

    def get_fallback_intent(self) -> IntentType:
        """Fallback when intent confidence is low."""
        return IntentType.GENERAL_INFO

    def filter_search_results_by_intent(
        self, search_results: list, intent: IntentType, course_id: str = None
    ):
        """Filter vector search results based on intent."""
        # Example: For course-specific summaries, filter only that course_id
        if course_id:
            return [
                r
                for r in search_results
                if r.get("payload", {}).get("course_id") == course_id
            ]
        return search_results

    def create_contextual_prompt(
        self, prefix: str, intent: IntentType, results: list
    ) -> tuple[str, dict]:
        """Generate final context for LLM."""
        context_parts = []
        for r in results:
            payload = r.get("payload", {})
            snippet = payload.get("text", "")
            title = payload.get("title", "")
            context_parts.append(f"{title}\n{snippet}")

        context = prefix + "\n".join(context_parts)
        metadata = {"results_used": len(results), "intent": intent.value}
        return context, metadata

    async def create_course_count_context_from_database(
        self, org_id: str, prefix: str
    ) -> tuple[str, dict]:
        """Direct DB access to count courses (placeholder)."""
        from app.services.resource_service import get_all_courses

        courses = await get_all_courses(org_id)
        count = len(courses)
        return f"{prefix}Total available courses: {count}", {"course_count": count}