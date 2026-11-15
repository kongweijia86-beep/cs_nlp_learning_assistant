from collections import defaultdict

CHAPTER_PAGE_RANGES = {
    "Chapter 1": (5, 21),
    "Chapter 2": (22, 63),
    "Chapter 3": (64, 112),
    "Chapter 4": (113, 154),
    "Chapter 5": (155, 203),
    "Chapter 6": (204, 243),
    "Chapter 7": (244, 302),
    "Appendix A": (303, 351),
    "Appendix B": (352, 361),
    "Appendix C": (362, 372),
    "Appendix D": (373, 388),
    "Appendix E": (389, 404),
}

SKIP_ROLES = {"front_matter", "references", "chapter_summary"}

APPENDIX_KEYWORDS = {
    "appendix", "computation graph", "computational graph",
    "autograd", "backpropagation", "lora", "parameter-efficient",
    "ddp", "distributed", "gpu", "warmup", "cosine decay",
    "gradient clipping", "exercise",
}

def _normalize_chapter(ch):
    """把章标准化"""
    if not ch: return None
    ch = ch.strip()
    if ch in CHAPTER_PAGE_RANGES: return ch
    if ch.isdigit():
        candidate = f"Chapter {ch}"
        if candidate in CHAPTER_PAGE_RANGES: return candidate
    ch_lower = ch.lower()
    for key in CHAPTER_PAGE_RANGES:
        if key.lower() == ch_lower: return key
    if len(ch) == 1 and ch.isalpha():
        candidate = f"Appendix {ch.upper()}"
        if candidate in CHAPTER_PAGE_RANGES: return candidate
    if ch_lower.startswith("appendix"):
        letter = ch_lower.replace("appendix", "").strip()
        if len(letter) == 1 and letter.isalpha():
            candidate = f"Appendix {letter.upper()}"
            if candidate in CHAPTER_PAGE_RANGES: return candidate
    return None

def _page_to_chapter(page_num):
    """页对应章"""
    for ch, (start, end) in CHAPTER_PAGE_RANGES.items():
        if start <= page_num <= end:
            return ch
    return None

def get_allowed_pages(chapter, tolerance=3):
    """返回章节对应的，前后扩展了容错页数的页码集合，是一个允许的页码范围"""
    if not chapter:
        return None
    def _range_for(ch):
        normalized = _normalize_chapter(ch)
        if normalized is None:
            return None
        start, end = CHAPTER_PAGE_RANGES[normalized]
        return set(range(max(1, start - tolerance), end + tolerance + 1))
    if "&" in chapter:
        pages = set()
        for ch in chapter.split("&"):
            r = _range_for(ch)
            if r:
                pages.update(r)
        return pages if pages else None
    return _range_for(chapter)

def should_search_appendix(question_text, category=None):
    """查询问题是否在appendix的关键词里"""
    q_lower = (question_text or "").lower()
    return any(kw in q_lower for kw in APPENDIX_KEYWORDS)

def classify_page_role(page_num, page_text=None):
    """对页所在的部分来分类"""
    if page_num <= 4: return "front_matter"
    if 352 <= page_num <= 361: return "references"
    if 362 <= page_num <= 372: return "exercise_solution"
    if 303 <= page_num <= 351: return "appendix"
    if page_num >= 373: return "appendix"
    if page_num in {21, 154}: return "chapter_summary"
    return "main_content"

def diversify_by_chapter(results, max_per_chapter=2, top_k=20):
    """按章节对结果限流，每个章节最多保留max_per_chapter条结果，防止某个章节的结果霸占整个列表，让不同章节的结果都能出现"""
    if not results:
        return results
    chapter_count = defaultdict(int)
    diversified = []
    for r in results:
        ch = r.get("metadata_chapter")
        if not ch:
            ch = _page_to_chapter(r['page'])
        ch = ch or "Unknown"
        if chapter_count[ch] < max_per_chapter:
            diversified.append(r)
            chapter_count[ch] += 1
    return diversified


def filter_retrieval_results(results, question_item, tolerance=2):
    """过滤查询的结果"""
    if not results:
        return results
    chapter = question_item.get("chapter")
    source = question_item.get("source", "")
    question_text = question_item.get("question", "")
    category = question_item.get("category")

    is_old_book = "Build_a_Large" in source or "From_Scratch" in source
    if is_old_book:
        allowed_pages = get_allowed_pages(chapter, tolerance=tolerance)
        if allowed_pages:
            results = [r for r in results if r["page"] in allowed_pages]
        if not should_search_appendix(question_text, category):
            results = [r for r in results if classify_page_role(r["page"]) not in SKIP_ROLES]
    else:
        if chapter:
            results = [r for r in results if r.get("metadata_chapter") == chapter]
    return results