import re
from typing import Any


def normalize_keyword(keyword: str) -> str:
    return keyword.strip().lower()


def sort_text_values(values: list[str]) -> list[str]:
    return sorted(values, key=lambda item: item.casefold())


def normalize_scalar_field(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_labeled_list_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        normalized_values = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(sort_text_values(normalized_values))
    return str(value).strip()


def normalize_unlabeled_list_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        normalized_values = [str(item).strip() for item in value if str(item).strip()]
        return " ".join(sort_text_values(normalized_values))
    return str(value).strip()


def cleanse_text(value: str) -> str:
    cleaned = str(value)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def cleansed_normalize_labeled_list_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        cleaned_values: list[str] = []
        seen: set[str] = set()
        for item in value:
            cleaned = cleanse_text(str(item))
            dedupe_key = cleaned.casefold()
            if cleaned and dedupe_key not in seen:
                seen.add(dedupe_key)
                cleaned_values.append(cleaned)
        return ",".join(sort_text_values(cleaned_values))
    return cleanse_text(str(value))


def cleansed_normalize_scalar_field(value: Any) -> str:
    if value is None:
        return ""
    return cleanse_text(str(value))
