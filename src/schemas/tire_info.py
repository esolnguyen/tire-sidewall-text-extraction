from typing import List, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class FieldWithBBox:
    """A single extracted field value with its source bounding boxes."""
    value: str
    source_bboxes: List[List[int]] = field(default_factory=list)


@dataclass
class TireInfo:
    """Data class for tire information extracted from sidewall"""
    manufacturer: FieldWithBBox
    model: FieldWithBBox
    size: FieldWithBBox
    load_speed: FieldWithBBox
    dot: FieldWithBBox
    special_markings: List[FieldWithBBox]

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return asdict(self)

    @classmethod
    def _parse_field(cls, data, key: str) -> FieldWithBBox:
        """Parse a field that may be a dict with value/source_bboxes or a plain string."""
        raw = data.get(key, "Not found")
        if isinstance(raw, dict):
            return FieldWithBBox(
                value=raw.get("value", "Not found"),
                source_bboxes=raw.get("source_bboxes", []),
            )
        return FieldWithBBox(value=str(raw))

    @classmethod
    def from_dict(cls, data: dict) -> 'TireInfo':
        """Create TireInfo instance from dictionary"""
        # Parse special markings (list of dicts or list of strings)
        raw_markings = data.get("SpecialMarkings", [])
        markings: List[FieldWithBBox] = []
        for item in raw_markings:
            if isinstance(item, dict):
                markings.append(FieldWithBBox(
                    value=item.get("value", ""),
                    source_bboxes=item.get("source_bboxes", []),
                ))
            else:
                markings.append(FieldWithBBox(value=str(item)))

        return cls(
            manufacturer=cls._parse_field(data, "Manufacturer"),
            model=cls._parse_field(data, "Model"),
            size=cls._parse_field(data, "Size"),
            load_speed=cls._parse_field(data, "LoadSpeed"),
            dot=cls._parse_field(data, "DOT"),
            special_markings=markings,
        )

    def __str__(self) -> str:
        """String representation of tire info"""
        markings_str = ', '.join(m.value for m in self.special_markings) if self.special_markings else 'None'
        return f"""Tire Information:
- Manufacturer: {self.manufacturer.value}
- Model: {self.model.value}
- Size: {self.size.value}
- Load/Speed: {self.load_speed.value}
- DOT: {self.dot.value}
- Special Markings: {markings_str}"""
