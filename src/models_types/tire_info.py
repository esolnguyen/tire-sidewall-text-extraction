from typing import List, Optional
from dataclasses import dataclass, asdict


@dataclass
class TireInfo:
    """Data class for tire information extracted from sidewall"""
    manufacturer: str
    model: str
    size: str
    load_speed: str
    dot: str
    special_markings: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'TireInfo':
        """Create TireInfo instance from dictionary"""
        return cls(
            manufacturer=data.get("Manufacturer", "Not found"),
            model=data.get("Model", "Not found"),
            size=data.get("Size", "Not found"),
            load_speed=data.get("LoadSpeed", "Not found"),
            dot=data.get("DOT", "Not found"),
            special_markings=data.get("SpecialMarkings", [])
        )

    def __str__(self) -> str:
        """String representation of tire info"""
        return f"""Tire Information:
- Manufacturer: {self.manufacturer}
- Model: {self.model}
- Size: {self.size}
- Load/Speed: {self.load_speed}
- DOT: {self.dot}
- Special Markings: {', '.join(self.special_markings) if self.special_markings else 'None'}"""
