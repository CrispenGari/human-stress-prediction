class Label:
    def __init__(self, label: str, labelId: int, confidence: float):
        self.label = label
        self.labelId = labelId
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"[HSP Preciction: {self.label}]"

    def __str__(self) -> str:
        return f"[HSP Preciction: {self.label}]"

    def to_json(self):
        return {
            'label':  self.label,
            'labelId':  self.labelId,
            'confidence':  self.confidence,
        }

class Prediction:
    def __init__(self, text: str, label: Label, _type: Label):
        self.text = text
        self.label = label
        self._type = _type

    def __repr__(self) -> str:
        return f"[HSP Preciction: {self.label.label} - {self._type.label}]"

    def __str__(self) -> str:
        return f"[HSP Preciction: {self.label.label} - {self._type.label}]"

    def to_json(self):
        return {
            'text':  self.text,
            '_type':  self._type.to_json(),
            'label':  self.label.to_json(),
        }

