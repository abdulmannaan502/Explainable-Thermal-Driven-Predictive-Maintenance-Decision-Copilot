from pydantic import BaseModel, Field
from typing import Literal


class MaintenanceDecision(BaseModel):
    failure_mode: str = Field(..., description="Most likely failure mode")
    reasoning: str = Field(..., description="Concise engineering reasoning")
    recommended_action: str = Field(..., description="Safe next maintenance step")
    downtime_hours_min: int
    downtime_hours_max: int
    repair_cost_usd_min: int
    repair_cost_usd_max: int
    confidence: float = Field(..., ge=0.0, le=1.0)
