from fastapi import FastAPI, Request, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import os
import joblib
import numpy as np

app = FastAPI()

# Load the model at startup
model = joblib.load('models/model.pkl')

# Security scheme for bearer token
class TokenBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(TokenBearer, self).__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials:
        credentials = await super(TokenBearer, self).__call__(request)
        if credentials:
            if not self.verify_token(credentials.credentials):
                raise HTTPException(status_code=403, detail="Invalid or expired token.")
            return credentials.credentials
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")
    
    def verify_token(self, token: str) -> bool:
        bearer_token = os.getenv("BEARER_TOKEN")
        return token == bearer_token

token_auth_scheme = TokenBearer()

# Define the expected input data model
class InputData(BaseModel):
    mob_inet: float
    fix_inet: float
    fix_ctv: float
    fix_ictv: float
    voice_mob: float
    voice_fix: float
    sms: float
    csd: float
    iot: float
    mms: float
    roaming: float
    mg: float
    mn: float
    mts: float
    conc: float
    fix_op: float
    vsr_roam: float
    national_roam: float
    mn_roam: float
    voice_ap: float
    voice_fee: float
    period_service: float
    one_time_service: float
    dop_service: float
    content: float
    services_service: float
    keo_sale: float
    discount: float
    only_inbound: float
    sms_a2p: float
    sms_gross: float
    skoring: float
    other_service: float
    voice_mail: float
    geo: float
    ep_for_number: float
    ep_for_line: float
    one_time_fee_for_number: float
    equipment_rent: float
    add_package: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(
    data: InputData,
    token: str = Depends(token_auth_scheme),
    threshold: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(None, ge=1)
):
    """
    Predict the top group_ids with probabilities above a threshold.

    - **threshold**: Return only predictions with probability greater than this value (default is 0.0).
    - **limit**: Maximum number of predictions to return.
    """
    try:
        # Convert input data to the format expected by the model
        input_features = np.array([[
            data.mob_inet, data.fix_inet, data.fix_ctv, data.fix_ictv,
            data.voice_mob, data.voice_fix, data.sms, data.csd, data.iot,
            data.mms, data.roaming, data.mg, data.mn, data.mts, data.conc,
            data.fix_op, data.vsr_roam, data.national_roam, data.mn_roam,
            data.voice_ap, data.voice_fee, data.period_service,
            data.one_time_service, data.dop_service, data.content,
            data.services_service, data.keo_sale, data.discount,
            data.only_inbound, data.sms_a2p, data.sms_gross, data.skoring,
            data.other_service, data.voice_mail, data.geo, data.ep_for_number,
            data.ep_for_line, data.one_time_fee_for_number, data.equipment_rent,
            data.add_package
        ]])

        # Get the probabilities for each class
        probabilities = model.predict_proba(input_features)[0]

        # Get the class labels
        class_labels = model.classes_

        # Combine class labels with their probabilities
        class_probabilities = list(zip(class_labels, probabilities))

        # Filter based on the threshold
        filtered_class_probabilities = [
            (group_id, prob) for group_id, prob in class_probabilities if prob > threshold
        ]

        # Sort the classes based on probability in descending order
        sorted_class_probabilities = sorted(filtered_class_probabilities, key=lambda x: x[1], reverse=True)

        # Apply the limit if provided
        if limit is not None:
            sorted_class_probabilities = sorted_class_probabilities[:limit]

        # Prepare the response
        predictions = [{"group_id": int(group_id), "probability": float(prob)} for group_id, prob in sorted_class_probabilities]

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
