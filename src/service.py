import numpy as np
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "user123": "password123",
    "user456": "password456"
}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/models/rf_regressor/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})

            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})

            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response

# Pydantic model to validate input data
class InputModel(BaseModel):
    GRE_score: int
    TOEFL_score : int
    University_Rating: int
    SOP: float
    LOR: float
    CGPA: float
    Research: int

# Get the model from the Model Store
students_rf_runner = bentoml.sklearn.get("students_rf:latest").to_runner()

# Create the service for BentoML 1.4.x (requires runners)
#rf_service = bentoml.Service("rf_service", runners=[students_rf_runner])

# Add the JWTAuthMiddleware to the service
#rf_service.add_asgi_middleware(JWTAuthMiddleware)

# Définition du service avec la nouvelle API
@bentoml.service(name="rf_service")
class RFService:
    runners = [students_rf_runner]

    # Endpoint pour le login
    @bentoml.api(input=JSON(), output=JSON())
    def login(self, credentials: dict) -> dict:
        username = credentials.get("username")
        password = credentials.get("password")

        if username in USERS and USERS[username] == password:
            token = create_jwt_token(username)
            return {"token": token}
        else:
            # Starlette est compatible avec BentoML pour les réponses HTTP
            return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    # Endpoint de prédiction
    @bentoml.api(
        input=JSON(pydantic_model="InputModel"),
        output=JSON(),
        route="/v1/models/rf_regressor/predict"
    )
    async def classify(self, input_data, ctx: bentoml.Context) -> dict:
        # Authentification optionnelle
        request = ctx.request
        user = getattr(request.state, "user", None)

        # Convertir les données d'entrée
        input_series = np.array([
            input_data.GRE_score,
            input_data.TOEFL_score,
            input_data.University_Rating,
            input_data.SOP,
            input_data.LOR,
            input_data.CGPA,
            input_data.Research
        ])

        # Prédiction asynchrone avec le runner
        result = await students_rf_runner.predict.async_run(input_series.reshape(1, -1))

        return {
            "prediction": result.tolist(),
            "user": user
        }

# Génération de token JWT
def create_jwt_token(user_id: str) -> str:
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {"sub": user_id, "exp": expiration}
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token