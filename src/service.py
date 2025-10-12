import numpy as np
import bentoml
from bentoml.io.json import JSON
from bentoml import Context
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

# Middleware JWT
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # On protège uniquement la route prédiction
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

# Pydantic model for input validation
class InputModel(BaseModel):
    GRE_score: int
    TOEFL_score: int
    University_Rating: int
    SOP: float
    LOR: float
    CGPA: float
    Research: int

# Load the model runner
students_rf_runner = bentoml.sklearn.get("students_rf:latest").to_runner()

# Définition du service BentoML (nouvelle API)
@bentoml.service(name="rf_service")
class RFService:
    runners = [students_rf_runner]

    # Ajout du middleware JWT
    @classmethod
    def configure_service(cls):
        cls.add_asgi_middleware(JWTAuthMiddleware)

    # Endpoint pour login
    @bentoml.api(input=JSON(), output=JSON())
    def login(self, credentials: dict) -> dict:
        username = credentials.get("username")
        password = credentials.get("password")

        if username in USERS and USERS[username] == password:
            token = create_jwt_token(username)
            return {"token": token}
        else:
            return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

    # Endpoint de prédiction
    @bentoml.api(
        input=JSON(pydantic_model=InputModel),
        output=JSON(),
        route="/v1/models/rf_regressor/predict"
    )
    async def classify(self, input_data: InputModel, ctx: Context) -> dict:
        # Authentification via middleware
        request = ctx.request
        user = getattr(request.state, "user", None)

        # Conversion en array
        input_series = np.array([
            input_data.GRE_score,
            input_data.TOEFL_score,
            input_data.University_Rating,
            input_data.SOP,
            input_data.LOR,
            input_data.CGPA,
            input_data.Research
        ])

        # Prédiction
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
