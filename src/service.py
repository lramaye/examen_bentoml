import numpy as np
import bentoml
from bentoml.io import JSON
from bentoml import Context
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta
import sys

# Ensure a single module instance regardless of being imported as 'service' or 'src.service'
_mod = sys.modules.get(__name__)
if _mod is not None:
    sys.modules.setdefault('service', _mod)
    sys.modules.setdefault('src.service', _mod)

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# specific users and credentials
USERS = {
    "user123": "password123",
    "user456": "password456"
}

# Middleware JWT
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # route predict
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

# model runner
# Use a mutable proxy to allow monkeypatching in tests while keeping the real Runner frozen
class RunnerProxy:
    def __init__(self, runner):
        self._runner = runner

    async def async_run(self, arr):
        return await self._runner.async_run(arr)

# Real BentoML runner (frozen attrs class)
_students_rf_runner = bentoml.sklearn.get("students_rf:latest").to_runner()
# Proxy exposed for tests to monkeypatch
students_rf_runner = RunnerProxy(_students_rf_runner)

# service BentoML
# Register the real runner with the Service
rf_service = bentoml.Service(name="admissions_prediction", runners=[_students_rf_runner])

# Ajout du middleware JWT
rf_service.add_asgi_middleware(JWTAuthMiddleware)

# Endpoint login
@rf_service.api(input=JSON(), output=JSON(), route="/login")
def login(credentials: dict, ctx: Context) -> dict:
    username = credentials.get("username")
    password = credentials.get("password")

    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        ctx.response.status_code = 401
        return {"detail": "Invalid credentials"}

# Endpoint predict
@rf_service.api(
    input=JSON(pydantic_model=InputModel),
    output=JSON(),
    route="/v1/models/rf_regressor/predict"
)
async def classify(input_data: InputModel, ctx: Context) -> dict:
    # Authentification
    auth = ctx.request.headers.get("authorization")
    user = None
    if auth:
        try:
            token = auth.split()[1]
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            user = payload.get("sub")
        except jwt.ExpiredSignatureError:
            user = None
        except jwt.InvalidTokenError:
            user = None

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
    result = await students_rf_runner.async_run(input_series.reshape(1, -1))

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
