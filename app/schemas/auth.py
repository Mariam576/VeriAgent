"""Request / response shapes for sign-in."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)
    name: str | None = Field(default=None, max_length=120)
    signup_code: str | None = Field(default=None, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class GoogleSignInRequest(BaseModel):
    credential: str = Field(min_length=20, max_length=4096, description="The ID token from Google's sign-in button.")
    signup_code: str | None = Field(default=None, max_length=200)


class ChangePasswordRequest(BaseModel):
    current_password: str | None = Field(default=None, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class DeleteAccountRequest(BaseModel):
    confirm: str = Field(description='Type DELETE to confirm.')


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    email: str | None = None
    email_verified: bool = False
    created_at: datetime
    has_password: bool = False
    has_google: bool = False


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserOut


class AuthConfig(BaseModel):
    """What the sign-in page needs to know. Contains no secrets."""

    auth_mode: str
    registration_open: bool
    signup_code_required: bool
    google_client_id: str | None = None
    min_password_chars: int
