from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import bcrypt
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    audits = relationship("AuditHistory", back_populates="owner")
    
    def verify_password(self, password: str):
        return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))

    @staticmethod
    def get_password_hash(password: str):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

class AuditHistory(Base):
    __tablename__ = "audit_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    dataset_name = Column(String)
    model_name = Column(String)
    critical_findings_count = Column(Integer)
    total_bss = Column(Float)
    max_bias_pct = Column(Float)
    results_json = Column(Text) # Store the JSON payload of the results
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="audits")
