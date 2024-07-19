from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    symbol = Column(String)
    order_type = Column(String)
    amount = Column(Float)
    price = Column(Float)
    status = Column(String)

engine = create_engine('sqlite:///trading_bot.db', echo=True)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def save_trade(trade):
    """Saves a trade to the database."""
    try:
        session.add(trade)
        session.commit()
        logging.info(f"Trade saved: {trade}")
    except Exception as e:
        logging.error(f"Error saving trade: {e}")
        session.rollback()
        raise

def get_trades():
    """Fetches all trades from the database."""
    try:
        trades = session.query(Trade).all()
        logging.info("Trades fetched successfully")
        return trades
    except Exception as e:
        logging.error(f"Error fetching trades: {e}")
        raise

def close_session():
    """Closes the database session."""
    try:
        session.close()
        logging.info("Database session closed")
    except Exception as e:
        logging.error(f"Error closing database session: {e}")
        raise
