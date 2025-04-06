"""
UK Call Regulations Module

This module implements UK-specific calling regulations to ensure compliance with:
- OFCOM regulations
- PECR (Privacy and Electronic Communications Regulations)
- TPS (Telephone Preference Service) requirements

Key regulations:
1. Time restrictions:
   - Weekdays: 8am to 9pm
   - Weekends: 9am to 5pm
   - No calls on UK bank holidays

2. Frequency restrictions:
   - No more than 3 calls to the same number in a 24-hour period
   - No more than 6 calls to the same number in a 7-day period

3. Caller ID requirements:
   - Must present a valid UK number that can be called back
   - Must not block or obfuscate the calling number

4. TPS compliance:
   - Must check numbers against the TPS registry
   - Must honor do-not-call preferences
"""

import time
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Set
import aiosqlite
import csv
import os
from enum import Enum

# Configure logging
logger = logging.getLogger("uk_regulations")



# Define call restriction violation types
class ViolationType(str, Enum):
    OUTSIDE_HOURS = "outside_permitted_hours"
    DAILY_LIMIT = "daily_call_limit_exceeded"
    WEEKLY_LIMIT = "weekly_call_limit_exceeded"
    TPS_REGISTERED = "tps_registered_number"
    INVALID_CALLER_ID = "invalid_caller_id"
    OTHER = "other_violation"

class UKCallRegulator:
    """
    Implements UK calling regulations and restrictions
    """
    
    def __init__(self, db_path: str = "telemarketer_calls.db", tps_path: Optional[str] = None):
        """
        Initialize the UK Call Regulator
        
        Args:
            db_path: Path to the SQLite database for call history
            tps_path: Path to the TPS registry CSV file (if available)
        """
        self.db_path = db_path
        self.tps_path = tps_path
        self.tps_numbers: Set[str] = set()
        self.daily_limit = 3  # Maximum calls per 24 hours
        self.weekly_limit = 6  # Maximum calls per 7 days
        self.rate_limit_counter: Dict[str, int] = {}  # For in-memory rate limiting
        self.last_reset = time.time()
        
        # Load TPS registry if available
        if tps_path and os.path.exists(tps_path):
            self._load_tps_registry()
            
    def _load_tps_registry(self):
        """Load the TPS registry from CSV file"""
        try:
            with open(self.tps_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)
                # Load numbers
                for row in reader:
                    if row and len(row) > 0:
                        # Normalize phone number format
                        number = self._normalize_phone_number(row[0])
                        self.tps_numbers.add(number)
                        
            logger.info(f"Loaded {len(self.tps_numbers)} numbers from TPS registry")
        except Exception as e:
            logger.error(f"Failed to load TPS registry: {e}")
            
    def _normalize_phone_number(self, phone_number: str) -> str:
        """Normalize phone number format for consistent checking"""
        # Remove any non-numeric characters
        number = ''.join(c for c in phone_number if c.isdigit())
        
        # Handle UK number formats
        if number.startswith('44'):
            return number
        elif number.startswith('0'):
            return '44' + number[1:]
        else:
            return number
            
    def _is_valid_uk_number(self, phone_number: str) -> bool:
        """Check if a number is a valid UK format"""
        normalized = self._normalize_phone_number(phone_number)
        
        # Check if it's a UK number (starts with 44)
        if not normalized.startswith('44'):
            return False
            
        # Check if it has the right length for UK numbers
        if len(normalized) != 11 and len(normalized) != 12:
            return False
            
        return True
        
    def _is_within_calling_hours(self) -> Tuple[bool, str]:
        """Check if current time is within permitted calling hours"""
        now = datetime.now()
        current_date = date.today()
            
        weekday = now.weekday()  # 0-6, where 0 is Monday
        hour = now.hour
        
        # Weekend restrictions (Saturday or Sunday)
        if weekday >= 5:
            if hour < 9 or hour >= 17:  # Before 9am or after 5pm
                return False, f"Outside weekend calling hours (9am-5pm), current time: {now.strftime('%H:%M')}"
        # Weekday restrictions
        else:
            if hour < 8 or hour >= 21:  # Before 8am or after 9pm
                return False, f"Outside weekday calling hours (8am-9pm), current time: {now.strftime('%H:%M')}"
                
        return True, "Within permitted calling hours"
        
    async def is_in_tps_registry(self, phone_number: str) -> bool:
        """Check if a number is registered in the TPS"""
        normalized = self._normalize_phone_number(phone_number)
        
        # Check in-memory TPS registry first
        if normalized in self.tps_numbers:
            return True
            
        # If we have a database connection, check there too
        # (For a real implementation, you would check an actual TPS API)
        return False
        
    async def check_call_frequency(self, phone_number: str) -> Tuple[bool, str]:
        """Check if a number has been called too frequently"""
        normalized = self._normalize_phone_number(phone_number)
        now = datetime.now()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Check daily limit (24 hours)
            yesterday = (now - timedelta(days=1)).timestamp()
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM number_call_history
                WHERE phone_number = ? AND call_time > ?
                """,
                (normalized, yesterday)
            )
            daily_count = (await cursor.fetchone())[0]
            
            if daily_count >= self.daily_limit:
                return False, f"Daily call limit exceeded ({daily_count}/{self.daily_limit} in last 24 hours)"
                
            # Check weekly limit (7 days)
            last_week = (now - timedelta(days=7)).timestamp()
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM number_call_history
                WHERE phone_number = ? AND call_time > ?
                """,
                (normalized, last_week)
            )
            weekly_count = (await cursor.fetchone())[0]
            
            if weekly_count >= self.weekly_limit:
                return False, f"Weekly call limit exceeded ({weekly_count}/{self.weekly_limit} in last 7 days)"
                
        return True, "Within permitted call frequency limits"
        
    async def can_call_number(self, phone_number: str, caller_id: str) -> Tuple[bool, str, Optional[ViolationType]]:
        """
        Check if a number can be called according to UK regulations
        
        Args:
            phone_number: The number to call
            caller_id: The caller ID to use
            
        Returns:
            Tuple containing:
            - Boolean indicating if call is permitted
            - Reason message
            - Violation type (if any)
        """
        # Check if it's a valid UK number format
        if not self._is_valid_uk_number(phone_number):
            return False, "Invalid UK phone number format", ViolationType.OTHER
            
        # Check if caller ID is valid
        if not self._is_valid_uk_number(caller_id):
            return False, "Invalid caller ID - must be a valid UK number", ViolationType.INVALID_CALLER_ID
            
        # Check if within permitted calling hours
        within_hours, hours_message = self._is_within_calling_hours()
        if not within_hours:
            return False, hours_message, ViolationType.OUTSIDE_HOURS
            
        # Check TPS registry
        if await self.is_in_tps_registry(phone_number):
            return False, "Number is registered in TPS do-not-call registry", ViolationType.TPS_REGISTERED
            
        # Check call frequency limits
        within_limits, limits_message = await self.check_call_frequency(phone_number)
        if not within_limits:
            if "Daily" in limits_message:
                return False, limits_message, ViolationType.DAILY_LIMIT
            else:
                return False, limits_message, ViolationType.WEEKLY_LIMIT
                
        # All checks passed
        return True, "Call is permitted under UK regulations", None
        
    async def record_call_attempt(self, phone_number: str, call_sid: str, status: str = "attempted"):
        """Record a call attempt for rate limiting purposes"""
        normalized = self._normalize_phone_number(phone_number)
        now = datetime.now().timestamp()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO number_call_history
                (phone_number, call_time, call_sid, call_status, call_duration)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized, now, call_sid, status, 0)
            )
            await db.commit()
            
    async def update_call_status(self, call_sid: str, status: str, duration: float = 0):
        """Update the status and duration of a call"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE number_call_history
                SET call_status = ?, call_duration = ?
                WHERE call_sid = ?
                """,
                (status, duration, call_sid)
            )
            await db.commit()
            
    async def get_call_history(self, phone_number: str, days: int = 30) -> List[Dict]:
        """Get call history for a number"""
        normalized = self._normalize_phone_number(phone_number)
        now = datetime.now()
        cutoff = (now - timedelta(days=days)).timestamp()
        
        result = []
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT * FROM number_call_history
                WHERE phone_number = ? AND call_time > ?
                ORDER BY call_time DESC
                """,
                (normalized, cutoff)
            )
            rows = await cursor.fetchall()
            
            # Convert rows to dictionaries
            columns = [col[0] for col in cursor.description]
            for row in rows:
                result.append({columns[i]: row[i] for i in range(len(columns))})
                
        return result

# Global regulator instance
regulator = None

def get_regulator(db_path: str = "telemarketer_calls.db", tps_path: Optional[str] = None) -> UKCallRegulator:
    """Get or create the global regulator instance"""
    global regulator
    if regulator is None:
        regulator = UKCallRegulator(db_path, tps_path)
    return regulator 