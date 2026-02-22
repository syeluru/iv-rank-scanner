"""
Signal approval queue for manual trade approval.

Manages pending signals awaiting user approval with database persistence
for crash recovery and audit trail.
"""

import uuid
import json
import asyncio
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from threading import Lock
from loguru import logger

from strategies.base_strategy import Signal
from data.storage.database import trades_db


class ApprovalStatus(Enum):
    """Status of signal approval."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class PendingSignal:
    """
    Signal awaiting manual approval.

    Tracks signal state, timestamps, and approval decision.
    """
    signal_id: str
    signal: Signal
    status: ApprovalStatus
    generated_at: datetime
    expires_at: datetime
    approved_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'signal_id': self.signal_id,
            'signal': asdict(self.signal),
            'status': self.status.value,
            'generated_at': self.generated_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'rejected_at': self.rejected_at.isoformat() if self.rejected_at else None,
            'rejection_reason': self.rejection_reason
        }

    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        return datetime.now() >= self.expires_at

    @property
    def time_remaining(self) -> timedelta:
        """Get time remaining until expiration."""
        remaining = self.expires_at - datetime.now()
        return remaining if remaining.total_seconds() > 0 else timedelta(0)


class SignalQueue:
    """
    Queue for managing signals awaiting approval.

    Provides thread-safe access to pending signals with database persistence.
    Supports approval workflow: add → approve/reject/expire → cleanup.
    """

    def __init__(self):
        """Initialize signal queue."""
        self._pending: Dict[str, PendingSignal] = {}
        self._lock = Lock()
        self._approval_events: Dict[str, asyncio.Event] = {}

        # Initialize database table if needed
        self._init_database()

        # Load any existing pending signals from database
        self._load_pending_signals()

        logger.info("SignalQueue initialized")

    def _init_database(self) -> None:
        """Initialize approval queue database table."""
        conn = trades_db.connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS approval_queue (
                signal_id VARCHAR PRIMARY KEY,
                strategy_name VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                signal_data JSON NOT NULL,
                status VARCHAR NOT NULL,
                generated_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                approved_at TIMESTAMP,
                rejected_at TIMESTAMP,
                rejection_reason VARCHAR
            )
        """)

        # Create index for efficient status queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_approval_status
            ON approval_queue(status, expires_at)
        """)

        conn.commit()
        logger.debug("Approval queue database table initialized")

    def _load_pending_signals(self) -> None:
        """Load pending signals from database on startup."""
        conn = trades_db.connect()
        result = conn.execute("""
            SELECT signal_id, signal_data, status,
                   generated_at, expires_at, approved_at,
                   rejected_at, rejection_reason
            FROM approval_queue
            WHERE status = 'pending' AND expires_at > ?
        """, (datetime.now(),)).fetchall()

        for row in result:
            signal_data = json.loads(row[1])

            # Reconstruct Signal object
            from strategies.base_strategy import SignalType, SignalDirection
            from datetime import date

            signal = Signal(
                signal_id=signal_data['signal_id'],
                strategy_name=signal_data['strategy_name'],
                signal_type=SignalType[signal_data['signal_type']],
                direction=SignalDirection[signal_data['direction']],
                symbol=signal_data['symbol'],
                generated_at=datetime.fromisoformat(signal_data['generated_at']),
                entry_price=signal_data.get('entry_price'),
                stop_loss=signal_data.get('stop_loss'),
                take_profit=signal_data.get('take_profit'),
                quantity=signal_data.get('quantity'),
                position_size_pct=signal_data.get('position_size_pct'),
                option_type=signal_data.get('option_type'),
                strike=signal_data.get('strike'),
                expiration=date.fromisoformat(signal_data['expiration']) if signal_data.get('expiration') else None,
                delta=signal_data.get('delta'),
                confidence=signal_data.get('confidence', 0.0),
                priority=signal_data.get('priority', 0),
                notes=signal_data.get('notes', ''),
                metadata=signal_data.get('metadata', {}),
                position_id=signal_data.get('position_id')
            )

            pending_signal = PendingSignal(
                signal_id=row[0],
                signal=signal,
                status=ApprovalStatus(row[2]),
                generated_at=row[3],
                expires_at=row[4],
                approved_at=row[5],
                rejected_at=row[6],
                rejection_reason=row[7]
            )

            self._pending[row[0]] = pending_signal
            logger.info(f"Loaded pending signal from database: {row[0]} ({signal.symbol})")

    def add_signal(self, signal: Signal, timeout_seconds: int = 300) -> str:
        """
        Add signal to approval queue.

        Args:
            signal: Signal to add
            timeout_seconds: Seconds until signal expires (default 5 minutes)

        Returns:
            Signal ID for tracking
        """
        signal_id = str(uuid.uuid4())
        generated_at = datetime.now()
        expires_at = generated_at + timedelta(seconds=timeout_seconds)

        pending_signal = PendingSignal(
            signal_id=signal_id,
            signal=signal,
            status=ApprovalStatus.PENDING,
            generated_at=generated_at,
            expires_at=expires_at
        )

        with self._lock:
            self._pending[signal_id] = pending_signal
            self._approval_events[signal_id] = asyncio.Event()

        # Persist to database
        self._save_to_database(pending_signal)

        logger.info(
            f"Signal {signal_id} added to approval queue: "
            f"{signal.symbol} ({signal.signal_type.value}), "
            f"expires in {timeout_seconds}s"
        )

        return signal_id

    def _save_to_database(self, pending_signal: PendingSignal) -> None:
        """Save pending signal to database."""
        signal = pending_signal.signal

        # Serialize signal to JSON
        signal_dict = asdict(signal)
        # Convert enum values to strings
        signal_dict['signal_type'] = signal.signal_type.value
        signal_dict['direction'] = signal.direction.value
        signal_dict['generated_at'] = signal.generated_at.isoformat()
        if signal.expiration:
            signal_dict['expiration'] = signal.expiration.isoformat()

        signal_json = json.dumps(signal_dict)

        conn = trades_db.connect()
        conn.execute("""
            INSERT OR REPLACE INTO approval_queue
            (signal_id, strategy_name, symbol, signal_data, status,
             generated_at, expires_at, approved_at, rejected_at, rejection_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pending_signal.signal_id,
            signal.strategy_name,
            signal.symbol,
            signal_json,
            pending_signal.status.value,
            pending_signal.generated_at,
            pending_signal.expires_at,
            pending_signal.approved_at,
            pending_signal.rejected_at,
            pending_signal.rejection_reason
        ))
        conn.commit()

    def get_pending_signal(self, signal_id: str) -> Optional[PendingSignal]:
        """
        Retrieve pending signal by ID.

        Args:
            signal_id: Signal ID

        Returns:
            PendingSignal if found, None otherwise
        """
        with self._lock:
            return self._pending.get(signal_id)

    def get_all_pending(self) -> List[PendingSignal]:
        """
        Get all pending signals.

        Returns:
            List of pending signals sorted by expiration time
        """
        with self._lock:
            pending = [
                ps for ps in self._pending.values()
                if ps.status == ApprovalStatus.PENDING and not ps.is_expired
            ]
            return sorted(pending, key=lambda ps: ps.expires_at)

    def approve_signal(self, signal_id: str) -> bool:
        """
        Approve a pending signal.

        Args:
            signal_id: Signal ID to approve

        Returns:
            True if approved, False if not found or already processed
        """
        with self._lock:
            pending_signal = self._pending.get(signal_id)

            if not pending_signal:
                logger.warning(f"Signal {signal_id} not found for approval")
                return False

            if pending_signal.status != ApprovalStatus.PENDING:
                logger.warning(
                    f"Signal {signal_id} already processed with status: "
                    f"{pending_signal.status.value}"
                )
                return False

            if pending_signal.is_expired:
                logger.warning(f"Signal {signal_id} expired, marking as EXPIRED")
                pending_signal.status = ApprovalStatus.EXPIRED
                self._save_to_database(pending_signal)
                return False

            # Approve signal
            pending_signal.status = ApprovalStatus.APPROVED
            pending_signal.approved_at = datetime.now()

            # Persist to database
            self._save_to_database(pending_signal)

            # Notify waiting coroutines
            if signal_id in self._approval_events:
                self._approval_events[signal_id].set()

            logger.info(f"Signal {signal_id} APPROVED: {pending_signal.signal.symbol}")
            return True

    def reject_signal(self, signal_id: str, reason: str = "User rejected") -> bool:
        """
        Reject a pending signal.

        Args:
            signal_id: Signal ID to reject
            reason: Rejection reason

        Returns:
            True if rejected, False if not found or already processed
        """
        with self._lock:
            pending_signal = self._pending.get(signal_id)

            if not pending_signal:
                logger.warning(f"Signal {signal_id} not found for rejection")
                return False

            if pending_signal.status != ApprovalStatus.PENDING:
                logger.warning(
                    f"Signal {signal_id} already processed with status: "
                    f"{pending_signal.status.value}"
                )
                return False

            # Reject signal
            pending_signal.status = ApprovalStatus.REJECTED
            pending_signal.rejected_at = datetime.now()
            pending_signal.rejection_reason = reason

            # Persist to database
            self._save_to_database(pending_signal)

            # Notify waiting coroutines
            if signal_id in self._approval_events:
                self._approval_events[signal_id].set()

            logger.info(f"Signal {signal_id} REJECTED: {pending_signal.signal.symbol} - {reason}")
            return True

    def expire_signal(self, signal_id: str) -> bool:
        """
        Mark signal as expired.

        Args:
            signal_id: Signal ID to expire

        Returns:
            True if expired, False if not found
        """
        with self._lock:
            pending_signal = self._pending.get(signal_id)

            if not pending_signal:
                return False

            if pending_signal.status != ApprovalStatus.PENDING:
                return False

            # Mark as expired
            pending_signal.status = ApprovalStatus.EXPIRED

            # Persist to database
            self._save_to_database(pending_signal)

            # Notify waiting coroutines
            if signal_id in self._approval_events:
                self._approval_events[signal_id].set()

            logger.warning(f"Signal {signal_id} EXPIRED: {pending_signal.signal.symbol}")
            return True

    async def wait_for_approval(
        self,
        signal_id: str,
        timeout_seconds: int = 300
    ) -> ApprovalStatus:
        """
        Wait for signal approval (blocking).

        Args:
            signal_id: Signal ID to wait for
            timeout_seconds: Maximum wait time

        Returns:
            ApprovalStatus after decision or timeout

        Raises:
            TimeoutError: If timeout reached without decision
        """
        pending_signal = self.get_pending_signal(signal_id)

        if not pending_signal:
            raise ValueError(f"Signal {signal_id} not found")

        # If already decided, return immediately
        if pending_signal.status != ApprovalStatus.PENDING:
            return pending_signal.status

        # Wait for approval event with timeout
        try:
            event = self._approval_events.get(signal_id)
            if not event:
                raise ValueError(f"Approval event not found for {signal_id}")

            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)

            # Get updated status
            updated_signal = self.get_pending_signal(signal_id)
            if updated_signal:
                return updated_signal.status

            return ApprovalStatus.EXPIRED

        except asyncio.TimeoutError:
            logger.warning(f"Approval timeout for signal {signal_id}")
            self.expire_signal(signal_id)
            raise TimeoutError(f"Approval timeout for signal {signal_id}")

    def cleanup_expired(self) -> int:
        """
        Remove expired signals from queue.

        Returns:
            Number of signals cleaned up
        """
        expired_ids = []

        with self._lock:
            for signal_id, pending_signal in self._pending.items():
                if pending_signal.is_expired and pending_signal.status == ApprovalStatus.PENDING:
                    pending_signal.status = ApprovalStatus.EXPIRED
                    self._save_to_database(pending_signal)
                    expired_ids.append(signal_id)

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired signals")

        return len(expired_ids)

    def cleanup_old_signals(self, days: int = 7) -> int:
        """
        Remove old processed signals from database.

        Args:
            days: Remove signals older than this many days

        Returns:
            Number of signals removed
        """
        cutoff = datetime.now() - timedelta(days=days)

        conn = trades_db.connect()
        result = conn.execute("""
            DELETE FROM approval_queue
            WHERE status IN ('approved', 'rejected', 'expired')
            AND generated_at < ?
        """, (cutoff,))

        count = result.rowcount
        conn.commit()

        if count > 0:
            logger.info(f"Cleaned up {count} old signals from database")

        return count

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get approval queue statistics.

        Returns:
            Dictionary with statistics
        """
        conn = trades_db.connect()

        # Get counts by status
        result = conn.execute("""
            SELECT status, COUNT(*) as count
            FROM approval_queue
            GROUP BY status
        """).fetchall()

        status_counts = {row[0]: row[1] for row in result}

        # Get counts for today
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_result = conn.execute("""
            SELECT status, COUNT(*) as count
            FROM approval_queue
            WHERE generated_at >= ?
            GROUP BY status
        """, (today,)).fetchall()

        today_counts = {row[0]: row[1] for row in today_result}

        with self._lock:
            pending_count = len([
                ps for ps in self._pending.values()
                if ps.status == ApprovalStatus.PENDING and not ps.is_expired
            ])

        return {
            'total_pending': pending_count,
            'total_approved': status_counts.get('approved', 0),
            'total_rejected': status_counts.get('rejected', 0),
            'total_expired': status_counts.get('expired', 0),
            'today_approved': today_counts.get('approved', 0),
            'today_rejected': today_counts.get('rejected', 0),
            'today_expired': today_counts.get('expired', 0),
            'approval_rate': (
                status_counts.get('approved', 0) /
                max(1, sum(status_counts.values()))
            ) if status_counts else 0.0
        }


# Global singleton instance
signal_queue = SignalQueue()
