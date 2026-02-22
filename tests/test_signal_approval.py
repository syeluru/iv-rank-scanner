"""
Unit tests for signal approval system.

Tests the complete approval workflow including queue management,
terminal display, and strategy engine integration.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from strategies.base_strategy import Signal, SignalType, SignalDirection
from execution.approval.signal_queue import SignalQueue, ApprovalStatus, PendingSignal
from monitoring.terminal.signal_monitor import TerminalMonitor
from config.settings import settings


@pytest.fixture
def sample_signal():
    """Create a sample signal for testing."""
    return Signal(
        signal_id="test_signal_001",
        strategy_name="test_strategy",
        signal_type=SignalType.ENTRY,
        direction=SignalDirection.SHORT,
        symbol="AAPL",
        entry_price=3.50,
        quantity=1,
        option_type="PUT",
        strike=145.0,
        expiration=datetime.now().date() + timedelta(days=42),
        delta=-0.25,
        confidence=0.82,
        priority=2,
        notes="Test signal for unit testing",
        metadata={
            'iv_rank': 72.5,
            'iv': 28.3,
            'stock_price': 152.45
        }
    )


@pytest.fixture
def signal_queue():
    """Create a fresh signal queue for testing."""
    # Use in-memory database for testing
    queue = SignalQueue()
    yield queue
    # Cleanup
    queue.cleanup_old_signals(days=0)


@pytest.fixture
def terminal_monitor():
    """Create terminal monitor for testing."""
    return TerminalMonitor()


class TestSignalQueue:
    """Test cases for SignalQueue."""

    def test_add_signal(self, signal_queue, sample_signal):
        """Test adding signal to queue."""
        signal_id = signal_queue.add_signal(sample_signal, timeout_seconds=300)

        assert signal_id is not None
        assert len(signal_id) > 0

        # Verify signal was added
        pending_signal = signal_queue.get_pending_signal(signal_id)
        assert pending_signal is not None
        assert pending_signal.signal.symbol == "AAPL"
        assert pending_signal.status == ApprovalStatus.PENDING

    def test_approve_signal(self, signal_queue, sample_signal):
        """Test approving a signal."""
        signal_id = signal_queue.add_signal(sample_signal, timeout_seconds=300)

        # Approve signal
        success = signal_queue.approve_signal(signal_id)
        assert success is True

        # Verify status changed
        pending_signal = signal_queue.get_pending_signal(signal_id)
        assert pending_signal.status == ApprovalStatus.APPROVED
        assert pending_signal.approved_at is not None

    def test_reject_signal(self, signal_queue, sample_signal):
        """Test rejecting a signal."""
        signal_id = signal_queue.add_signal(sample_signal, timeout_seconds=300)

        # Reject signal
        success = signal_queue.reject_signal(signal_id, reason="Test rejection")
        assert success is True

        # Verify status changed
        pending_signal = signal_queue.get_pending_signal(signal_id)
        assert pending_signal.status == ApprovalStatus.REJECTED
        assert pending_signal.rejected_at is not None
        assert pending_signal.rejection_reason == "Test rejection"

    def test_expire_signal(self, signal_queue, sample_signal):
        """Test expiring a signal."""
        signal_id = signal_queue.add_signal(sample_signal, timeout_seconds=1)

        # Wait for expiration
        import time
        time.sleep(2)

        # Expire signal
        success = signal_queue.expire_signal(signal_id)
        assert success is True

        # Verify status changed
        pending_signal = signal_queue.get_pending_signal(signal_id)
        assert pending_signal.status == ApprovalStatus.EXPIRED

    def test_get_all_pending(self, signal_queue, sample_signal):
        """Test retrieving all pending signals."""
        # Add multiple signals
        signal_id_1 = signal_queue.add_signal(sample_signal, timeout_seconds=300)
        signal_id_2 = signal_queue.add_signal(sample_signal, timeout_seconds=300)

        # Get all pending
        pending = signal_queue.get_all_pending()
        assert len(pending) == 2

        # Approve one
        signal_queue.approve_signal(signal_id_1)

        # Should only have one pending now
        pending = signal_queue.get_all_pending()
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_wait_for_approval_approved(self, signal_queue, sample_signal):
        """Test waiting for approval (approved scenario)."""
        signal_id = signal_queue.add_signal(sample_signal, timeout_seconds=10)

        # Approve after short delay
        async def approve_after_delay():
            await asyncio.sleep(0.5)
            signal_queue.approve_signal(signal_id)

        # Start approval task
        approval_task = asyncio.create_task(approve_after_delay())

        # Wait for approval
        status = await signal_queue.wait_for_approval(signal_id, timeout_seconds=10)

        assert status == ApprovalStatus.APPROVED
        await approval_task

    @pytest.mark.asyncio
    async def test_wait_for_approval_timeout(self, signal_queue, sample_signal):
        """Test waiting for approval (timeout scenario)."""
        signal_id = signal_queue.add_signal(sample_signal, timeout_seconds=1)

        # Wait for approval with short timeout
        with pytest.raises(TimeoutError):
            await signal_queue.wait_for_approval(signal_id, timeout_seconds=1)

        # Verify signal expired
        pending_signal = signal_queue.get_pending_signal(signal_id)
        assert pending_signal.status == ApprovalStatus.EXPIRED

    def test_cleanup_expired(self, signal_queue, sample_signal):
        """Test cleaning up expired signals."""
        # Add signals with short timeout
        signal_queue.add_signal(sample_signal, timeout_seconds=1)
        signal_queue.add_signal(sample_signal, timeout_seconds=1)

        # Wait for expiration
        import time
        time.sleep(2)

        # Cleanup
        count = signal_queue.cleanup_expired()
        assert count == 2

    def test_get_statistics(self, signal_queue, sample_signal):
        """Test getting queue statistics."""
        # Add and process signals
        signal_id_1 = signal_queue.add_signal(sample_signal, timeout_seconds=300)
        signal_id_2 = signal_queue.add_signal(sample_signal, timeout_seconds=300)
        signal_id_3 = signal_queue.add_signal(sample_signal, timeout_seconds=300)

        signal_queue.approve_signal(signal_id_1)
        signal_queue.reject_signal(signal_id_2, "Test")
        signal_queue.expire_signal(signal_id_3)

        # Get statistics
        stats = signal_queue.get_statistics()

        assert stats['total_pending'] == 0
        assert stats['total_approved'] >= 1
        assert stats['total_rejected'] >= 1
        assert stats['total_expired'] >= 1


class TestTerminalMonitor:
    """Test cases for TerminalMonitor."""

    def test_initialization(self, terminal_monitor):
        """Test terminal monitor initialization."""
        assert terminal_monitor is not None
        assert not terminal_monitor.is_running

    def test_display_signal(self, terminal_monitor, sample_signal):
        """Test displaying signal in terminal."""
        pending_signal = PendingSignal(
            signal_id="test_001",
            signal=sample_signal,
            status=ApprovalStatus.PENDING,
            generated_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5)
        )

        # Should not raise exception
        terminal_monitor.display_signal(pending_signal)

    def test_show_approval_confirmation(self, terminal_monitor, sample_signal):
        """Test showing approval confirmation."""
        # Should not raise exception
        terminal_monitor.show_approval_confirmation(sample_signal)

    def test_show_rejection_confirmation(self, terminal_monitor, sample_signal):
        """Test showing rejection confirmation."""
        # Should not raise exception
        terminal_monitor.show_rejection_confirmation(sample_signal, "Test rejection")

    def test_process_input_approve(self, terminal_monitor):
        """Test processing approval input."""
        approved, reason = terminal_monitor._process_input("y")
        assert approved is True
        assert reason is None

        approved, reason = terminal_monitor._process_input("yes")
        assert approved is True

    def test_process_input_reject(self, terminal_monitor):
        """Test processing rejection input."""
        approved, reason = terminal_monitor._process_input("n")
        assert approved is False
        assert reason is not None

    def test_process_input_quit(self, terminal_monitor):
        """Test processing quit input."""
        approved, reason = terminal_monitor._process_input("q")
        assert approved is False
        assert reason == "User quit"


class TestStrategyEngineIntegration:
    """Test cases for strategy engine integration."""

    @pytest.mark.asyncio
    async def test_request_approval_with_mock(self):
        """Test approval request with mocked components."""
        from strategies.strategy_engine import StrategyEngine

        engine = StrategyEngine()

        # Create test signal
        signal = Signal(
            signal_id="test_001",
            strategy_name="test_strategy",
            signal_type=SignalType.ENTRY,
            direction=SignalDirection.SHORT,
            symbol="AAPL",
            entry_price=3.50,
            quantity=1,
            confidence=0.8
        )

        # Mock settings to enable approval
        with patch('strategies.strategy_engine.settings') as mock_settings:
            mock_settings.ENABLE_MANUAL_APPROVAL = True
            mock_settings.APPROVAL_TIMEOUT_SECONDS = 5
            mock_settings.TERMINAL_REFRESH_RATE = 0.5

            # Mock terminal monitor to auto-approve
            with patch('strategies.strategy_engine.terminal_monitor') as mock_monitor:
                mock_monitor.get_user_decision = AsyncMock(return_value=(True, None))
                mock_monitor.show_approval_confirmation = Mock()

                # Request approval
                status = await engine._request_approval(signal)

                assert status == ApprovalStatus.APPROVED
                mock_monitor.get_user_decision.assert_called_once()
                mock_monitor.show_approval_confirmation.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_approval_rejected(self):
        """Test approval request with rejection."""
        from strategies.strategy_engine import StrategyEngine

        engine = StrategyEngine()

        signal = Signal(
            signal_id="test_002",
            strategy_name="test_strategy",
            signal_type=SignalType.ENTRY,
            direction=SignalDirection.SHORT,
            symbol="AAPL",
            entry_price=3.50,
            quantity=1,
            confidence=0.8
        )

        with patch('strategies.strategy_engine.settings') as mock_settings:
            mock_settings.ENABLE_MANUAL_APPROVAL = True
            mock_settings.APPROVAL_TIMEOUT_SECONDS = 5
            mock_settings.TERMINAL_REFRESH_RATE = 0.5

            # Mock terminal monitor to reject
            with patch('strategies.strategy_engine.terminal_monitor') as mock_monitor:
                mock_monitor.get_user_decision = AsyncMock(return_value=(False, "Test rejection"))
                mock_monitor.show_rejection_confirmation = Mock()

                # Request approval
                status = await engine._request_approval(signal)

                assert status == ApprovalStatus.REJECTED
                mock_monitor.show_rejection_confirmation.assert_called_once()


class TestPendingSignal:
    """Test cases for PendingSignal dataclass."""

    def test_is_expired(self, sample_signal):
        """Test checking if signal is expired."""
        # Not expired
        pending = PendingSignal(
            signal_id="test_001",
            signal=sample_signal,
            status=ApprovalStatus.PENDING,
            generated_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5)
        )
        assert not pending.is_expired

        # Expired
        pending_expired = PendingSignal(
            signal_id="test_002",
            signal=sample_signal,
            status=ApprovalStatus.PENDING,
            generated_at=datetime.now() - timedelta(minutes=10),
            expires_at=datetime.now() - timedelta(minutes=5)
        )
        assert pending_expired.is_expired

    def test_time_remaining(self, sample_signal):
        """Test calculating time remaining."""
        pending = PendingSignal(
            signal_id="test_001",
            signal=sample_signal,
            status=ApprovalStatus.PENDING,
            generated_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5)
        )

        remaining = pending.time_remaining
        assert remaining.total_seconds() > 0
        assert remaining.total_seconds() <= 300  # 5 minutes

    def test_to_dict(self, sample_signal):
        """Test converting to dictionary."""
        pending = PendingSignal(
            signal_id="test_001",
            signal=sample_signal,
            status=ApprovalStatus.PENDING,
            generated_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5)
        )

        data = pending.to_dict()
        assert data['signal_id'] == "test_001"
        assert data['status'] == "pending"
        assert 'signal' in data
        assert 'generated_at' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
