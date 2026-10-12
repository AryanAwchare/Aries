"""Initial migration creating all 7 core tables + EA bridge table.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-10-03
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. leaders table
    op.create_table(
        'leaders',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('myfxbook_id', sa.String(length=64), nullable=True),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('broker', sa.String(length=64), server_default='', nullable=False),
        sa.Column('account_age', sa.Integer(), server_default='0', nullable=False),
        sa.Column('verified_track_record', sa.Boolean(), server_default='0', nullable=False),
        sa.Column('verified_trading_privileges', sa.Boolean(), server_default='0', nullable=False),
        sa.Column('symbol', sa.String(length=32), server_default='XAUUSD', nullable=False),
        sa.Column('strategy_type', sa.String(length=64), server_default='unknown', nullable=False),
        sa.Column('status', sa.String(length=32), server_default='watch', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='1', nullable=False),
        sa.Column('url', sa.String(length=512), server_default='', nullable=False),
        sa.Column('audited', sa.Boolean(), server_default='0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_leaders_myfxbook_id', 'leaders', ['myfxbook_id'])
    op.create_index('ix_leaders_symbol', 'leaders', ['symbol'])

    # 2. leader_scores table
    op.create_table(
        'leader_scores',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('leader_id', sa.String(length=64), sa.ForeignKey('leaders.id'), nullable=False),
        sa.Column('date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('profit_factor', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('win_rate', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('max_drawdown', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('expectancy', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('consistency', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('longevity', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('recent_score', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('composite_score', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('avg_win', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('avg_loss', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('total_pips', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('trade_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('consistency_score', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('leader_id', 'date', name='uq_leader_score_date'),
    )
    op.create_index('ix_leader_scores_leader_id', 'leader_scores', ['leader_id'])
    op.create_index('ix_leader_scores_date', 'leader_scores', ['date'])

    # 3. leader_trades table
    op.create_table(
        'leader_trades',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('leader_id', sa.String(length=64), sa.ForeignKey('leaders.id'), nullable=False),
        sa.Column('trade_id', sa.String(length=128), nullable=False),
        sa.Column('symbol', sa.String(length=32), nullable=False),
        sa.Column('direction', sa.String(length=16), nullable=False),
        sa.Column('entry', sa.Float(), nullable=True),
        sa.Column('exit', sa.Float(), nullable=True),
        sa.Column('lot_size', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('profit', sa.Float(), nullable=True),
        sa.Column('opened_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_open', sa.Boolean(), server_default='1', nullable=False),
        sa.Column('raw_json', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('leader_id', 'trade_id', name='uq_leader_trade_trade_id'),
    )
    op.create_index('ix_leader_trades_leader_id', 'leader_trades', ['leader_id'])
    op.create_index('ix_leader_trades_trade_id', 'leader_trades', ['trade_id'])
    op.create_index('ix_leader_trades_symbol', 'leader_trades', ['symbol'])
    op.create_index('ix_leader_trades_opened_at', 'leader_trades', ['opened_at'])

    # 4. signals table
    op.create_table(
        'signals',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('symbol', sa.String(length=32), server_default='XAUUSD', nullable=False),
        sa.Column('timeframe', sa.String(length=16), server_default='M15', nullable=False),
        sa.Column('leader_direction', sa.String(length=16), nullable=True),
        sa.Column('kronos_direction', sa.String(length=16), nullable=True),
        sa.Column('kronos_confidence', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('xgb_probability', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('regime', sa.String(length=32), server_default='ranging', nullable=False),
        sa.Column('final_signal', sa.String(length=16), server_default='NO_TRADE', nullable=False),
        sa.Column('contributing_factors', sa.JSON(), nullable=False),
        sa.Column('trade_event_id', sa.String(length=64), nullable=True),
        sa.Column('leader_id', sa.String(length=64), nullable=True),
        sa.Column('side', sa.String(length=16), nullable=True),
        sa.Column('leader_score', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('kronos_forecast_pct', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('meta_model_probability', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('threshold', sa.Float(), server_default='0.6', nullable=False),
        sa.Column('verb', sa.String(length=16), server_default='alert', nullable=False),
        sa.Column('status', sa.String(length=16), server_default='pending', nullable=False),
        sa.Column('rejection_reason', sa.Text(), server_default='', nullable=False),
        sa.Column('features_json', sa.JSON(), nullable=False),
        sa.Column('decided_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_signals_timestamp', 'signals', ['timestamp'])
    op.create_index('ix_signals_symbol', 'signals', ['symbol'])
    op.create_index('ix_signals_leader_id', 'signals', ['leader_id'])
    op.create_index('ix_signals_trade_event_id', 'signals', ['trade_event_id'])
    op.create_index('ix_signals_decided_at', 'signals', ['decided_at'])

    # 5. orders table
    op.create_table(
        'orders',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('signal_id', sa.String(length=64), sa.ForeignKey('signals.id'), nullable=True),
        sa.Column('broker_order_id', sa.String(length=64), nullable=True),
        sa.Column('symbol', sa.String(length=32), server_default='XAUUSD', nullable=False),
        sa.Column('direction', sa.String(length=16), nullable=False),
        sa.Column('volume', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('entry', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('sl', sa.Float(), nullable=True),
        sa.Column('tp', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=16), server_default='submitted', nullable=False),
        sa.Column('mode', sa.String(length=16), server_default='paper', nullable=False),
        sa.Column('error', sa.Text(), server_default='', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('filled_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_orders_signal_id', 'orders', ['signal_id'])
    op.create_index('ix_orders_broker_order_id', 'orders', ['broker_order_id'])
    op.create_index('ix_orders_symbol', 'orders', ['symbol'])
    op.create_index('ix_orders_created_at', 'orders', ['created_at'])

    # 6. risk_events table
    op.create_table(
        'risk_events',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('event', sa.String(length=64), nullable=False),
        sa.Column('severity', sa.String(length=32), server_default='INFO', nullable=False),
        sa.Column('drawdown', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('daily_loss', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('action', sa.String(length=64), server_default='', nullable=False),
        sa.Column('detail', sa.JSON(), nullable=False),
    )
    op.create_index('ix_risk_events_timestamp', 'risk_events', ['timestamp'])
    op.create_index('ix_risk_events_event', 'risk_events', ['event'])

    # 7. system_events table
    op.create_table(
        'system_events',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('component', sa.String(length=64), nullable=False),
        sa.Column('event', sa.String(length=64), nullable=False),
        sa.Column('detail', sa.JSON(), nullable=False),
    )
    op.create_index('ix_system_events_timestamp', 'system_events', ['timestamp'])
    op.create_index('ix_system_events_component', 'system_events', ['component'])
    op.create_index('ix_system_events_event', 'system_events', ['event'])

    # 8. ea_orders table (for MT5 EA polling bridge)
    op.create_table(
        'ea_orders',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('external_id', sa.String(length=64), nullable=False),
        sa.Column('order_id', sa.String(length=64), nullable=True),
        sa.Column('symbol', sa.String(length=32), nullable=False),
        sa.Column('side', sa.String(length=16), nullable=False),
        sa.Column('volume', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('stop_loss', sa.Float(), nullable=True),
        sa.Column('take_profit', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=16), server_default='pending', nullable=False),
        sa.Column('broker_order_id', sa.String(length=64), nullable=True),
        sa.Column('fill_price', sa.Float(), nullable=True),
        sa.Column('error', sa.Text(), server_default='', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('dispatched_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_ea_orders_external_id', 'ea_orders', ['external_id'])
    op.create_index('ix_ea_orders_order_id', 'ea_orders', ['order_id'])
    op.create_index('ix_ea_orders_symbol', 'ea_orders', ['symbol'])
    op.create_index('ix_ea_orders_created_at', 'ea_orders', ['created_at'])


def downgrade() -> None:
    op.drop_table('ea_orders')
    op.drop_table('system_events')
    op.drop_table('risk_events')
    op.drop_table('orders')
    op.drop_table('signals')
    op.drop_table('leader_trades')
    op.drop_table('leader_scores')
    op.drop_table('leaders')