# Alpha Arena - Operational Manual

**Production deployment, monitoring, and maintenance guide**

---

## Quick Navigation

| I want to... | Go to |
|--------------|-------|
| Understand the architecture | [System Overview](#1-system-overview) |
| Deploy to production | [Deployment Guide](#3-deployment-guide) |
| Set up monitoring | [Monitoring & Alerting](#5-monitoring--alerting) |
| Handle an incident | [Incident Response](#10-incident-response) |
| Back up the system | [Backup & Recovery](#7-backup--recovery) |
| Improve performance | [Scaling & Performance](#8-scaling--performance) |

---

## Table of Contents

| Section | Topics |
|---------|--------|
| [1. System Overview](#1-system-overview) | Architecture, components, data flow |
| [2. Infrastructure Requirements](#2-infrastructure-requirements) | Hardware, software, network |
| [3. Deployment Guide](#3-deployment-guide) | Installation, configuration, launch |
| [4. Daily Operations](#4-daily-operations) | Checklists, monitoring, routine tasks |
| [5. Monitoring & Alerting](#5-monitoring--alerting) | Metrics, dashboards, alerts |
| [6. Maintenance Procedures](#6-maintenance-procedures) | Updates, patches, cleanup |
| [7. Backup & Recovery](#7-backup--recovery) | Backup strategy, restore procedures |
| [8. Scaling & Performance](#8-scaling--performance) | Optimization, horizontal scaling |
| [9. Security Operations](#9-security-operations) | Access control, secrets, auditing |
| [10. Incident Response](#10-incident-response) | Runbooks, escalation, post-mortems |
| [11. Cost Management](#11-cost-management) | API costs, optimization, budgeting |
| [12. Long-Term Sustainability](#12-long-term-sustainability) | Roadmap, upgrades, deprecation |

---

## 1. System Overview

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRADING HARNESS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Research   │    │     Risk     │    │  Execution   │       │
│  │    Agent     │◄──►│    Agent     │◄──►│    Agent     │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             ▼                                    │
│                   ┌──────────────────┐                           │
│                   │    Coordinator   │                           │
│                   └────────┬─────────┘                           │
│                            │                                     │
│         ┌──────────────────┼──────────────────┐                  │
│         ▼                  ▼                  ▼                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│  │   Memory    │   │  Indicators │   │  Arbitrage  │            │
│  │   System    │   │  Calculator │   │  Detector   │            │
│  └─────────────┘   └─────────────┘   └─────────────┘            │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     EXTERNAL SERVICES                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │Polymarket│  │ Anthropic│  │  OpenAI  │  │   xAI    │         │
│  │   CLOB   │  │  Claude  │  │  GPT-4o  │  │   Grok   │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Responsibilities

| Component | Responsibility | Criticality |
|-----------|---------------|-------------|
| Agent Coordinator | Orchestrates multi-agent decisions | Critical |
| Research Agent | Market analysis, opportunity identification | High |
| Risk Agent | Position sizing, risk validation | Critical |
| Execution Agent | Order placement optimization | High |
| Reflection Agent | Post-trade learning | Medium |
| Memory System | Context persistence, pattern storage | Medium |
| Indicator Calculator | Technical analysis | Medium |
| Arbitrage Detector | Cross-platform opportunities | Low |

### 1.3 Data Flow

1. **Market Data** → Polymarket API → MarketData → Agents
2. **Decisions** → Coordinator → Broker → Polymarket CLOB
3. **Logs** → MetricsLogger → SQLite/JSONL
4. **Memory** → Short-term → Long-term → Episodic

---

## 2. Infrastructure Requirements

### 2.1 Minimum Requirements

```yaml
Hardware:
  CPU: 4 cores
  RAM: 8 GB
  Storage: 50 GB SSD
  Network: 100 Mbps stable connection

Software:
  OS: Linux (Ubuntu 22.04+) / macOS 13+
  Python: 3.11+
  SQLite: 3.35+
```

### 2.2 Recommended Production Setup

```yaml
Hardware:
  CPU: 8+ cores
  RAM: 16 GB
  Storage: 200 GB NVMe SSD
  Network: 1 Gbps with redundancy

Software:
  OS: Ubuntu 22.04 LTS
  Python: 3.12
  Process Manager: systemd
  Monitoring: Prometheus + Grafana

Infrastructure:
  Primary: Cloud VM (AWS/GCP/Azure)
  Backup: Secondary region
  Database: Dedicated SQLite with WAL mode
```

### 2.3 Network Requirements

| Service | Endpoint | Port | Protocol |
|---------|----------|------|----------|
| Polymarket CLOB | clob.polymarket.com | 443 | HTTPS/WSS |
| Polymarket Gamma | gamma-api.polymarket.com | 443 | HTTPS |
| Polygon RPC | polygon-rpc.com | 443 | HTTPS |
| Anthropic | api.anthropic.com | 443 | HTTPS |
| OpenAI | api.openai.com | 443 | HTTPS |
| xAI | api.x.ai | 443 | HTTPS |

### 2.4 API Rate Limits

| Provider | Limit | Strategy |
|----------|-------|----------|
| Polymarket | 100 req/min | Exponential backoff |
| Anthropic | 60 req/min (varies by tier) | Queue with delays |
| OpenAI | 60 req/min (varies by tier) | Queue with delays |
| xAI | TBD | Conservative defaults |

---

## 3. Deployment Guide

### 3.1 Initial Setup

```bash
# 1. Clone repository
git clone <repository-url> alpha-arena
cd alpha-arena

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Copy and configure environment
cp .env.example .env
nano .env  # Fill in all credentials

# 5. Initialize directories
mkdir -p logs data/historical data/memory

# 6. Verify installation
python -m pytest tests/ -v
python cli.py --help
```

### 3.2 Production Deployment

```bash
# 1. Create systemd service
sudo nano /etc/systemd/system/polymarket-harness.service
```

```ini
[Unit]
Description=Polymarket Trading Harness
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/opt/alpha-arena
Environment="PATH=/opt/alpha-arena/.venv/bin"
ExecStart=/opt/alpha-arena/.venv/bin/python cli.py run-enhanced --interval 60
Restart=always
RestartSec=30
StandardOutput=append:/var/log/polymarket-harness/stdout.log
StandardError=append:/var/log/polymarket-harness/stderr.log

# Resource limits
MemoryMax=4G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
```

```bash
# 2. Enable and start
sudo systemctl daemon-reload
sudo systemctl enable polymarket-harness
sudo systemctl start polymarket-harness

# 3. Verify status
sudo systemctl status polymarket-harness
journalctl -u polymarket-harness -f
```

### 3.3 Configuration Validation

```bash
# Validate config before deployment
python -c "
import yaml
from pathlib import Path

config = yaml.safe_load(Path('config.yaml').read_text())
required = ['llm', 'risk', 'agent', 'markets']
for key in required:
    assert key in config, f'Missing: {key}'
print('Config valid')
"

# Validate environment
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

required = ['WALLET_PRIVATE_KEY', 'ANTHROPIC_API_KEY']
missing = [v for v in required if not os.getenv(v)]
if missing:
    print(f'Missing: {missing}')
    exit(1)
print('Environment valid')
"
```

---

## 4. Daily Operations

### 4.1 Daily Checklist

| Time | Task | Command/Action |
|------|------|----------------|
| 00:00 | Automated backup | Cron job runs |
| 06:00 | Review overnight activity | `python cli.py stats` |
| 08:00 | Check system health | `systemctl status polymarket-harness` |
| 12:00 | Midday performance check | Review logs and PnL |
| 18:00 | Verify risk limits | Check position sizes |
| 22:00 | Pre-night verification | Ensure system stable |

### 4.2 Operational Commands

```bash
# Check current status
python cli.py status

# View wallet balance
python cli.py wallet

# List open positions
python cli.py status

# View trading statistics
python cli.py stats

# Check LLM providers
python cli.py providers

# View recent markets
python cli.py markets --limit 20

# Scan arbitrage opportunities
python cli.py arbitrage

# Run backtest
python cli.py backtest --days 30 --capital 10000
```

### 4.3 Log Locations

```
logs/
├── decisions.jsonl      # All trading decisions
├── metrics.db           # SQLite metrics database
├── stdout.log           # Application output
└── stderr.log           # Error output

data/
├── memory.db            # Long-term memory
├── historical/          # Historical market data
└── backtest/            # Backtest results
```

### 4.4 Log Analysis Queries

```sql
-- Recent decisions (SQLite)
SELECT timestamp, balance, signals_generated, signals_executed
FROM decisions
ORDER BY timestamp DESC
LIMIT 20;

-- Daily PnL summary
SELECT date(timestamp) as date,
       SUM(realized_pnl) as total_pnl,
       COUNT(*) as trades
FROM pnl_snapshots
GROUP BY date(timestamp)
ORDER BY date DESC;

-- Error rate
SELECT date(timestamp),
       COUNT(CASE WHEN status = 'error' THEN 1 END) as errors,
       COUNT(*) as total
FROM decisions
GROUP BY date(timestamp);
```

---

## 5. Monitoring & Alerting

### 5.1 Key Metrics to Monitor

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|-------------------|
| System Memory | > 70% | > 90% |
| CPU Usage | > 70% | > 90% |
| Disk Space | < 20% free | < 10% free |
| API Latency | > 2s | > 5s |
| Error Rate | > 5% | > 20% |
| Daily Loss | > 50% of limit | > 80% of limit |
| Position Count | > 80% of max | = max |

### 5.2 Health Check Script

```python
#!/usr/bin/env python3
"""health_check.py - System health monitoring"""

import asyncio
import os
import psutil
from datetime import datetime, timedelta
from pathlib import Path

async def check_system():
    """Run all health checks."""
    issues = []

    # Memory check
    mem = psutil.virtual_memory()
    if mem.percent > 90:
        issues.append(f"CRITICAL: Memory at {mem.percent}%")
    elif mem.percent > 70:
        issues.append(f"WARNING: Memory at {mem.percent}%")

    # Disk check
    disk = psutil.disk_usage('/')
    free_pct = 100 - disk.percent
    if free_pct < 10:
        issues.append(f"CRITICAL: Disk {free_pct:.1f}% free")
    elif free_pct < 20:
        issues.append(f"WARNING: Disk {free_pct:.1f}% free")

    # Log freshness check
    log_path = Path("logs/decisions.jsonl")
    if log_path.exists():
        mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
        age = datetime.now() - mtime
        if age > timedelta(hours=1):
            issues.append(f"WARNING: No decisions logged for {age}")

    # Process check
    harness_running = False
    for proc in psutil.process_iter(['name', 'cmdline']):
        if 'python' in proc.info['name'] and 'cli.py' in str(proc.info['cmdline']):
            harness_running = True
            break

    if not harness_running:
        issues.append("CRITICAL: Harness process not running")

    return issues

if __name__ == "__main__":
    issues = asyncio.run(check_system())
    if issues:
        for issue in issues:
            print(issue)
        exit(1)
    else:
        print("OK: All checks passed")
        exit(0)
```

### 5.3 Alerting Setup (Prometheus + Alertmanager)

```yaml
# prometheus/alerts.yml
groups:
  - name: trading_harness
    rules:
      - alert: HarnessDown
        expr: up{job="polymarket-harness"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Trading harness is down"

      - alert: HighErrorRate
        expr: rate(harness_errors_total[5m]) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: DailyLossLimit
        expr: harness_daily_pnl < -40
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Approaching daily loss limit"
```

### 5.4 Notification Channels

Configure notifications via:

1. **Email**: SMTP configuration
2. **Slack**: Webhook integration
3. **Telegram**: Bot API
4. **PagerDuty**: Critical alerts

```python
# Example notification function
import httpx

async def send_alert(severity: str, message: str):
    """Send alert to configured channels."""

    # Slack webhook
    if slack_webhook := os.getenv("SLACK_WEBHOOK"):
        color = {"critical": "danger", "warning": "warning", "info": "good"}
        await httpx.post(slack_webhook, json={
            "attachments": [{
                "color": color.get(severity, "good"),
                "title": f"[{severity.upper()}] Trading Harness Alert",
                "text": message,
            }]
        })
```

---

## 6. Maintenance Procedures

### 6.1 Routine Maintenance Schedule

| Frequency | Task | Procedure |
|-----------|------|-----------|
| Daily | Log rotation | Automated via logrotate |
| Weekly | Database optimization | `VACUUM` on SQLite |
| Weekly | Clear expired memory | Memory cleanup script |
| Monthly | Dependency updates | Review and update packages |
| Monthly | Performance review | Analyze metrics and tune |
| Quarterly | Security audit | Review access, rotate keys |
| Quarterly | Full backup verification | Restore test |

### 6.2 Database Maintenance

```bash
# Weekly SQLite optimization
sqlite3 logs/metrics.db "VACUUM;"
sqlite3 data/memory.db "VACUUM;"

# Analyze query performance
sqlite3 logs/metrics.db "ANALYZE;"

# Check integrity
sqlite3 logs/metrics.db "PRAGMA integrity_check;"
```

### 6.3 Memory Cleanup

```python
#!/usr/bin/env python3
"""cleanup_memory.py - Clean old memory entries"""

import asyncio
import aiosqlite
from datetime import datetime, timedelta

async def cleanup_memory(db_path: str, days: int = 90):
    """Remove memories older than specified days."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    async with aiosqlite.connect(db_path) as db:
        # Count before
        async with db.execute("SELECT COUNT(*) FROM memories") as cur:
            before = (await cur.fetchone())[0]

        # Delete old, low-importance entries
        await db.execute("""
            DELETE FROM memories
            WHERE timestamp < ? AND importance < 0.5
        """, (cutoff,))

        await db.commit()

        # Count after
        async with db.execute("SELECT COUNT(*) FROM memories") as cur:
            after = (await cur.fetchone())[0]

        print(f"Cleaned {before - after} entries, {after} remaining")

if __name__ == "__main__":
    asyncio.run(cleanup_memory("data/memory.db", days=90))
```

### 6.4 Log Rotation Configuration

```bash
# /etc/logrotate.d/polymarket-harness
/var/log/polymarket-harness/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 trader trader
    postrotate
        systemctl reload polymarket-harness > /dev/null 2>&1 || true
    endscript
}
```

### 6.5 Dependency Updates

```bash
# Check for updates
pip list --outdated

# Update specific package
pip install --upgrade package_name

# Update all (with caution)
pip install --upgrade -r requirements.txt

# Always test after updates
python -m pytest tests/ -v
```

---

## 7. Backup & Recovery

### 7.1 Backup Strategy

| Data Type | Frequency | Retention | Location |
|-----------|-----------|-----------|----------|
| Configuration | On change | Forever | Git + Cloud |
| SQLite DBs | Daily | 30 days | Cloud storage |
| JSONL logs | Daily | 90 days | Cloud storage |
| Memory DB | Daily | 30 days | Cloud storage |

### 7.2 Backup Script

```bash
#!/bin/bash
# backup.sh - Daily backup script

set -e

BACKUP_DIR="/backups/polymarket-harness"
DATE=$(date +%Y%m%d)
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup databases
sqlite3 logs/metrics.db ".backup '$BACKUP_DIR/metrics_$DATE.db'"
sqlite3 data/memory.db ".backup '$BACKUP_DIR/memory_$DATE.db'"

# Backup logs
tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" logs/

# Backup config (exclude secrets)
cp config.yaml "$BACKUP_DIR/config_$DATE.yaml"

# Upload to cloud (example: AWS S3)
# aws s3 sync "$BACKUP_DIR" s3://your-bucket/backups/

# Clean old backups
find "$BACKUP_DIR" -type f -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $DATE"
```

### 7.3 Recovery Procedures

```bash
# 1. Stop the service
sudo systemctl stop polymarket-harness

# 2. Restore from backup
DATE="20240115"  # Specify backup date
cp /backups/polymarket-harness/metrics_$DATE.db logs/metrics.db
cp /backups/polymarket-harness/memory_$DATE.db data/memory.db

# 3. Verify integrity
sqlite3 logs/metrics.db "PRAGMA integrity_check;"
sqlite3 data/memory.db "PRAGMA integrity_check;"

# 4. Restart service
sudo systemctl start polymarket-harness

# 5. Verify operation
python cli.py status
```

### 7.4 Disaster Recovery

**Complete System Loss:**

1. Provision new server with same specs
2. Install dependencies and clone repository
3. Restore `.env` from secure storage
4. Restore latest backups
5. Verify wallet address matches
6. Start with `--dry-run` to verify
7. Remove `--dry-run` for live trading

**Wallet Compromise:**

1. **IMMEDIATELY** transfer all funds to new wallet
2. Revoke API keys on all platforms
3. Generate new wallet
4. Update all configurations
5. Investigate breach cause

---

## 8. Scaling & Performance

### 8.1 Performance Benchmarks

| Operation | Target | Acceptable | Action if Exceeded |
|-----------|--------|------------|-------------------|
| Cycle Time | < 30s | < 60s | Reduce market count |
| LLM Latency | < 3s | < 10s | Use faster model |
| DB Query | < 100ms | < 500ms | Add indexes |
| Memory Usage | < 4GB | < 6GB | Cleanup old data |

### 8.2 Optimization Strategies

**Reduce LLM Costs:**
```yaml
# Use cheaper models for routine tasks
agents:
  research_model: claude-sonnet-4-20250514
  risk_model: claude-haiku-3-5-20241022     # Cheaper
  execution_model: claude-haiku-3-5-20241022 # Cheaper
  reflection_model: claude-haiku-3-5-20241022 # Cheaper
```

**Reduce API Calls:**
```yaml
# Increase refresh intervals
data:
  polymarket_refresh_seconds: 60      # Was 30
  sentiment_refresh_seconds: 600      # Was 300
  indicator_refresh_seconds: 120      # Was 60
```

**Database Optimization:**
```sql
-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp
ON decisions(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_pnl_timestamp
ON pnl_snapshots(timestamp DESC);
```

### 8.3 Horizontal Scaling

For high-volume operation:

```
┌─────────────────────────────────────────────────────┐
│                  LOAD BALANCER                       │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │Instance │   │Instance │   │Instance │
   │   #1    │   │   #2    │   │   #3    │
   │(Markets │   │(Markets │   │(Markets │
   │  1-50)  │   │ 51-100) │   │101-150) │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        └─────────────┼─────────────┘
                      ▼
              ┌───────────────┐
              │  Shared DB    │
              │  (PostgreSQL) │
              └───────────────┘
```

### 8.4 Resource Limits

```python
# config.yaml additions for resource management
performance:
  max_concurrent_markets: 20
  max_memory_mb: 4096
  max_api_retries: 3
  api_timeout_seconds: 30
  batch_size: 10
```

---

## 9. Security Operations

### 9.1 Security Checklist

- [ ] Private key stored securely (not in git)
- [ ] API keys rotated quarterly
- [ ] Server access via SSH keys only
- [ ] Firewall configured (only required ports)
- [ ] Regular security updates applied
- [ ] Audit logs enabled
- [ ] Principle of least privilege applied

### 9.2 Credential Management

```bash
# Store credentials securely
# Option 1: Encrypted .env
gpg -c .env  # Creates .env.gpg
shred -u .env  # Securely delete plaintext

# Option 2: System keyring
secret-tool store --label='ANTHROPIC_KEY' service polymarket key anthropic

# Option 3: HashiCorp Vault
vault kv put secret/polymarket anthropic_key="sk-..." wallet_key="0x..."
```

### 9.3 Network Security

```bash
# UFW firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw enable

# Fail2ban for SSH protection
sudo apt install fail2ban
sudo systemctl enable fail2ban
```

### 9.4 Audit Logging

```python
# Add to operations for audit trail
import logging

audit_logger = logging.getLogger('audit')
audit_handler = logging.FileHandler('logs/audit.log')
audit_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)

# Log all sensitive operations
audit_logger.info(f"Order placed: {order_id} by {user}")
audit_logger.info(f"Config changed: {change_description}")
audit_logger.info(f"Login from: {ip_address}")
```

---

## 10. Incident Response

### 10.1 Incident Severity Levels

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| P1 | Critical - Trading stopped | 15 min | System down, funds at risk |
| P2 | Major - Degraded operation | 1 hour | LLM failures, high errors |
| P3 | Minor - Reduced capability | 4 hours | One provider down |
| P4 | Low - Cosmetic/logging | 24 hours | Log formatting issues |

### 10.2 Incident Response Procedure

```
1. DETECT
   ├── Automated alerts
   ├── Manual monitoring
   └── User reports

2. ASSESS
   ├── Determine severity
   ├── Identify impact
   └── Notify stakeholders

3. CONTAIN
   ├── Enable kill switch if needed
   ├── Stop affected components
   └── Preserve evidence

4. RESOLVE
   ├── Identify root cause
   ├── Implement fix
   └── Test thoroughly

5. RECOVER
   ├── Gradual restart
   ├── Monitor closely
   └── Verify normal operation

6. REVIEW
   ├── Document timeline
   ├── Identify improvements
   └── Update procedures
```

### 10.3 Emergency Procedures

**Kill Switch Activation:**
```bash
# Immediate stop
sudo systemctl stop polymarket-harness

# Or via config
python -c "
import yaml
config = yaml.safe_load(open('config.yaml'))
config['risk']['kill_switch'] = True
yaml.dump(config, open('config.yaml', 'w'))
"
```

**Emergency Fund Withdrawal:**
```python
# emergency_withdraw.py
import asyncio
from src.wallet.polygon import PolygonWallet

async def emergency_withdraw(to_address: str):
    wallet = PolygonWallet(private_key=os.environ['WALLET_PRIVATE_KEY'])
    await wallet.connect()

    balance = await wallet.get_usdc_balance()
    print(f"Withdrawing {balance} USDC to {to_address}")

    # Confirm before proceeding
    confirm = input("Type 'CONFIRM' to proceed: ")
    if confirm == "CONFIRM":
        tx = await wallet.transfer_usdc(to_address, balance)
        print(f"Transaction: {tx}")
```

### 10.4 Post-Incident Template

```markdown
# Incident Report: [TITLE]

**Date:** YYYY-MM-DD
**Duration:** HH:MM
**Severity:** P1/P2/P3/P4
**Author:** [Name]

## Summary
[Brief description of what happened]

## Timeline
- HH:MM - Incident detected
- HH:MM - Team notified
- HH:MM - Root cause identified
- HH:MM - Fix deployed
- HH:MM - Normal operation restored

## Root Cause
[Detailed explanation]

## Impact
- Trading: [stopped/degraded for X hours]
- Financial: [PnL impact if any]
- Data: [any data loss]

## Resolution
[What was done to fix it]

## Prevention
- [Action item 1]
- [Action item 2]
```

---

## 11. Cost Management

### 11.1 Cost Breakdown

| Category | Estimated Monthly | Notes |
|----------|------------------|-------|
| LLM APIs | $50-500 | Depends on trading frequency |
| Server | $20-100 | Cloud VM |
| Network | $5-20 | Data transfer |
| Storage | $5-10 | Backups |
| Monitoring | $0-20 | Optional services |

### 11.2 LLM Cost Optimization

```yaml
# Tiered model selection based on task complexity
model_selection:
  # Use cheaper models for routine tasks
  quick_checks: claude-haiku-3-5-20241022  # ~$0.001/1k tokens
  analysis: claude-sonnet-4-20250514        # ~$0.003/1k tokens
  complex_decisions: claude-opus-4-20250514 # ~$0.015/1k tokens

# Batch similar requests
batching:
  enabled: true
  max_batch_size: 5
  max_wait_ms: 1000
```

### 11.3 Cost Tracking

```python
# Track API costs in real-time
class CostTracker:
    def __init__(self):
        self.costs = defaultdict(Decimal)

    def record(self, provider: str, model: str, tokens: int):
        rate = self.get_rate(provider, model)
        cost = Decimal(tokens) / 1000 * rate
        self.costs[f"{provider}/{model}"] += cost
        self.costs["total"] += cost

    def daily_report(self) -> str:
        return "\n".join(
            f"{k}: ${v:.4f}"
            for k, v in sorted(self.costs.items())
        )
```

### 11.4 Budget Alerts

```yaml
# config.yaml
cost_limits:
  daily_llm_budget: 20.00
  monthly_llm_budget: 500.00
  alert_threshold_pct: 80
```

---

## 12. Long-Term Sustainability

### 12.1 Knowledge Base

Maintain documentation for:

1. **Runbooks** - Step-by-step procedures
2. **Architecture decisions** - Why choices were made
3. **Lessons learned** - From incidents and reviews
4. **Performance baselines** - Expected behavior

### 12.2 Continuous Improvement Cycle

```
┌──────────────┐
│   MEASURE    │
│  (Metrics)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   ANALYZE    │
│  (Weekly)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   IMPROVE    │
│  (Monthly)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   CONTROL    │
│ (Quarterly)  │
└──────┴───────┘
       │
       └─────────► (Repeat)
```

### 12.3 Monthly Review Checklist

- [ ] Performance vs benchmarks
- [ ] Cost efficiency review
- [ ] Security patch status
- [ ] Dependency updates needed
- [ ] Memory/storage cleanup
- [ ] Strategy effectiveness analysis
- [ ] Model performance comparison
- [ ] Documentation updates

### 12.4 Technology Roadmap

| Quarter | Focus Area | Deliverables |
|---------|-----------|--------------|
| Q1 | Stability | Automated testing, monitoring |
| Q2 | Performance | Optimization, scaling |
| Q3 | Features | New strategies, platforms |
| Q4 | Innovation | ML models, advanced analysis |

### 12.5 Handoff Documentation

For operator transitions:

```markdown
# Handoff Checklist

## Access
- [ ] Server SSH access transferred
- [ ] API key access granted
- [ ] Monitoring dashboards shared
- [ ] Documentation reviewed together

## Knowledge Transfer
- [ ] System walkthrough completed
- [ ] Common issues reviewed
- [ ] Emergency procedures practiced
- [ ] On-call schedule established

## Verification
- [ ] New operator can perform all tasks
- [ ] Contact information updated
- [ ] Escalation paths confirmed
```

---

## Appendix A: Quick Reference

### Common Commands

```bash
# Status
python cli.py status
python cli.py wallet
python cli.py stats

# Control
sudo systemctl start polymarket-harness
sudo systemctl stop polymarket-harness
sudo systemctl restart polymarket-harness

# Logs
tail -f logs/decisions.jsonl | jq .
journalctl -u polymarket-harness -f

# Maintenance
python scripts/cleanup_memory.py
sqlite3 logs/metrics.db "VACUUM;"
```

### Troubleshooting Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| High memory | Restart service, cleanup old data |
| API timeout | Check network, increase timeout |
| No trades | Check kill switch, verify API keys |
| LLM errors | Switch to backup provider |
| DB locked | Restart service, check WAL mode |

### Contact Information

| Role | Contact |
|------|---------|
| Primary On-Call | [Configure] |
| Secondary On-Call | [Configure] |
| Escalation | [Configure] |
| Vendor Support | [API provider contacts] |

---

*Last Updated: [Date]*
*Version: 1.0*
