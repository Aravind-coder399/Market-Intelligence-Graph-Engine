"""
Alert System for Market Intelligence
Monitor for specific patterns and notify when similar news appears.
"""
import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class AlertManager:
    """Manage pattern-based alerts for news monitoring."""
    
    def __init__(self, alerts_file: str = "alerts.json"):
        self.alerts_file = alerts_file
        self.alerts = self._load_alerts()
        
        # Email config from env
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        self.alert_email = os.getenv("ALERT_EMAIL")
    
    def _load_alerts(self) -> List[Dict]:
        """Load alerts from file."""
        try:
            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")
        return []
    
    def _save_alerts(self):
        """Save alerts to file."""
        try:
            with open(self.alerts_file, 'w') as f:
                json.dump(self.alerts, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")
    
    def add_alert(self, pattern: str, ticker: str = None, threshold: float = 0.7, 
                  notify_email: bool = True) -> Dict:
        """Add a new alert pattern to monitor."""
        alert = {
            "id": len(self.alerts) + 1,
            "pattern": pattern,
            "ticker": ticker,
            "threshold": threshold,
            "notify_email": notify_email,
            "created_at": datetime.now().isoformat(),
            "triggered_count": 0,
            "last_triggered": None,
            "active": True
        }
        self.alerts.append(alert)
        self._save_alerts()
        logger.info(f"Alert added: {pattern[:50]}...")
        return alert
    
    def remove_alert(self, alert_id: int) -> bool:
        """Remove an alert by ID."""
        for i, alert in enumerate(self.alerts):
            if alert["id"] == alert_id:
                self.alerts.pop(i)
                self._save_alerts()
                return True
        return False
    
    def get_alerts(self, active_only: bool = True) -> List[Dict]:
        """Get all alerts."""
        if active_only:
            return [a for a in self.alerts if a.get("active", True)]
        return self.alerts
    
    def check_alerts(self, predictor, news_items: List[Dict]) -> List[Dict]:
        """Check incoming news against all active alerts."""
        triggered = []
        
        for news in news_items:
            headline = news.get("headline", "")
            ticker = news.get("ticker")
            
            for alert in self.get_alerts():
                # Skip if alert is ticker-specific and doesn't match
                if alert.get("ticker") and alert["ticker"] != ticker:
                    continue
                
                # Calculate similarity between alert pattern and news
                try:
                    similarity = self._calculate_similarity(
                        predictor, alert["pattern"], headline
                    )
                    
                    if similarity >= alert["threshold"]:
                        triggered_info = {
                            "alert": alert,
                            "news": news,
                            "similarity": similarity,
                            "triggered_at": datetime.now().isoformat()
                        }
                        triggered.append(triggered_info)
                        
                        # Update alert stats
                        alert["triggered_count"] += 1
                        alert["last_triggered"] = datetime.now().isoformat()
                        
                        # Send notification
                        if alert.get("notify_email"):
                            self._send_email_alert(triggered_info)
                        
                        logger.info(f"Alert triggered: {alert['pattern'][:30]}... matched {headline[:30]}...")
                        
                except Exception as e:
                    logger.error(f"Error checking alert: {e}")
        
        self._save_alerts()
        return triggered
    
    def _calculate_similarity(self, predictor, pattern: str, headline: str) -> float:
        """Calculate semantic similarity between pattern and headline."""
        try:
            # Use the predictor's encoder for embeddings
            emb1 = predictor.encoder.encode(pattern)
            emb2 = predictor.encoder.encode(headline)
            
            # Cosine similarity
            import numpy as np
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except:
            return 0.0
    
    def _send_email_alert(self, triggered_info: Dict):
        """Send email notification for triggered alert."""
        if not self.smtp_user or not self.alert_email:
            logger.warning("Email not configured - skipping notification")
            return
        
        try:
            alert = triggered_info["alert"]
            news = triggered_info["news"]
            
            subject = f"🚨 Market Alert: Similar news detected for {news.get('ticker', 'Unknown')}"
            body = f"""
Alert Pattern: {alert['pattern']}

Matched News: {news.get('headline')}
Ticker: {news.get('ticker', 'N/A')}
Source: {news.get('source', 'N/A')}
Date: {news.get('date', 'N/A')}

Similarity Score: {triggered_info['similarity']:.2%}
Triggered At: {triggered_info['triggered_at']}

---
Market Intelligence Graph Engine
            """
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.smtp_user
            msg['To'] = self.alert_email
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
            
            logger.info(f"Alert email sent to {self.alert_email}")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")


# Singleton instance
_alert_manager = None

def get_alert_manager() -> AlertManager:
    """Get or create global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
