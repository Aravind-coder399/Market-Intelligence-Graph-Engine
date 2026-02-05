"""
Backtesting Module for Market Intelligence
Test prediction accuracy against historical data.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class Backtester:
    """Backtest prediction strategies against historical data."""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.results = []
    
    def run_backtest(self, test_data: List[Dict], 
                     start_date: str = None, 
                     end_date: str = None) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            test_data: List of historical news items with actual returns
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Backtest results with accuracy metrics
        """
        logger.info(f"Starting backtest with {len(test_data)} items...")
        
        correct_predictions = 0
        total_predictions = 0
        predictions = []
        
        for item in test_data:
            headline = item.get('headline')
            ticker = item.get('ticker')
            actual_d1 = item.get('day_1_return') or item.get('d1_return')
            actual_d7 = item.get('day_7_return') or item.get('d7_return')
            
            if not headline or actual_d1 is None:
                continue
            
            # Run prediction
            try:
                result = self.predictor.predict(headline, ticker=ticker)
                predicted_d1 = result.get('predicted_d1', 0)
                
                # Determine direction accuracy
                actual_direction = 'POSITIVE' if actual_d1 > 0 else 'NEGATIVE' if actual_d1 < 0 else 'NEUTRAL'
                predicted_direction = result.get('sentiment', 'NEUTRAL')
                
                is_correct = (
                    (actual_direction == 'POSITIVE' and predicted_direction == 'POSITIVE') or
                    (actual_direction == 'NEGATIVE' and predicted_direction == 'NEGATIVE') or
                    (actual_direction == 'NEUTRAL' and predicted_direction == 'NEUTRAL')
                )
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                predictions.append({
                    'headline': headline[:50] + '...',
                    'ticker': ticker,
                    'actual_d1': actual_d1,
                    'predicted_d1': predicted_d1,
                    'actual_direction': actual_direction,
                    'predicted_direction': predicted_direction,
                    'correct': is_correct,
                    'error': abs(actual_d1 - predicted_d1)
                })
                
            except Exception as e:
                logger.warning(f"Backtest error for {headline[:30]}: {e}")
        
        # Calculate metrics
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        mae = np.mean([p['error'] for p in predictions]) if predictions else 0
        
        self.results = predictions
        
        summary = {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': round(accuracy, 2),
            'mean_absolute_error': round(mae, 2),
            'predictions': predictions
        }
        
        logger.info(f"Backtest complete: {accuracy:.1f}% accuracy ({correct_predictions}/{total_predictions})")
        
        return summary
    
    def run_backtest_from_graph(self, limit: int = 15) -> Dict:
        """Run backtest using data from the knowledge graph.
        
        Uses a smaller default limit (15) for faster execution.
        """
        try:
            # Fetch historical data from graph - ONLY items with return data
            query = f"""
            MATCH (n:News)
            WHERE n.day_1_return IS NOT NULL AND n.ticker IS NOT NULL
            RETURN n.headline, n.ticker, n.day_1_return, n.day_7_return, n.date
            ORDER BY rand()
            LIMIT {limit}
            """
            
            data = self.predictor.falkor.query(query)
            
            if not data:
                return {'error': 'No historical data with returns found', 'total_predictions': 0}
            
            test_data = []
            for row in data:
                test_data.append({
                    'headline': row[0],
                    'ticker': row[1],
                    'd1_return': row[2],
                    'd7_return': row[3],
                    'date': row[4]
                })
            
            return self.run_backtest(test_data)
            
        except Exception as e:
            logger.error(f"Backtest from graph failed: {e}")
            return {'error': str(e), 'total_predictions': 0}

    
    def get_confusion_matrix(self) -> Dict:
        """Generate confusion matrix from backtest results."""
        if not self.results:
            return {}
        
        matrix = {
            'true_positive': 0,  # Predicted positive, was positive
            'true_negative': 0,  # Predicted negative, was negative
            'false_positive': 0,  # Predicted positive, was negative
            'false_negative': 0   # Predicted negative, was positive
        }
        
        for p in self.results:
            actual = p['actual_direction']
            predicted = p['predicted_direction']
            
            if predicted == 'POSITIVE' and actual == 'POSITIVE':
                matrix['true_positive'] += 1
            elif predicted == 'NEGATIVE' and actual == 'NEGATIVE':
                matrix['true_negative'] += 1
            elif predicted == 'POSITIVE' and actual == 'NEGATIVE':
                matrix['false_positive'] += 1
            elif predicted == 'NEGATIVE' and actual == 'POSITIVE':
                matrix['false_negative'] += 1
        
        # Calculate additional metrics
        tp, tn, fp, fn = matrix['true_positive'], matrix['true_negative'], matrix['false_positive'], matrix['false_negative']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        matrix['precision'] = round(precision, 3)
        matrix['recall'] = round(recall, 3)
        matrix['f1_score'] = round(f1, 3)
        
        return matrix
    
    def generate_report(self) -> str:
        """Generate a markdown report of backtest results."""
        if not self.results:
            return "No backtest results available. Run a backtest first."
        
        total = len(self.results)
        correct = sum(1 for p in self.results if p['correct'])
        accuracy = (correct / total * 100) if total > 0 else 0
        
        matrix = self.get_confusion_matrix()
        
        report = f"""# Backtest Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Predictions**: {total}
- **Correct Predictions**: {correct}
- **Accuracy**: {accuracy:.1f}%
- **Mean Absolute Error**: {np.mean([p['error'] for p in self.results]):.2f}%

## Confusion Matrix
| | Actual Positive | Actual Negative |
|---|---|---|
| **Predicted Positive** | {matrix.get('true_positive', 0)} | {matrix.get('false_positive', 0)} |
| **Predicted Negative** | {matrix.get('false_negative', 0)} | {matrix.get('true_negative', 0)} |

## Metrics
- **Precision**: {matrix.get('precision', 0):.1%}
- **Recall**: {matrix.get('recall', 0):.1%}
- **F1 Score**: {matrix.get('f1_score', 0):.1%}

## Interpretation
"""
        
        if accuracy >= 60:
            report += "✅ The model shows statistically significant predictive power.\n"
        elif accuracy >= 50:
            report += "⚠️ The model is near random chance. Consider adding more features.\n"
        else:
            report += "❌ The model performs worse than random. Strategy needs revision.\n"
        
        return report
