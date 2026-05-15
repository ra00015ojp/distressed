"""
Signal 5 Automation Demo & Testing Script
==========================================
Demonstrates gate event detection without Streamlit.
Run this to validate the module and see real SEC filing scans.

Usage:
    python signal_5_demo.py
"""

import sys
from signal_5_gate_automation import (
    scan_sec_for_gate_events,
    fetch_gate_news,
    detect_metric_acceleration,
    compute_ytd_gate_events,
    signal_5_status,
)
from datetime import datetime

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def demo_sec_filing_scan():
    """Demonstrate SEC EDGAR 8-K scanning."""
    print_header("DEMO 1: SEC EDGAR 8-K Filing Scan")
    
    print("Scanning for gate events in recent 8-K filings...")
    print("(First run will be slow; subsequent runs use 24-hour cache)\n")
    
    try:
        sec_gates = scan_sec_for_gate_events(days_back=90)
        
        if sec_gates.empty:
            print("✅ No gate events detected in recent 8-K filings.")
            print("   (This is actually a GOOD sign — means cycle stress is low)")
        else:
            print(f"⚠️  Found {len(sec_gates)} potential gate event(s):\n")
            
            for idx, (_, row) in enumerate(sec_gates.iterrows(), 1):
                print(f"  {idx}. {row['date'].strftime('%Y-%m-%d')} | {row['ticker']}")
                print(f"     SEC Filing: {row['accession']}")
                print(f"     URL: {row['filing_url']}\n")
    
    except Exception as e:
        print(f"❌ Error scanning SEC EDGAR: {e}")
        print("   (Check internet connection; SEC EDGAR may be temporarily unavailable)")


def demo_news_corroboration():
    """Demonstrate news API corroboration (requires API key)."""
    print_header("DEMO 2: News API Corroboration (Optional)")
    
    api_key = input("Enter NewsAPI key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("⏭️  Skipping news layer demo.\n")
        return
    
    print("Fetching gate event mentions from financial news...\n")
    
    try:
        news_gates = fetch_gate_news(api_key=api_key, days_back=30)
        
        if news_gates.empty:
            print("✅ No BDC gating news found in last 30 days.")
        else:
            print(f"Found {len(news_gates)} news articles mentioning BDC gates:\n")
            
            for idx, (_, row) in enumerate(news_gates.iterrows(), 1):
                print(f"  {idx}. {row['date'].strftime('%Y-%m-%d')} | {row['source']}")
                print(f"     Headline: {row['headline']}")
                print(f"     Tickers: {', '.join(row['tickers'])}")
                print(f"     URL: {row['url']}\n")
    
    except Exception as e:
        print(f"❌ Error fetching news: {e}")


def demo_metric_acceleration():
    """Demonstrate leading indicator detection."""
    print_header("DEMO 3: BDC Metric Acceleration Detection (Leading Indicator)")
    
    print("Testing PIK ratio and non-accrual acceleration detection...\n")
    
    # Test cases
    test_cases = [
        {
            "name": "Normal conditions",
            "current_pik": 7.5,
            "prev_pik": 7.4,
            "current_non_accrual": 2.0,
            "prev_non_accrual": 1.9,
        },
        {
            "name": "Moderate stress (PIK rising)",
            "current_pik": 9.2,
            "prev_pik": 8.8,
            "current_non_accrual": 3.5,
            "prev_non_accrual": 3.2,
        },
        {
            "name": "HIGH STRESS (PIK + Non-Accrual accelerating)",
            "current_pik": 12.5,
            "prev_pik": 10.0,
            "current_non_accrual": 6.8,
            "prev_non_accrual": 4.5,
        },
    ]
    
    for test in test_cases:
        name = test.pop("name")
        result = detect_metric_acceleration(**test)
        
        risk_emoji = {
            "HIGH": "🔴",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }[result["risk_level"]]
        
        print(f"{risk_emoji} {name}")
        print(f"   PIK change: {result['pik_change_pct']:+.2f}%")
        print(f"   Non-Accrual change: {result['non_accrual_change_pct']:+.2f}%")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Acceleration detected: {result['acceleration_detected']}\n")


def demo_full_orchestration():
    """Demonstrate full gate detection orchestration."""
    print_header("DEMO 4: Full Gate Detection Orchestration")
    
    print("Computing YTD gate events using all three layers...\n")
    
    # Sample BDC metric data
    manual_pik = {
        "ARCC": {
            "current_pik": 8.5,
            "prev_pik": 7.8,
            "current_non_accrual": 3.2,
            "prev_non_accrual": 2.8,
        },
        "FSK": {
            "current_pik": 12.1,
            "prev_pik": 10.5,
            "current_non_accrual": 5.8,
            "prev_non_accrual": 4.2,
        },
        "OBDC": {
            "current_pik": 6.3,
            "prev_pik": 6.1,
            "current_non_accrual": 1.9,
            "prev_non_accrual": 1.8,
        },
    }
    
    print("Using sample metric data (you would fetch from 10-Q filings):\n")
    for ticker, metrics in manual_pik.items():
        print(f"  {ticker}: PIK {metrics['current_pik']:.1f}% "
              f"(was {metrics['prev_pik']:.1f}%), "
              f"Non-Accrual {metrics['current_non_accrual']:.1f}%")
    
    print("\nComputing gate count...\n")
    
    try:
        result = compute_ytd_gate_events(manual_pik_metrics=manual_pik)
        
        print(f"Gate Count (YTD): {result['gate_count']}")
        print(f"Confidence: {result['confidence']:.0f}%")
        print(f"Last Updated: {result['last_updated']}\n")
        
        if result["gate_events"]:
            print(f"Detected {len(result['gate_events'])} unique gate event(s):\n")
            
            for event in sorted(result["gate_events"], key=lambda x: x["date"], reverse=True):
                source = event.get("source", "Unknown")
                conf = event.get("confidence", 0)
                print(f"  • {event['date'].strftime('%Y-%m-%d')} | {event['ticker']} | "
                      f"{source} ({conf}% confidence)")
                
                if "note" in event:
                    print(f"    Note: {event['note']}")
        else:
            print("✅ No gate events detected from SEC or news layers.")
        
        print("\nLeading Indicator Risk Summary:\n")
        
        for ticker, risk in result["leading_indicators"].items():
            risk_emoji = {
                "HIGH": "🔴",
                "MEDIUM": "🟡",
                "LOW": "🟢"
            }[risk["risk_level"]]
            
            print(f"{risk_emoji} {ticker}: {risk['risk_level']}")
            print(f"   PIK change: {risk['pik_change_pct']:+.2f}%")
            print(f"   Non-Accrual change: {risk['non_accrual_change_pct']:+.2f}%")
            print(f"   Prediction: ", end="")
            
            if risk["risk_level"] == "HIGH":
                print("Gating likely within 4–8 weeks")
            elif risk["risk_level"] == "MEDIUM":
                print("Monitor closely; gating possible")
            else:
                print("Low gating risk")
            print()
        
        # Evaluate signal status
        print("\n--- SIGNAL 5 STATUS EVALUATION ---\n")
        
        icon, state, detail = signal_5_status(
            result["gate_count"],
            result["leading_indicators"],
            result["confidence"]
        )
        
        print(f"{icon} State: {state}")
        print(f"Detail: {detail}")
        
    except Exception as e:
        print(f"❌ Error computing gate events: {e}")
        import traceback
        traceback.print_exc()


def demo_monthly_validation():
    """Show how to validate automated results against Moody's data."""
    print_header("DEMO 5: Monthly Validation Procedure")
    
    print("""
When Moody's publishes the monthly default rate report (~5th of each month),
validate the automated gate count like this:

1. Go to https://www.moodysanalytics.com/research/report
2. Download "Speculative-Grade Default Rate" monthly report (free PDF)
3. Note the "12-month trailing speculative-grade default rate" (%)
4. Compare to your dashboard automated reading

Example validation:

    Moody's Data (Mar 2026):
    ├─ Default Rate: 3.2%
    ├─ Gate Events Mentioned: 4 in the month
    └─ Trend: "Accelerating"
    
    Your Dashboard (Mar 2026):
    ├─ Computed Default Rate: 3.1%
    ├─ Detected Gate Events: 4 YTD
    └─ Leading Indicators: 2 BDCs at HIGH risk
    
    ✅ Validation: +0.1% difference is excellent alignment

Alignment benchmarks:

    Difference < 0.5%: ✅ Excellent
    Difference 0.5–1.0%: ✅ Good
    Difference > 1.0%: ⚠️ Investigate discrepancy

""")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("   SIGNAL 5 GATE EVENTS AUTOMATION — DEMO & VALIDATION")
    print("="*70)
    
    print("""
This script demonstrates the three-layer gate detection system:

  1. SEC EDGAR 8-K Filing Scan (Authoritative)
  2. News API Corroboration (Confirmatory)
  3. BDC Metric Acceleration (Leading Indicator)

All data is free and public. Let's run through each layer...
""")
    
    try:
        # Demo 1: SEC Filing Scan
        demo_sec_filing_scan()
        
        # Demo 2: News Corroboration
        demo_news_corroboration()
        
        # Demo 3: Metric Acceleration
        demo_metric_acceleration()
        
        # Demo 4: Full Orchestration
        demo_full_orchestration()
        
        # Demo 5: Validation Procedure
        demo_monthly_validation()
        
        print_header("Demo Complete")
        
        print("""
✅ Signal 5 automation is ready for integration into your dashboard!

Next steps:

  1. Copy signal_5_gate_automation.py to your project directory
  2. Follow SIGNAL_5_INTEGRATION_GUIDE.md to integrate into dashboard
  3. Add your NewsAPI key (optional) in the dashboard sidebar
  4. Monitor monthly Moody's reports to validate accuracy

For questions or issues, refer to DEFAULT_RATE_AUTOMATION_METHOD.md
and SIGNAL_5_INTEGRATION_GUIDE.md.
""")
    
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user.")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
