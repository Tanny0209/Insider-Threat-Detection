import pandas as pd
import re
from datetime import datetime

ORG_DOMAIN = "enron.com"  # set your enterprise domain here

def parse_email_date(date_str):
    """Convert email date string to datetime object."""
    try:
        return pd.to_datetime(date_str, errors="coerce")
    except Exception:
        return None

def extract_domain(email):
    """Extract domain from email address."""
    if pd.isna(email) or "@" not in str(email):
        return None
    return email.split("@")[-1].strip().lower()

def detect_attachments(text):
    """Simulate attachment detection by keywords and extensions."""
    if pd.isna(text):
        return 0, []
    patterns = re.findall(r"\b\w+\.(pdf|docx?|xls[xm]?|pptx?|zip)\b", text, re.IGNORECASE)
    return len(patterns), list(set([p.lower() for p in patterns]))

def count_links(text):
    """Count links in the email body."""
    if pd.isna(text):
        return 0
    return len(re.findall(r"http[s]?://\S+", text))

def process_dataset(input_file, output_file="datasets/processed_emails.csv"):
    df = pd.read_csv(input_file)

    processed = []

    for _, row in df.iterrows():
        msg = row["message"]

        # Extract headers
        from_addr = re.search(r"From: (.+)", msg)
        to_addr = re.search(r"To: (.+)", msg)
        cc_addr = re.search(r"Cc: (.+)", msg, re.IGNORECASE)
        bcc_addr = re.search(r"Bcc: (.+)", msg, re.IGNORECASE)
        subject = re.search(r"Subject: (.*)", msg)
        date = re.search(r"Date: (.+)", msg)

        # Extract body (everything after the headers block)
        body_split = msg.split("\n\n", 1)
        body = body_split[1].strip() if len(body_split) > 1 else ""

        # Parse time
        timestamp = parse_email_date(date.group(1).strip() if date else None)

        # Derived features
        sender_domain = extract_domain(from_addr.group(1).strip()) if from_addr else None
        receiver_domains = [extract_domain(x.strip()) for x in (to_addr.group(1).split(",") if to_addr else [])]
        num_recipients = len(receiver_domains) + (len(cc_addr.group(1).split(",")) if cc_addr else 0) + (len(bcc_addr.group(1).split(",")) if bcc_addr else 0)

        is_external_sender = sender_domain != ORG_DOMAIN if sender_domain else None
        is_external_recipient = any(d != ORG_DOMAIN for d in receiver_domains if d)

        attachments_count, attachment_types = detect_attachments(body + " " + (subject.group(1) if subject else ""))
        links_count = count_links(body)

        processed.append({
            "message_id": row["file"],
            "date": date.group(1).strip() if date else None,
            "from": from_addr.group(1).strip() if from_addr else None,
            "to": to_addr.group(1).strip() if to_addr else None,
            "cc": cc_addr.group(1).strip() if cc_addr else None,
            "bcc": bcc_addr.group(1).strip() if bcc_addr else None,
            "subject": subject.group(1).strip() if subject else None,
            "body": body,
            "timestamp": timestamp,
            "day_of_week": timestamp.day_name() if timestamp is not pd.NaT else None,
            "hour_of_day": timestamp.hour if timestamp is not pd.NaT else None,
            "is_weekend": timestamp.weekday() >= 5 if timestamp is not pd.NaT else None,
            "sender_domain": sender_domain,
            "receiver_domains": receiver_domains,
            "num_recipients": num_recipients,
            "is_external_sender": is_external_sender,
            "is_external_recipient": is_external_recipient,
            "attachments_count": attachments_count,
            "attachment_types": attachment_types,
            "links_count": links_count,
            "body_length": len(body),
            "subject_length": len(subject.group(1)) if subject else 0
        })

    out_df = pd.DataFrame(processed)
    out_df.to_csv(output_file, index=False)
    print(f"✅ Processed dataset saved to {output_file}")
    return out_df


if __name__ == "__main__":
    process_dataset("datasets/emails.csv")
