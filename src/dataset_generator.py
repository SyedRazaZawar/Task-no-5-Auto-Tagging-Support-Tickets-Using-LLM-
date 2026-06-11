import os
import random
import pandas as pd
from typing import List, Tuple
import sys

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set random seed for reproducibility
random.seed(config.SEED)

# Define templates for each category
TEMPLATES = {
    "Billing & Payments": [
        "Hi, I noticed a double charge of ${amount} on my credit card statement for my subscription. Please refund the extra charge.",
        "Hello, my credit card payment failed today. How can I update my billing information to prevent my account from being suspended?",
        "Can I get an invoice for my payment on {date}? Our finance department needs it for accounting purposes.",
        "I would like to cancel my premium subscription and revert to the free plan. Please confirm the cancellation details.",
        "Is there a annual discount option? I am currently on the monthly plan and want to switch to save some costs.",
        "I received a charge of ${amount} even though I cancelled my account last week. Please reverse this immediately.",
        "My subscription shows as inactive even though I paid yesterday. Here is my transaction ID: {tx_id}. Please resolve this."
    ],
    "Technical Support": [
        "The web application keeps crashing whenever I try to export my report. It displays a white screen and says 'Application Error'.",
        "I am getting a 500 Internal Server Error when clicking on the sync button. Can you look into this? I've tried clearing my cache.",
        "The Slack integration is not working. Messages are not being sent to the designated channel. Is the webhook url wrong?",
        "Our API calls are getting timed out repeatedly today. Are your servers experiencing downtime or high load?",
        "The mobile app is extremely slow to load today. It hangs on the login screen for over a minute before giving an connection error.",
        "I updated my system yesterday, and now the desktop agent won't launch. It fails with error code {error_code}.",
        "Some images are not rendering on my dashboard. They show up as broken image links. Please check the attachment service."
    ],
    "Account Access": [
        "I forgot my password and the password reset link is not arriving in my inbox. Can you reset it manually or resend the link?",
        "My account has been locked due to too many failed login attempts. Can you unlock my email {email}?",
        "I lost access to my multi-factor authentication (MFA) device. Can you disable MFA temporarily so I can log in and reset it?",
        "I am trying to log in but it keeps redirecting me back to the login page without any error message. Please help.",
        "We need to transfer the owner role of our organization account from {email} to {email2}. How can we do this?",
        "I want to change the primary email address associated with my account to {email}. The option is greyed out in settings.",
        "I am getting an 'unauthorized access' error when trying to view our team folder, even though I am an admin."
    ],
    "Product Feedback": [
        "The new interface looks absolutely beautiful! Great job on the redesign, it is much cleaner and easier to navigate.",
        "I find the new navigation sidebar to be quite confusing. The previous layout was much more intuitive for daily tasks.",
        "The search function is excellent, but it would be even better if it cached recent search terms.",
        "I am really enjoying the new dashboard widgets. It has made tracking our metrics much easier. Thank you!",
        "The latest update removed a shortcut I used every day. I wish there was a way to customize keyboard shortcuts.",
        "Just wanted to say that your support team was incredibly helpful yesterday. Quick and friendly response!",
        "The tool is good but the documentation lacks clear examples for advanced configurations. Please add some tutorials."
    ],
    "Feature Request": [
        "It would be amazing if we could export our dashboard data directly to a PDF report instead of just CSV.",
        "Are there any plans to implement a dark mode? My eyes get really tired reading the screen during night shifts.",
        "We really need an integration with Jira so our support tickets can be automatically pushed to our developer backlog.",
        "Please add a bulk edit feature to the tables so we can change the status of multiple items at once.",
        "Could you support SSO (Single Sign-On) via Okta? Our enterprise security policy requires this for all tools.",
        "It would be great to have role-based access control (RBAC) where we can define custom permissions for viewers and editors.",
        "Can you add support for webhook notifications? We want to trigger external scripts when a ticket status changes."
    ],
    "Data & Privacy": [
        "Under GDPR regulations, I would like to request a complete export of all personal data your company holds about me.",
        "Please delete my account and permanently purge all my data, including my chat logs and file attachments.",
        "Where are your servers located and is the data encrypted at rest? Our security team needs this for compliance review.",
        "I need to sign a Data Processing Agreement (DPA) with your company. Who should I contact to get the document signed?",
        "I noticed some suspicious login attempts on my account. Can you share the IP log audit for the last 30 days?",
        "Is your service SOC 2 compliant? We need to verify your compliance certifications for our annual security audit.",
        "I want to opt out of all promotional and tracking cookies. The cookie banner doesn't save my preferences."
    ]
}

# Values to populate templates
EMAILS = ["john.doe@example.com", "sarah.smith@company.org", "dev-ops@startup.io", "billing@enterprise.co"]
DATES = ["2026-06-01", "2026-05-15", "last Tuesday", "yesterday morning"]
AMOUNTS = ["49.00", "199.00", "12.50", "999.00"]
TX_IDS = ["TXN-98231", "TXN-00129", "PAY-88271", "CHG-7716A"]
ERROR_CODES = ["0x80070005", "ERR_CONNECTION_REFUSED", "Code 403-Forbidden", "Exception in thread 'main'"]

def generate_random_ticket(primary_category: str) -> Tuple[str, List[str]]:
    """
    Generates a single support ticket and lists all applicable tags.
    Sometimes adds crossover elements (e.g. billing + technical error) for multi-tag tests.
    """
    template = random.choice(TEMPLATES[primary_category])
    
    # Fill in template placeholders
    text = template.format(
        amount=random.choice(AMOUNTS),
        date=random.choice(DATES),
        tx_id=random.choice(TX_IDS),
        error_code=random.choice(ERROR_CODES),
        email=random.choice(EMAILS),
        email2=random.choice(EMAILS)
    )
    
    all_tags = [primary_category]
    
    # Introduce secondary category crossover for 20% of tickets
    crossover = random.random()
    if crossover < 0.20:
        if primary_category == "Billing & Payments":
            # Crossover to Technical Support (e.g. portal broken while paying)
            text += " Also, when I tried to pay, the screen froze and gave a database error."
            all_tags.append("Technical Support")
        elif primary_category == "Technical Support":
            # Crossover to Feature Request or Account Access
            if random.choice([True, False]):
                text += " This issue started right after I tried to update my password in my account settings."
                all_tags.append("Account Access")
            else:
                text += " By the way, while fixing this, could you also tell me if you support Okta SSO?"
                all_tags.append("Feature Request")
        elif primary_category == "Account Access":
            # Crossover to Data & Privacy (MFA / GDPR)
            text += " I need to resolve this quickly because our security team is auditing our user list for GDPR compliance."
            all_tags.append("Data & Privacy")
        elif primary_category == "Feature Request":
            # Crossover to Product Feedback
            text += " The current dashboard is good, but adding this feature would make it perfect."
            all_tags.append("Product Feedback")
            
    return text, all_tags

def create_synthetic_dataset(num_samples_per_category: int = 40) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates full dataset and splits into train/test sets.
    """
    data = []
    
    for category in config.CATEGORIES:
        for _ in range(num_samples_per_category):
            text, tags = generate_random_ticket(category)
            data.append({
                "text": text,
                "primary_tag": category,
                "all_tags": ", ".join(tags)
            })
            
    df = pd.DataFrame(data)
    # Shuffle dataset
    df = df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    
    # Split train/test
    split_idx = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    return train_df, test_df

def save_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Saves generated dataset to config paths"""
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)
    print(f"Dataset generated successfully!")
    print(f"Saved {len(train_df)} training samples to: {config.TRAIN_DATA_PATH}")
    print(f"Saved {len(test_df)} testing samples to: {config.TEST_DATA_PATH}")

if __name__ == "__main__":
    train, test = create_synthetic_dataset(num_samples_per_category=50) # 300 total samples
    save_dataset(train, test)
