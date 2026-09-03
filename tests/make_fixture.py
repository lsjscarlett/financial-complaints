"""Generate a schema-faithful synthetic slice of the Kaggle CFPB export.

Used by tests/test_pipeline.py so the pipeline can be exercised without the
167 MB real dataset. Column names, value vocabularies, blank/duplicate patterns
and XXXX-redacted narratives mimic the real file; the text itself is synthetic.

    python tests/make_fixture.py --rows 20000 --output dataset/fixture_complaints.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random

COLUMNS = [
    "date_received", "product", "sub_product", "issue", "sub_issue",
    "consumer_complaint_narrative", "company_public_response", "company", "state",
    "zipcode", "tags", "consumer_consent_provided", "submitted_via",
    "date_sent_to_company", "company_response_to_consumer", "timely_response",
    "consumer_disputed?", "complaint_id",
]

# (product, sub_product, issue, sub_issue, relative weight). Sub-issue may be
# blank, "Other", or a repeat of the issue to mimic the real export.
TAXONOMY = [
    ("Mortgage", "Conventional fixed mortgage", "Loan servicing, payments, escrow account", "", 60),
    ("Mortgage", "FHA mortgage", "Loan modification,collection,foreclosure", "", 55),
    ("Mortgage", "Conventional adjustable mortgage (ARM)", "Application, originator, mortgage broker", "", 15),
    ("Mortgage", "Home equity loan or line of credit", "Settlement process and costs", "", 8),
    ("Debt collection", "Credit card", "Cont'd attempts collect debt not owed", "Debt is not mine", 40),
    ("Debt collection", "Medical", "Cont'd attempts collect debt not owed", "Debt was paid", 22),
    ("Debt collection", "Other (i.e. phone, health club, etc.)", "Cont'd attempts collect debt not owed", "Debt resulted from identity theft", 12),
    ("Debt collection", "Payday loan", "Communication tactics", "Frequent or repeated calls", 25),
    ("Debt collection", "Credit card", "Communication tactics", "Threatened to take legal action", 10),
    ("Debt collection", "Mortgage", "Communication tactics", "Called after sent written cease of comm", 4),
    ("Debt collection", "Auto", "Disclosure verification of debt", "Not given enough info to verify debt", 30),
    ("Debt collection", "Medical", "Disclosure verification of debt", "Right to dispute notice not received", 9),
    ("Debt collection", "Other (i.e. phone, health club, etc.)", "False statements or representation", "Attempted to collect wrong amount", 14),
    ("Debt collection", "Credit card", "False statements or representation", "Impersonated an attorney or official", 2),
    ("Debt collection", "Federal student loan", "Improper contact or sharing of info", "Contacted employer after asked not to", 3),
    ("Debt collection", "Non-federal student loan", "Taking/threatening an illegal action", "Threatened arrest/jail if do not pay", 3),
    ("Debt collection", "Medical", "Cont'd attempts collect debt not owed", "Other", 6),
    ("Credit reporting", "", "Incorrect information on credit report", "Account status", 50),
    ("Credit reporting", "", "Incorrect information on credit report", "Information is not mine", 35),
    ("Credit reporting", "", "Incorrect information on credit report", "Account terms", 12),
    ("Credit reporting", "", "Incorrect information on credit report", "Public record", 8),
    ("Credit reporting", "", "Incorrect information on credit report", "Reinserted previously deleted info", 3),
    ("Credit reporting", "", "Incorrect information on credit report", "Personal information", 6),
    ("Credit reporting", "", "Credit reporting company's investigation", "Investigation took too long", 9),
    ("Credit reporting", "", "Credit reporting company's investigation", "Problem with statement of dispute", 7),
    ("Credit reporting", "", "Credit reporting company's investigation", "Inadequate help over the phone", 5),
    ("Credit reporting", "", "Credit reporting company's investigation", "No notice of investigation status/result", 4),
    ("Credit reporting", "", "Unable to get credit report/credit score", "Problem getting my free annual report", 5),
    ("Credit reporting", "", "Unable to get credit report/credit score", "Problem getting report or credit score", 6),
    ("Credit reporting", "", "Improper use of my credit report", "Report improperly shared by CRC", 3),
    ("Credit reporting", "", "Improper use of my credit report", "Received unsolicited financial offers", 1),
    ("Credit reporting", "", "Credit monitoring or identity protection", "Billing dispute", 2),
    ("Credit reporting", "", "Credit monitoring or identity protection", "Problem cancelling or closing account", 1),
    ("Credit card", "", "Billing disputes", "", 30),
    ("Credit card", "", "Identity theft / Fraud / Embezzlement", "", 12),
    ("Credit card", "", "Closing/Cancelling account", "", 10),
    ("Credit card", "", "Other", "Other", 4),
    ("Bank account or service", "Checking account", "Deposits and withdrawals", "", 28),
    ("Bank account or service", "Checking account", "Account opening, closing, or management", "", 25),
    ("Bank account or service", "Savings account", "Problems caused by my funds being low", "", 10),
    ("Bank account or service", "Other bank product/service", "Using a debit or ATM card", "", 6),
    ("Consumer Loan", "Vehicle loan", "Managing the loan or lease", "", 12),
    ("Consumer Loan", "Installment loan", "Problems when you are unable to pay", "", 7),
    ("Consumer Loan", "Vehicle lease", "Taking out the loan or lease", "", 4),
    ("Student loan", "Non-federal student loan", "Dealing with my lender or servicer", "Received bad information about my loan", 9),
    ("Student loan", "Non-federal student loan", "Dealing with my lender or servicer", "Trouble with how payments are handled", 8),
    ("Student loan", "Non-federal student loan", "Dealing with my lender or servicer", "Need information about my balance/terms", 4),
    ("Student loan", "Federal student loan servicing", "Can't repay my loan", "Can't decrease my monthly payments", 6),
    ("Student loan", "Federal student loan servicing", "Can't repay my loan", "Can't get flexible payment options", 4),
    ("Student loan", "Non-federal student loan", "Can't repay my loan", "Can't temporarily postpone payments", 2),
    ("Payday loan", "", "Charged fees or interest I didn't expect", "", 5),
    ("Payday loan", "", "Can't contact lender", "", 2),
    ("Payday loan", "", "Applied for loan/did not receive money", "", 1),
    ("Money transfers", "International money transfer", "Fraud or scam", "", 3),
    ("Money transfers", "Domestic (US) money transfer", "Money was not available when promised", "", 2),
    ("Money transfers", "International money transfer", "Other transaction issues", "", 1),
    ("Prepaid card", "Gift or merchant gift card", "Unauthorized transactions/trans. issues", "", 2),
    ("Prepaid card", "General purpose card", "Fees", "", 1),
    ("Prepaid card", "Payroll card", "Managing, opening, or closing account", "", 1),
    ("Other financial service", "Debt settlement", "Fraud or scam", "", 1),
    ("Other financial service", "Check cashing", "Customer service/Customer relations", "", 1),
    ("Virtual currency", "Domestic (US) money transfer", "Money was not available when promised", "", 1),
    # Sub-issue that repeats the issue verbatim (should be filtered out).
    ("Credit reporting", "", "Improper use of my credit report", "Improper use of my credit report", 2),
]

SENTENCES = [
    "I have contacted the company several times about this and nobody will help me.",
    "On XX/XX/XXXX I received a letter stating that I owed {amt} which is not correct.",
    "The account was closed in XXXX and I have proof of the payoff.",
    "They keep calling my cell phone XXXX times a day even after I asked them to stop.",
    "My credit report from XXXX shows an account that does not belong to me.",
    "I disputed this with the bureau and they said it was verified but nobody sent me any documents.",
    "The representative told me one thing on the phone and the paperwork said another.",
    "I was charged a fee of {amt} that was never disclosed when I opened the account.",
    "This has hurt my credit score and I was denied a mortgage because of it.",
    "I am asking for this item to be removed and for written confirmation.",
    "The loan was sold to XXXX and now both companies are asking for payment.",
    "I made every payment on time and they still reported me 30 days late.",
    "Please investigate and provide the original signed contract.",
    "I have been a customer for over XXXX years and never had a problem until now.",
]

COMPANIES = ["Bank of America", "Wells Fargo & Company", "Equifax", "Experian", "TransUnion Intermediate Holdings, Inc.", "JPMorgan Chase & Co.", "Citibank", "Navient Solutions, Inc.", "Ocwen", "Encore Capital Group", "Synchrony Financial", "Capital One", "Nationstar Mortgage", "Portfolio Recovery Associates, Inc.", "Ally Financial Inc."]
STATES = ["CA", "TX", "FL", "NY", "GA", "NJ", "IL", "PA", "OH", "NC", "VA", "MD", "AZ", "WA", "MA"]
RESPONSES = ["Closed with explanation", "Closed with non-monetary relief", "Closed with monetary relief", "Closed without relief", "Closed", "In progress", "Untimely response"]
TAGS = ["", "", "", "", "Older American", "Servicemember", "Older American, Servicemember"]


def narrative(rng: random.Random) -> str:
    n = rng.choice([1, 2, 3, 4, 5, 6, 8, 12])
    parts = [rng.choice(SENTENCES).format(amt=f"${rng.randint(50, 9000)}.00") for _ in range(n)]
    text = " ".join(parts)
    if rng.random() < 0.08:  # a heavily redacted narrative
        text = " ".join("XXXX" if rng.random() < 0.5 else w for w in text.split())
    return text


def make_rows(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    weights = [t[4] for t in TAXONOMY]
    rows = []
    narratives_pool: list[str] = []
    for i in range(n):
        product, sub_product, issue, sub_issue, _ = rng.choices(TAXONOMY, weights=weights, k=1)[0]
        # Real export: sub_issue blank for a large share even where it exists.
        if sub_issue and rng.random() < 0.12:
            sub_issue = ""
        year = rng.choice([2013, 2014, 2015, 2015, 2016, 2016])
        month, day = rng.randint(1, 12), rng.randint(1, 28)
        has_narr = year >= 2015 and rng.random() < 0.22
        if has_narr:
            if narratives_pool and rng.random() < 0.03:
                text = rng.choice(narratives_pool)  # duplicate submission
            else:
                text = narrative(rng)
                narratives_pool.append(text)
            consent = "Consent provided"
        else:
            text, consent = "", rng.choice(["", "Consent not provided", "Other"])
        rows.append({
            "date_received": f"{month:02d}/{day:02d}/{year}",
            "product": product,
            "sub_product": sub_product,
            "issue": issue,
            "sub_issue": sub_issue,
            "consumer_complaint_narrative": text,
            "company_public_response": rng.choice(["", "", "Company chooses not to provide a public response", "Company believes it acted appropriately as authorized by contract or law"]),
            "company": rng.choice(COMPANIES),
            "state": rng.choice(STATES),
            "zipcode": f"{rng.randint(100, 999)}XX",
            "tags": rng.choice(TAGS),
            "consumer_consent_provided": consent,
            "submitted_via": rng.choice(["Web", "Web", "Web", "Referral", "Phone", "Postal mail", "Fax"]),
            "date_sent_to_company": f"{month:02d}/{min(day + 2, 28):02d}/{year}",
            "company_response_to_consumer": rng.choice(RESPONSES),
            "timely_response": rng.choice(["Yes", "Yes", "Yes", "No"]),
            "consumer_disputed?": rng.choice(["Yes", "No", "No", "No", ""]),
            "complaint_id": str(1_000_000 + i),
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "fixture_complaints.csv"))
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    rows = make_rows(args.rows, args.seed)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):,} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
