"""One-off script: generates a real, 50+ page, multi-section PDF
('data/pdfs/employee_handbook.pdf') with headings, paragraphs, and a table,
so the ingestion/chunking pipeline has real heterogeneous structure to work
with -- not just a single flat wall of text."""

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

doc = SimpleDocTemplate(
    "data/pdfs/employee_handbook.pdf",
    pagesize=LETTER,
    topMargin=0.9 * inch,
    bottomMargin=0.9 * inch,
)
styles = getSampleStyleSheet()
title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=20, spaceAfter=20)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceBefore=18, spaceAfter=10)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=15, spaceAfter=10)

# A bank of realistic, varied paragraph content re-used and lightly varied
# across sections/chapters so the document has genuine bulk (50+ pages)
# without being meaningless repeated filler -- each chapter covers a
# distinct, plausible HR/operations topic.

CHAPTERS = [
    ("Chapter 1: Company Overview", [
        ("1.1 Our Mission", "Acme Corporation was founded in 2010 with a mission to provide reliable, "
         "affordable logistics software to small and mid-sized businesses. Over the past decade, the "
         "company has grown from a three-person startup to a workforce of over 800 employees across "
         "four countries, while maintaining the founding team's original commitment to transparent "
         "customer communication and rapid support response times."),
        ("1.2 Company Values", "Employees are expected to embody four core values in their daily work: "
         "Integrity, meaning honest communication even when the truth is inconvenient; Ownership, "
         "meaning taking responsibility for outcomes rather than just tasks; Curiosity, meaning a "
         "willingness to question existing processes and propose improvements; and Respect, meaning "
         "treating colleagues, customers, and vendors with basic professional courtesy at all times."),
        ("1.3 Organizational Structure", "The company is organized into five departments: Engineering, "
         "Customer Success, Sales, Finance, and People Operations. Each department is led by a Vice "
         "President who reports directly to the CEO. Department heads meet weekly to coordinate "
         "cross-functional priorities and resolve resourcing conflicts before they escalate."),
    ]),
    ("Chapter 2: Employment Policies", [
        ("2.1 Employment Classification", "Employees are classified as either Full-Time, Part-Time, or "
         "Contract. Full-Time employees work a minimum of 32 hours per week and are eligible for the "
         "full benefits package described in Chapter 4. Part-Time employees working under 32 hours "
         "are eligible for prorated paid time off but not health insurance. Contract employees are "
         "engaged through fixed-term agreements and are not eligible for any benefits described in "
         "this handbook, as their engagement is governed by a separate contractor agreement."),
        ("2.2 Probationary Period", "All new Full-Time and Part-Time employees serve a 90-calendar-day "
         "probationary period beginning on their official start date. During this period, either the "
         "employee or the company may terminate employment with five business days' written notice, "
         "rather than the standard notice period described in Section 2.4. Performance and cultural "
         "fit are formally reviewed by the employee's manager at the 45-day and 90-day marks."),
        ("2.3 Working Hours", "Standard working hours are 9:00 AM to 5:30 PM local time, Monday through "
         "Friday, with a 30-minute unpaid lunch break. Employees in Engineering and Customer Success "
         "may request a flexible schedule with manager approval, provided core hours of 11:00 AM to "
         "3:00 PM are maintained for team collaboration purposes."),
        ("2.4 Termination Notice", "Outside the probationary period, employees are asked to provide a "
         "minimum of two weeks' written notice before resigning. The company will provide a minimum of "
         "two weeks' notice or equivalent severance pay in the case of termination without cause. "
         "Termination for cause, including gross misconduct or repeated policy violations documented "
         "in writing, may occur without notice or severance."),
    ]),
    ("Chapter 3: Compensation", [
        ("3.1 Pay Schedule", "Employees are paid on a bi-weekly basis, with payments issued every other "
         "Friday via direct deposit. Pay stubs are available for download through the employee portal "
         "no later than the Wednesday preceding each pay date. Any discrepancies in pay must be "
         "reported to People Operations within 30 days of the affected pay date."),
        ("3.2 Performance Reviews and Raises", "Formal performance reviews occur twice annually, in "
         "January and July. Salary adjustments resulting from performance reviews take effect the "
         "first full pay period following the review, and are never applied retroactively. Employees "
         "who joined less than 90 days before a review cycle are excluded from that cycle and are "
         "reviewed at the following cycle instead."),
        ("3.3 Bonus Structure", "Annual performance bonuses, where applicable, are paid out in March "
         "based on the prior calendar year's performance and company-wide financial results. Bonus "
         "eligibility requires continuous employment through the payout date; employees who have "
         "resigned or been terminated prior to the March payout date, for any reason, forfeit that "
         "year's bonus regardless of performance rating."),
    ]),
    ("Chapter 4: Benefits", [
        ("4.1 Health Insurance", "Full-Time employees become eligible for company-subsidized health "
         "insurance on the first day of the month following 30 days of continuous employment. The "
         "company covers 80% of the premium for the employee and 50% of the premium for dependents "
         "under the standard PPO plan. A high-deductible HSA-eligible plan is also available at a "
         "lower premium share for employees who prefer it."),
        ("4.2 Paid Time Off", "Full-Time employees accrue 15 days of paid time off (PTO) per calendar "
         "year during their first two years of employment, increasing to 20 days beginning in year "
         "three. PTO accrues monthly and unused PTO up to a maximum of 5 days may be carried over into "
         "the following calendar year; any balance beyond 5 days is forfeited on December 31st unless "
         "local law requires otherwise."),
        ("4.3 Parental Leave", "Employees who have completed at least 12 months of continuous "
         "employment are eligible for 12 weeks of paid parental leave following the birth, adoption, "
         "or foster placement of a child. This leave may be taken continuously or split into two "
         "blocks within the first 12 months following the qualifying event, subject to manager "
         "approval of the split schedule."),
        ("4.4 Retirement Plan", "The company offers a 401(k) retirement plan with a 4% company match, "
         "vesting immediately upon the employee's first contribution. Enrollment is automatic 60 days "
         "after hire at a default contribution rate of 3%, unless the employee opts out or changes "
         "their contribution rate through the plan administrator's portal."),
    ]),
    ("Chapter 5: Workplace Conduct", [
        ("5.1 Anti-Harassment Policy", "The company maintains a zero-tolerance policy toward harassment "
         "of any kind, including but not limited to conduct based on race, gender, religion, age, "
         "disability, or sexual orientation. All reports of harassment are investigated by People "
         "Operations within 5 business days of being received, and reporting employees are protected "
         "from retaliation under this policy regardless of the investigation's outcome."),
        ("5.2 Conflict of Interest", "Employees must disclose any outside employment, board membership, "
         "or financial interest that could reasonably be perceived to conflict with the company's "
         "interests. Disclosures are reviewed by the employee's department head and, where necessary, "
         "by Legal, within 10 business days of submission."),
        ("5.3 Confidential Information", "Employees are prohibited from disclosing confidential company "
         "information, including customer data, unreleased product plans, and internal financial "
         "results, to any party outside the company without explicit written authorization from a "
         "Vice President or the CEO. This obligation continues for a period of two years following "
         "termination of employment, as detailed in each employee's signed confidentiality agreement."),
    ]),
    ("Chapter 6: Remote Work Policy", [
        ("6.1 Eligibility", "Employees in Engineering, Customer Success, and Finance are eligible to "
         "work remotely up to 3 days per week, subject to manager approval and demonstrated performance "
         "during an initial 90-day in-office ramp-up period for new hires. Sales and People Operations "
         "roles are generally expected to maintain an in-office presence of at least 4 days per week "
         "due to the collaborative and client-facing nature of the work."),
        ("6.2 Equipment and Expenses", "The company provides a one-time $500 home office stipend to "
         "employees approved for regular remote work, covering items such as monitors, chairs, and "
         "keyboards. Ongoing internet costs are not reimbursed. Equipment purchased through the "
         "stipend remains company property and must be returned or accounted for upon termination of "
         "employment."),
        ("6.3 Security Requirements", "Remote employees must connect to company systems exclusively "
         "through the company-issued VPN and must not access confidential systems from personal, "
         "unmanaged devices. Any suspected security incident involving a remote work device must be "
         "reported to IT Security within 1 hour of discovery."),
    ]),
]


EXTRA_CHAPTERS = [
    ("Chapter 9: Engineering Department Practices", "9.1 Code Review Standards",
     "All code changes must receive at least one approving review from a peer engineer before merging "
     "to the main branch. Reviewers are expected to respond to review requests within one business day. "
     "Critical hotfixes may bypass this requirement with Director-level sign-off, but must still receive "
     "a retroactive review within 48 hours of deployment."),
    ("Chapter 10: On-Call Rotation Policy", "10.1 Rotation Structure",
     "Engineers on the on-call rotation carry a company-issued pager for one full week at a time, "
     "rotating on Mondays at 10:00 AM. Engineers are compensated with a flat on-call stipend regardless "
     "of incident volume, plus overtime pay for any incident response occurring outside standard "
     "working hours, calculated in 30-minute increments."),
    ("Chapter 11: Sales Commission Structure", "11.1 Commission Calculation",
     "Sales representatives earn a base commission of 8% on new annual contract value and 4% on "
     "renewal contract value. Commissions are paid the month following the quarter in which a deal is "
     "signed and payment is received in full, not the month the deal is signed."),
    ("Chapter 12: Customer Success Escalation Procedures", "12.1 Escalation Tiers",
     "Customer issues are triaged into three tiers: Tier 1 (standard support, 24-hour response target), "
     "Tier 2 (account-impacting issues, 4-hour response target), and Tier 3 (business-critical outages, "
     "immediate response required). Tier 3 escalations automatically page the on-call engineer defined "
     "in Chapter 10."),
    ("Chapter 13: Finance Department Expense Policy", "13.1 Expense Reimbursement",
     "Business expenses under $75 may be submitted with a receipt through the expense portal for "
     "reimbursement within one pay cycle. Expenses over $75 require pre-approval from the employee's "
     "manager. Personal expenses submitted in error must be repaid to the company within 30 days of "
     "discovery."),
    ("Chapter 14: Business Travel Policy", "14.1 Booking and Approval",
     "All business travel must be booked through the company's designated travel portal and approved "
     "by the employee's manager at least 5 business days in advance, except in documented emergency "
     "circumstances. Economy class is standard for flights under 6 hours; premium economy is permitted "
     "for longer flights with manager approval."),
    ("Chapter 15: Equipment and Asset Policy", "15.1 Company-Issued Equipment",
     "All company-issued laptops, monitors, and mobile devices remain company property at all times "
     "and must be returned within 5 business days of employment termination. Lost or damaged equipment "
     "due to employee negligence may result in a deduction from final pay, where permitted by local law."),
    ("Chapter 16: Data Retention Policy", "16.1 Customer Data Retention",
     "Customer data is retained for the duration of the active contract plus 90 days following contract "
     "termination, after which it is permanently deleted from production systems unless a longer "
     "retention period is required by a signed data processing agreement or applicable regulation."),
    ("Chapter 17: Vendor Management Policy", "17.1 Vendor Approval Process",
     "New vendor relationships involving recurring spend over $10,000 annually require review by "
     "Finance and Legal prior to contract signature. Department heads may approve one-time purchases "
     "under $5,000 without additional review."),
    ("Chapter 18: Recruiting and Hiring Policy", "18.1 Interview Process Standards",
     "All candidates for Full-Time roles must complete a minimum of three interview stages: a recruiter "
     "screen, a hiring manager interview, and a panel interview with at least two team members. Hiring "
     "decisions require consensus from at least 80% of panel participants."),
    ("Chapter 19: Onboarding Checklist", "19.1 First Week Requirements",
     "New employees must complete mandatory compliance training, receive their equipment, and meet with "
     "their manager to review a 30-60-90 day plan within their first five business days. IT access "
     "provisioning must be completed prior to the employee's start date wherever possible."),
    ("Chapter 20: Offboarding Checklist", "20.1 Departure Procedures",
     "Upon an employee's departure, People Operations coordinates the return of company equipment, "
     "revocation of system access effective end-of-day on the employee's last working day, and "
     "processing of final pay in accordance with Chapter 3."),
    ("Chapter 21: Anti-Corruption Policy", "21.1 Gifts and Hospitality",
     "Employees may not accept gifts from vendors or customers valued over $100 without written "
     "disclosure to their manager and Legal. Cash gifts of any amount are strictly prohibited under "
     "all circumstances."),
    ("Chapter 22: Whistleblower Protection Policy", "22.1 Reporting Channels",
     "Employees may report suspected violations of law or company policy through their manager, People "
     "Operations, or an anonymous third-party hotline. Retaliation against a good-faith reporter is "
     "grounds for immediate termination of the retaliating party, regardless of position or tenure."),
    ("Chapter 23: Social Media Policy", "23.1 Personal Social Media Use",
     "Employees are free to discuss their work publicly but must not disclose confidential company "
     "information, speak on behalf of the company without authorization, or post content that could "
     "reasonably be seen as harassing toward colleagues or customers."),
    ("Chapter 24: Diversity and Inclusion Policy", "24.1 Commitment to Inclusive Hiring",
     "The company is committed to building a workforce reflective of the communities it serves. "
     "Hiring panels are required to include at least one member from outside the immediately hiring "
     "team to reduce the risk of insular hiring patterns."),
    ("Chapter 25: Training and Development Policy", "25.1 Annual Learning Stipend",
     "Full-Time employees receive an annual $1,500 learning and development stipend, which may be used "
     "for courses, conferences, books, or certifications directly relevant to their role, subject to "
     "manager approval. Unused stipend does not carry over to the following year."),
    ("Chapter 26: Performance Improvement Plans", "26.1 PIP Process",
     "Employees whose performance falls below expectations over two consecutive review cycles may be "
     "placed on a formal 30-60-90 day Performance Improvement Plan, with clearly defined success "
     "criteria reviewed jointly by the employee, manager, and People Operations."),
    ("Chapter 27: Workplace Health and Safety", "27.1 Incident Reporting",
     "Any workplace injury, regardless of severity, must be reported to a manager and People "
     "Operations within 24 hours. Failure to report an injury promptly may affect eligibility for "
     "workers' compensation coverage under applicable local law."),
    ("Chapter 28: Intellectual Property Assignment", "28.1 Work Product Ownership",
     "All work product created by an employee within the scope of their employment, using company "
     "resources or during company time, is the sole property of the company, as detailed in each "
     "employee's signed IP assignment agreement executed at the time of hire."),
    ("Chapter 29: Non-Solicitation Policy", "29.1 Customer and Employee Non-Solicitation",
     "For a period of 12 months following termination of employment, former employees may not solicit "
     "the company's customers or employees for the benefit of a competing business, to the extent "
     "permitted by applicable state law."),
    ("Chapter 30: Drug and Alcohol Policy", "30.1 Workplace Substance Policy",
     "Employees may not report to work or perform job duties while impaired by alcohol or illegal "
     "substances. Moderate alcohol consumption at approved company events is permitted, provided "
     "employees do not operate a vehicle or perform safety-sensitive duties while impaired."),
    ("Chapter 31: Immigration Sponsorship Policy", "31.1 Visa Sponsorship Eligibility",
     "The company may sponsor employment-based visa petitions for roles where a qualified candidate "
     "cannot be identified through standard recruiting channels, subject to budget approval from "
     "Finance and legal review of the specific case."),
    ("Chapter 32: Relocation Assistance Policy", "32.1 Relocation Reimbursement",
     "Employees relocating for an approved role change may receive up to $5,000 in relocation expense "
     "reimbursement, including moving company fees and temporary housing, subject to a 12-month "
     "repayment obligation if the employee voluntarily resigns within that period."),
    ("Chapter 33: Overtime and Timekeeping Policy", "33.1 Non-Exempt Employee Timekeeping",
     "Non-exempt employees must record all hours worked, including any time worked outside standard "
     "hours, through the company timekeeping system. Overtime must be pre-approved by a manager "
     "wherever operationally feasible, though all hours worked must still be recorded and paid "
     "regardless of prior approval."),
    ("Chapter 34: Company Property Access Policy", "34.1 Badge Access",
     "Physical badge access to company offices is provisioned based on role and location and is "
     "automatically revoked at end-of-day on an employee's last working day. Lost badges must be "
     "reported to Facilities immediately to prevent unauthorized access."),
    ("Chapter 35: Bereavement Leave Policy", "35.1 Leave Entitlement",
     "Employees are entitled to up to 5 paid days of bereavement leave for the death of an immediate "
     "family member and up to 2 paid days for the death of an extended family member, in addition to "
     "any leave required under applicable local law."),
    ("Chapter 36: Jury Duty and Civic Leave", "36.1 Jury Duty Pay",
     "Employees summoned for jury duty receive full base pay for up to 10 business days of service, "
     "with documentation from the court required. Any pay received directly from the court for jury "
     "service may be retained by the employee in addition to their regular pay."),
    ("Chapter 37: Workplace Accommodation Policy", "37.1 Reasonable Accommodation Requests",
     "Employees requesting workplace accommodation for a disability, religious practice, or pregnancy "
     "should submit a request to People Operations, which will engage in an interactive process to "
     "identify a reasonable accommodation within 10 business days wherever operationally feasible."),
    ("Chapter 38: Internal Transfer Policy", "38.1 Transfer Eligibility",
     "Employees become eligible to apply for internal transfers after completing 12 months in their "
     "current role. Managers may not block an employee's internal transfer application but may request "
     "a transition period of up to 30 days to identify a replacement."),
    ("Chapter 39: Referral Bonus Program", "39.1 Referral Bonus Amounts",
     "Employees who refer a candidate who is hired and completes 90 days of employment receive a $2,000 "
     "referral bonus for standard roles and $4,000 for designated hard-to-fill Engineering roles, paid "
     "in the pay cycle following the referred employee's 90-day mark."),
    ("Chapter 40: Sabbatical Leave Policy", "40.1 Sabbatical Eligibility",
     "Employees who reach 5 years of continuous tenure are eligible for a 4-week paid sabbatical leave, "
     "renewable every subsequent 5 years. Sabbatical timing must be coordinated with the employee's "
     "manager at least 90 days in advance to ensure adequate team coverage."),
    ("Chapter 41: Company Equipment Software Policy", "41.1 Approved Software Installation",
     "Employees may only install software on company-issued devices that has been approved by IT "
     "Security through the internal software request portal. Unapproved software found during routine "
     "security audits must be removed within 48 hours of notification."),
    ("Chapter 42: Meeting Culture Guidelines", "42.1 Meeting Scheduling Norms",
     "Meetings longer than 30 minutes must include a written agenda circulated at least 24 hours in "
     "advance. No-meeting Wednesdays are observed company-wide to protect focused work time, except "
     "for customer-facing meetings scheduled at the customer's request."),
    ("Chapter 43: Performance Bonus Clawback Policy", "43.1 Clawback Circumstances",
     "The company reserves the right to reclaim a previously paid performance bonus if it is later "
     "determined the bonus was based on materially inaccurate financial reporting or data, provided "
     "such reclamation is pursued within 24 months of the original payout."),
    ("Chapter 44: Company Credit Card Policy", "44.1 Card Issuance and Limits",
     "Company credit cards are issued to employees with a recurring business need for travel or "
     "vendor payments, subject to Finance approval. Default credit limits are set at $5,000 per month "
     "and may be increased with Director-level sign-off for specific projects."),
    ("Chapter 45: Employee Stock Option Policy", "45.1 Vesting Schedule",
     "Stock options granted at hire vest over a standard 4-year schedule with a 1-year cliff, meaning "
     "no options vest until the employee's first anniversary, after which 25% vest immediately and the "
     "remainder vests monthly over the following 36 months."),
    ("Chapter 46: Internal Communication Standards", "46.1 Response Time Expectations",
     "Employees are expected to respond to internal messages during standard working hours within 4 "
     "hours for non-urgent matters and within 30 minutes for messages marked urgent, provided the "
     "employee is not in a scheduled meeting or approved focus block."),
    ("Chapter 47: Company Holiday Schedule", "47.1 Observed Holidays",
     "The company observes 10 paid holidays annually, including New Year's Day, Memorial Day, "
     "Independence Day, Labor Day, Thanksgiving Day and the following Friday, and a company-wide "
     "winter break spanning the final week of December, communicated annually by People Operations."),
    ("Chapter 48: Exit Interview Policy", "48.1 Exit Interview Process",
     "All departing employees are offered a voluntary exit interview with People Operations, conducted "
     "either in person or via a written survey, within the employee's final two weeks of employment, "
     "to gather feedback on their experience and reasons for departure."),
    ("Chapter 49: Employee Referral Data Handling", "49.1 Referral Confidentiality",
     "Employee referral submissions, including any notes provided about the candidate, are treated as "
     "confidential HR records and are not shared outside the recruiting team and the relevant hiring "
     "manager without the referring employee's consent."),
    ("Chapter 50: Company Vehicle Policy", "50.1 Fleet Vehicle Use",
     "Employees assigned a company vehicle for field or delivery roles must maintain a valid driver's "
     "license and clean driving record, verified annually by People Operations, and must report any "
     "moving violations incurred while operating the vehicle within 5 business days."),
    ("Chapter 51: Professional Licensing Support", "51.1 License Renewal Reimbursement",
     "Employees whose roles require a professional license or certification are reimbursed for renewal "
     "fees and required continuing education, up to $1,000 annually, upon submission of proof of "
     "renewal to People Operations."),
    ("Chapter 52: Workplace Visitor Policy", "52.1 Visitor Sign-In Requirements",
     "All visitors to company offices must sign in at reception, receive a visitor badge, and be "
     "escorted by an employee at all times while in non-public areas of the building, in accordance "
     "with the security requirements outlined in Chapter 34."),
    ("Chapter 53: Company Asset Disposal Policy", "53.1 Retired Equipment Disposal",
     "Retired company equipment containing storage media must be securely wiped by IT in accordance "
     "with the data retention standards in Chapter 16 before being donated, resold, or recycled through "
     "an approved e-waste vendor."),
    ("Chapter 54: Emergency Preparedness Policy", "54.1 Office Emergency Procedures",
     "Each office location maintains a posted emergency evacuation plan and conducts a fire drill at "
     "least twice annually. Employees are expected to familiarize themselves with the nearest exit and "
     "designated assembly point upon starting work at a new office location."),
    ("Chapter 55: Handbook Acknowledgment", "55.1 Annual Acknowledgment Requirement",
     "All employees are required to electronically acknowledge that they have read and understood this "
     "handbook within 5 business days of hire, and again annually thereafter following any material "
     "revision, through the employee portal."),
]


def build():
    flow = []
    flow.append(Paragraph("ACME CORPORATION", title_style))
    flow.append(Paragraph("Employee Handbook", ParagraphStyle("subtitle", parent=styles["Heading2"], fontSize=14)))
    flow.append(Paragraph("Effective Date: January 1, 2026 | Version 4.0", body))
    flow.append(Spacer(1, 20))
    flow.append(Paragraph(
        "This handbook describes the policies, benefits, and expectations that apply to all "
        "employees of Acme Corporation. It supersedes all previous versions of the handbook. "
        "Employees are responsible for reading and understanding its contents; questions should "
        "be directed to People Operations.", body))
    flow.append(PageBreak())

    for chapter_title, sections in CHAPTERS:
        flow.append(Paragraph(chapter_title, h1))
        for section_title, section_text in sections:
            flow.append(Paragraph(section_title, h2))
            flow.append(Paragraph(section_text, body))
        flow.append(PageBreak())

    for chapter_title, section_title, section_text in EXTRA_CHAPTERS:
        flow.append(Paragraph(chapter_title, h1))
        flow.append(Paragraph(section_title, h2))
        flow.append(Paragraph(section_text, body))
        flow.append(PageBreak())

    # A real table -- to exercise/expose how plain-text extraction handles
    # tabular data (usually poorly, without special handling)
    flow.append(Paragraph("Chapter 7: PTO Accrual Reference Table", h1))
    flow.append(Paragraph(
        "The table below summarizes annual PTO accrual by tenure band, for quick reference "
        "alongside the detailed policy in Section 4.2.", body))
    table_data = [
        ["Tenure", "Annual PTO Days", "Monthly Accrual", "Max Carryover"],
        ["0-2 years", "15 days", "1.25 days", "5 days"],
        ["3-5 years", "20 days", "1.67 days", "5 days"],
        ["6-10 years", "22 days", "1.83 days", "7 days"],
        ["10+ years", "25 days", "2.08 days", "10 days"],
    ]
    tbl = Table(table_data, colWidths=[1.4 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    flow.append(tbl)
    flow.append(PageBreak())

    # Pad out to comfortably exceed 50 pages with additional realistic
    # sub-sections (FAQ-style), since ~7 chapters of dense paragraphs alone
    # land close to the 50-page line depending on PDF layout/font metrics.
    flow.append(Paragraph("Chapter 8: Frequently Asked Questions", h1))
    faqs = [
        ("Can I take PTO during my probationary period?",
         "Yes, but PTO taken during the first 90 days is unpaid unless the employee has a positive "
         "accrued balance from a previous employer transferred under an acquisition agreement."),
        ("What happens to unused PTO if I am terminated?",
         "Unused, accrued PTO is paid out at the employee's final base salary rate within the final "
         "paycheck, in accordance with applicable state and local law."),
        ("Can I switch from the PPO plan to the HSA plan mid-year?",
         "Mid-year plan changes are only permitted following a qualifying life event, such as marriage, "
         "birth of a child, or loss of other coverage, and must be requested within 30 days of the event."),
        ("Is remote work eligibility permanent once approved?",
         "No. Remote work approval is reviewed at each formal performance review cycle and may be "
         "adjusted based on team needs, individual performance, or role changes."),
        ("Who do I contact if I believe I was underpaid?",
         "Pay discrepancies should be reported directly to People Operations within 30 days of the "
         "affected pay date, as described in Section 3.1."),
        ("Does the parental leave policy apply to adoptive parents?",
         "Yes. The 12-week paid parental leave policy applies equally to birth, adoption, and foster "
         "placement, as stated in Section 4.3."),
        ("Can contract employees convert to Full-Time status?",
         "Contract employees may be considered for Full-Time conversion at the discretion of their "
         "hiring manager and People Operations, typically after a minimum of 6 months of engagement."),
        ("What is the process for disclosing a conflict of interest?",
         "Employees must submit a written disclosure to their department head, who will route it to "
         "Legal if necessary, per the process described in Section 5.2."),
    ]
    for q, a in faqs:
        flow.append(Paragraph(f"Q: {q}", h2))
        flow.append(Paragraph(f"A: {a}", body))

    doc.build(flow)


if __name__ == "__main__":
    build()
    print("PDF generated.")
