"""
Sample Documents Generator
Generates realistic multi-page PDF and TXT test documents for SmartRAG.
Matches the company policy topics from the assignment reference screenshots.
"""

import os
import pymupdf


def generate_sample_pdf(output_path: str):
    """Generates a professional 3-page Employee Handbook PDF using PyMuPDF."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = pymupdf.open()

    # --- PAGE 1: Leave Policy & Working Hours ---
    page1 = doc.new_page(width=595, height=842)  # Standard A4 size
    rect1 = pymupdf.Rect(50, 50, 545, 792)

    page1_text = """Acme Global Technologies - Employee Handbook
Document Version: 4.2 | Classification: Internal Confidential
Page 1: General Employment & Leave Policies

1. Welcome to Acme Global Technologies
Welcome to Acme Global Technologies. This handbook outlines our operational policies, rights, responsibilities, and benefits available to all full-time team members.

2. Comprehensive Leave Policy
All full-time employees are entitled to 20 days of annual leave, 10 days of sick leave, and 5 days of casual leave per calendar year. Leave quotas are credited on January 1st annually and prorated for mid-year joiners.

Application Procedures:
Leave requests must be submitted through the internal HR portal at least 3 working days in advance and approved by the reporting manager. In cases of sudden medical emergency, notice must be given within 24 hours of absence.

Medical Certification:
Any sick leave extending beyond 2 consecutive working days requires a verified medical certificate from a licensed healthcare practitioner before returning to duty.

Unused Leave Carryover:
A maximum of 8 days of unused annual leave may be carried forward into the next calendar year. Any additional unused leave beyond 8 days lapses automatically on December 31st and cannot be encashed.

3. Standard Working Hours & Attendance
Standard office working hours are from 9:00 AM to 6:00 PM, Monday through Friday, including a mandatory 60-minute lunch break. All employees must log their attendance via the biometric access system or the remote single sign-on (SSO) gateway.
"""
    page1.insert_text(pymupdf.Point(50, 70), page1_text, fontsize=11, fontname="helv", color=(0.1, 0.1, 0.15))

    # --- PAGE 2: Remote Work & Hardware Security ---
    page2 = doc.new_page(width=595, height=842)
    page2_text = """Acme Global Technologies - Employee Handbook
Document Version: 4.2 | Classification: Internal Confidential
Page 2: Remote Work & Information Security

4. Remote Work and Work From Home (WFH) Policy
Based on company policy, employees may work from home up to 3 days per week, subject to prior manager approval and business operational needs.

Permanent Remote Work Restriction:
Permanent remote work is strictly not permitted for core engineering, operations, and customer-facing teams. Employees must be physically present at their assigned base office for at least 2 days each work week.

Core Hours & Communication:
When working remotely, team members are required to maintain core availability between 10:00 AM and 4:00 PM local time on Slack and corporate email. All video conference calls must be conducted with the camera enabled.

5. IT Equipment & Data Security Standards
All corporate computing equipment (laptops, monitors, security tokens) remain the exclusive property of Acme Global Technologies. Equipment is provisioned strictly for official company work.

Security Compliance Requirements:
- Employees must connect exclusively through the company encrypted VPN when accessing internal code repositories and databases.
- Installing unauthorized third-party software, file sharing tools, or mining software is strictly prohibited.
- Clean Desk Standard: Sensitive physical documents, client files, and removable storage media must be locked in designated office pedestals or destroyed using cross-cut shredders.
"""
    page2.insert_text(pymupdf.Point(50, 70), page2_text, fontsize=11, fontname="helv", color=(0.1, 0.1, 0.15))

    # --- PAGE 3: Code of Conduct & Ethics ---
    page3 = doc.new_page(width=595, height=842)
    page3_text = """Acme Global Technologies - Employee Handbook
Document Version: 4.2 | Classification: Internal Confidential
Page 3: Professional Conduct & Grievance Mechanism

6. Workplace Code of Conduct & Ethics
Acme Global Technologies fosters an inclusive, safe, and respectful work environment for everyone. We enforce a zero-tolerance policy against workplace harassment, bullying, discrimination, and retaliation.

Conflicts of Interest:
Employees may not engage in outside consulting, freelance commercial software development, or moonlighting activities that conflict with Acme Global Technologies' business interests.

Data Protection & Non-Disclosure:
Proprietary software algorithms, client lists, financial records, and design documents are protected under the Employee Non-Disclosure Agreement (NDA). Storing company assets on unapproved personal cloud accounts (e.g., personal Google Drive or Dropbox) is a severe security violation resulting in immediate disciplinary review.

7. Whistleblower & Grievance Redressal
Employees who witness misconduct or ethical violations may file an anonymous confidential report through the whistleblower portal at ethics@acmetech.internal.

Escalation Timeline:
All formal grievances submitted to the HR Ombudsman will be acknowledged within 48 business hours and formally resolved within 14 working days.
"""
    page3.insert_text(pymupdf.Point(50, 70), page3_text, fontsize=11, fontname="helv", color=(0.1, 0.1, 0.15))

    doc.save(output_path)
    doc.close()
    print(f"Successfully generated sample PDF at: {output_path}")


def generate_sample_txt(output_path: str):
    """Generates a comprehensive HR Guidelines TXT file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    content = """HR GUIDELINES & COMPENSATION FRAMEWORK
Acme Global Technologies | Human Resources Division

SECTION 1: PROBATIONARY PERIOD & EVALUATION
1.1 All newly hired employees undergo a standard 6-month probationary period starting on their official date of joining.
1.2 A formal performance appraisal is conducted at the end of the 5th month by the department lead.
1.3 Confirmation of permanent employment is contingent upon receiving a 'Satisfactory' or 'Exceeds Expectations' rating.
1.4 During probation, the notice period for separation from either party is 15 calendar days; following permanent confirmation, the notice period is 60 calendar days.

SECTION 2: HEALTH & WELLNESS BENEFITS
2.1 Comprehensive Health Insurance: Every confirmed full-time employee is covered under the corporate group health insurance policy up to $500,000 per family unit annually.
2.2 The insurance policy covers hospitalization, emergency surgical interventions, and prescription medication costs.
2.3 Mental Health Allowance: Employees are entitled to an annual wellness stipend of $1,200 for certified mental health counseling and fitness memberships.
2.4 Annual Executive Health Checkup: All employees above 30 years of age are eligible for a fully sponsored comprehensive annual health screening at partner clinics.

SECTION 3: ANNUAL PERFORMANCE BONUS
3.1 The company operates an annual discretionary performance bonus scheme distributed in March of each financial year.
3.2 Bonus payouts range from 10% to 30% of base annual salary, benchmarked against individual KPI attainment and global corporate profitability.
3.3 Employees must be actively employed on the date of payout to receive the bonus.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"Successfully generated sample TXT at: {output_path}")


if __name__ == "__main__":
    pdf_dest = os.path.join("sample_documents", "Employee_Handbook.pdf")
    txt_dest = os.path.join("sample_documents", "HR_Guidelines.txt")
    generate_sample_pdf(pdf_dest)
    generate_sample_txt(txt_dest)

