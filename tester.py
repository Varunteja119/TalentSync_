# Sample resume-like input payload used for testing the resume generator.
sample_data = {
    # Basic contact/profile details.
    "name": "Alex Reed",
    "email": "alex.reed@email.com",
    "phone": "(555) 012-3456",
    "location": "Austin, TX",

    # Professional summary shown near the top of the resume.
    "summary": "Project Manager with 8+ years of experience in software development life cycles (SDLC). Proven track record of delivering cross-functional projects 15% under budget while maintaining high quality standards.",

    # Core skills list.
    "skills": [
        "Python", "SQL", "Agile Methodology", "Project Management", 
        "Data Analysis", "Cloud Computing (AWS)", "CRM Systems"
    ],

    # Work history entries (most recent first).
    "experience": [
        {
            # Job title for this role.
            "role": "Senior Project Lead",
            # Employer name.
            "company": "TechStream Solutions",
            # Date range in this position.
            "years": "2020 – Present",
            # Achievement/responsibility bullet points.
            "description": [
                "Led a team of 15 developers to launch a SaaS platform ahead of schedule.",
                "Implemented automated reporting tools, reducing manual entry by 20 hours per week.",
                "Managed a $2M annual budget with strict adherence to financial compliance."
            ]
        },
        {
            # Previous role details.
            "role": "Operations Analyst",
            "company": "Global Logistics Corp",
            "years": "2016 – 2020",
            "description": [
                "Optimized supply chain workflows, resulting in a 10% increase in distribution speed.",
                "Analyzed KPI data to identify bottlenecks in the regional delivery system."
            ]
        }
    ],
    # Academic background and certifications.
    "education": [
        {
            # Degree name.
            "degree": "B.S. in Computer Science",
            # Institution granting the degree.
            "school": "University of Texas at Austin"
        },
        {
            # Professional certification.
            "degree": "PMP Certification",
            "school": "Project Management Institute"
        }
    ]
}

# Generate a resume document from the sample data.
# Output file will be created as: Alex_Reed_Resume.docx
resume(sample_data, "Alex_Reed_Resume.docx")
