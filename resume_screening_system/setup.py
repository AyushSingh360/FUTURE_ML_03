from setuptools import setup

setup(
    name="resume_screening_system",
    version="0.1.0",
    # List each standalone module file in src/
    py_modules=[
        "data_loader",
        "text_preprocessing",
        "skill_extraction",
        "vectorization",
        "similarity_scoring",
        "candidate_ranking",
        "skill_gap_analysis",
    ],
    # Tell setuptools that the root for these modules is src/
    package_dir={"": "src"},
)
