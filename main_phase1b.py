from src.generate_company import generate_company, save_company
from src.generate_prompt_sessions import generate_prompt_sessions
from src.label_leakage import label_prompt_dataset
from src.dataset_statistics import compute_dataset_statistics
from src.label_validation_and_figures import generate_label_validation_figures
from src.profile_calibration import generate_profile_calibration_outputs


PROFILES = [
    ("realistic_mixture", 43),
    ("balanced", 42),
    ("risk_enriched", 44),
]


if __name__ == "__main__":
    company = generate_company(
        name="TechNova Ltd",
        num_employees=150,
        num_projects=30,
        num_vendors=25,
        num_systems=25,
        seed=42,
    )
    save_company(company, path="data/company.json")
    print("Company data saved.")

    for profile, seed in PROFILES:
        base_dir = f"data/profiles/{profile}"
        print(f"Generating {profile}")

        generate_prompt_sessions(
            company_path="data/company.json",
            output_path=f"{base_dir}/prompt_sessions.json",
            num_sessions=5000,
            seed=seed,
            profile=profile,
        )

        label_prompt_dataset(
            input_path=f"{base_dir}/prompt_sessions.json",
            output_path=f"{base_dir}/prompt_sessions_labeled.json",
            audit_output_path=f"{base_dir}/labelling_audit.json",
        )

        compute_dataset_statistics(
            prompt_path=f"{base_dir}/prompt_sessions.json",
            labelled_path=f"{base_dir}/prompt_sessions_labeled.json",
            output_path=f"{base_dir}/dataset_statistics.json",
        )

        generate_label_validation_figures(
            labelled_path=f"{base_dir}/prompt_sessions_labeled.json",
            validation_output_path=f"{base_dir}/label_validation_statistics.json",
            figure_dir=f"figures/profiles/{profile}",
        )

    generate_profile_calibration_outputs(base_dir="data/profiles")
