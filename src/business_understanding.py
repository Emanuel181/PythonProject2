# src/business_understanding.py

import logging # Imports the standard Python logging library for logging messages.

from src.config import BENIGN_LABEL, TARGET_COLUMN # Imports specific constants (BENIGN_LABEL, TARGET_COLUMN) from the project's configuration file.
from src.utils import setup_logging # Imports the setup_logging function from the project's utility module to configure logging.

logger = setup_logging()  # Get the configured logger # Initializes a logger instance by calling setup_logging(), which configures how messages are logged (e.g., to console, file).


def business_understanding_phase() -> None:
    """
    Defines the business objectives, problem statement, and key success criteria for the project.
    This phase, part of the CRISP-DM lifecycle, is crucial for aligning the technical work
    with the real-world problem of DDoS and Botnet attack detection.
    """
    logger.info("--- CRISP-DM Phase 1: Business Understanding ---") # Logs an informational message indicating the start of the Business Understanding phase.

    logger.info(
        "Project Objective: To develop a highly accurate and robust machine learning model "
        "for the identification and classification of Distributed Denial of Service (DDoS) "
        "and Botnet attacks within network traffic data from the CIC-IDS2017 dataset. "
        "The ultimate goal is to enhance network security posture and enable proactive, "
        "automated threat response mechanisms."
    ) # Logs the main objective of the project, detailing the type of attacks to detect, the dataset used, and the ultimate security goal.

    logger.info(
        "\nProblem Statement: Modern cyber threats, particularly DDoS and Botnet attacks, "
        "are increasingly sophisticated and volumetric, posing severe risks to network "
        "availability, data integrity, and service continuity. Traditional signature-based "
        "intrusion detection systems (IDS) are often insufficient against zero-day or "
        "polymorphic attacks. This project addresses the critical need for an intelligent, "
        "adaptive system capable of detecting these evolving threats by leveraging "
        "machine learning on comprehensive network flow features."
    ) # Logs the problem statement, explaining the nature of modern cyber threats and the limitations of traditional IDSs, justifying the need for this ML-based solution.

    logger.info("\nBig Data Challenges Addressed (inherent in network traffic):") # Logs a header for the Big Data challenges.
    logger.info(
        "  - Volume: Network traffic datasets are inherently large, requiring efficient "
        "    data loading, storage, and processing techniques. CIC-IDS2017, while not "
        "    petabyte-scale, provides a substantial volume of data to demonstrate "
        "    handling of large datasets."
    ) # Logs the 'Volume' challenge, explaining the large size of network traffic data and how the project addresses it.
    logger.info(
        "  - Variety: Network flow data consists of diverse feature types (numerical, "
        "    categorical, temporal) and varying attack patterns, necessitating robust "
        "    preprocessing and versatile machine learning algorithms."
    ) # Logs the 'Variety' challenge, describing the diverse types of data and the need for adaptable algorithms.
    logger.info(
        "  - Velocity (potential): While this project uses historical data, the context "
        "    of network security implies a need for real-time or near real-time detection, "
        "    which influences model choice and deployment considerations."
    ) # Logs the 'Velocity' challenge, acknowledging the need for real-time detection despite using historical data.

    logger.info("\nKey Success Criteria (for an 'Excellent' grade):") # Logs a header for the project's success criteria.
    logger.info(
        "  1. Model Performance: Achieve excellent classification performance across all "
        "     attack types (DDoS, Botnet, etc.) and 'BENIGN' traffic, particularly focusing "
        "     on high Recall for attack classes to minimize false negatives (missed attacks). "
        "     Metrics like F1-score, Precision-Recall AUC, and ROC AUC will be critical, "
        "     with a target F1-score > 0.90 for major attack types."
    ) # Logs the first success criterion: Model Performance, detailing specific metrics and targets.
    logger.info(
        "  2. Robustness: Model should demonstrate stability and generalize well to unseen "
        "     data, validated through cross-validation and rigorous test set evaluation."
    ) # Logs the second success criterion: Robustness, emphasizing generalization and validation methods.
    logger.info(
        "  3. Interpretability (if applicable): Identify and analyze the most influential "
        "     features contributing to attack detection, providing actionable insights for "
        "     security analysts."
    ) # Logs the third success criterion: Interpretability, focusing on understanding model decisions.
    logger.info(
        "  4. Scalability Awareness: Demonstrate consideration for handling large volumes "
        "     of data (e.g., efficient data loading, memory optimization) and discuss "
        "     scalability to streaming data."
    ) # Logs the fourth success criterion: Scalability Awareness, including data handling and future streaming.
    logger.info(
        "  5. Code Quality: Professionally written, well-documented (docstrings, comments), "
        "     modular, and extensible Python code with appropriate error handling and type hints."
    ) # Logs the fifth success criterion: Code Quality, detailing aspects like documentation and modularity.
    logger.info(
        "  6. Comprehensive Analysis: In-depth descriptive analytics, comparative model "
        "     evaluation, and thorough interpretation of results, including limitations and future work."
    ) # Logs the sixth success criterion: Comprehensive Analysis, covering reporting and future considerations.

    logger.info(
        f"\nExpected Output: A production-ready (conceptually) trained and evaluated ML model, "
        "a detailed analysis report, and a robust, well-structured codebase."
    ) # Logs the expected deliverables of the project.
    logger.info("-----------------------------------------------\n") # Logs a separator to mark the end of the phase's output.



if __name__ == "__main__": # Checks if the script is being run directly (not imported as a module).
    # This block allows running the phase script independently for testing/development.
    business_understanding_phase() # Calls the business_understanding_phase function when the script is executed directly.