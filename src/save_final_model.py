"""Compatibility entry point for the current production artifact audit.

The previous version retrained a retired Gradient Boosting model and wrote
metadata that did not describe the deployed CatBoost bundle. Run the verifier
to validate the current artifact without replacing it.
"""

from verify_final_model import main


if __name__ == "__main__":
    main()
