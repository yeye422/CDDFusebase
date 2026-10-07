"""Stage-wise Optical-SAR trainer entry point.

Stage 0 currently exposes the loss-matched B1 baseline through ``train_OS``.
Later A1/A2/A3 model variants can extend this entry point without changing the
frozen experiment command family.
"""

from train_OS import main


if __name__ == "__main__":
    main()
