from monopriors.apis.promptda_polycam import PDAPolycamConfig, pda_polycam_inference
import tyro


if __name__ == "__main__":
    pda_polycam_inference(tyro.cli(PDAPolycamConfig))
