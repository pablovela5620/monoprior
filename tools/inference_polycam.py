from monopriors.apis.polycam_inference import polycam_inference, PolycamConfig
import tyro


if __name__ == "__main__":
    polycam_inference(tyro.cli(PolycamConfig))
